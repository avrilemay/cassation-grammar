#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune a doctrinal-family baseline (see baselines/README.md for the published results).

The side flag on the command line chooses the task: accepted grounds (one
complaint against the ruling) get exactly one of six doctrinal families, while
rejected grounds can carry more than one of three rejection families, summary
rejection (RNSM), inadmissibility (IRREC), and rejection on the merits (FOND).
Long court answers are truncated by keeping their start and their end, since
the closing formula that carries the decision usually sits at the end.

Usage: python3 train.py --model {juribert,camembertv2} --side {accepte,rejete} [--smoke]
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from _config import load_config, resolve  # noqa: E402

FAMILLES6 = ["VIOLATION", "MBL", "VICE_MOTIFS", "EXCES_OFFICE", "DENATURATION", "VICE_FORME"]
BORE3 = ["RNSM", "IRREC", "FOND"]

BATCH = 8
GRAD_ACCUM = 4
LR = 2e-5
SEED = 20260729


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=["juribert", "camembertv2"],
                         help="Pretrained model to fine-tune, resolved through config.yaml::baselines.models")
    parser.add_argument("--side", required=True, choices=["accepte", "rejete"])
    parser.add_argument("--epochs", type=int, default=None,
                         help="Max epochs (default: 4 accepted / 3 rejected, as published)")
    parser.add_argument("--data-dir", default=None, help="Override baselines.out_dir from config.yaml")
    parser.add_argument("--checkpoints-dir", default=None,
                         help="Override baselines.checkpoints_dir from config.yaml")
    parser.add_argument("--smoke", action="store_true",
                         help="Tiny CPU run (few dozen rows, 1 epoch) to check the mechanics, "
                              "not to reproduce the published numbers")
    return parser.parse_args()


def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bore_label(r):
    return [1.0 if f in r["bore"] else 0.0 for f in BORE3]


def make_encode(tok, max_len):
    pad = tok.pad_token_id
    n_trunc = Counter()

    def encode(texts):
        """Tokenize each text, keeping its start and its end (CLS at the
        start, SEP at the end) and dropping the middle when it is too long."""
        enc = tok(list(texts), truncation=False)["input_ids"]
        out = []
        for x in enc:
            if len(x) > max_len:
                head = (max_len - 2) // 2
                tail = max_len - 2 - head
                x = x[:1 + head] + x[-(tail + 1):]
                n_trunc["truncated"] += 1
            n_trunc["total"] += 1
            out.append(x)
        m = max(len(x) for x in out)
        ids = torch.full((len(out), m), pad, dtype=torch.long)
        att = torch.zeros((len(out), m), dtype=torch.long)
        for i, x in enumerate(out):
            ids[i, :len(x)] = torch.tensor(x)
            att[i, :len(x)] = 1
        return {"input_ids": ids, "attention_mask": att}

    return encode, n_trunc


class GroundsDataset(Dataset):
    def __init__(self, rows, side):
        self.rows = rows
        self.side = side

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        label = FAMILLES6.index(r["famille"]) if self.side == "accepte" else bore_label(r)
        return r["text"] or "", label


def make_collate(encode, side):
    def collate(batch):
        texts, y = zip(*batch)
        enc = encode(texts)
        y = torch.tensor(y, dtype=torch.long if side == "accepte" else torch.float)
        return enc, y
    return collate


class Model(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()
        self.trunk = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(self.trunk.config.hidden_size, n_classes)

    def forward(self, enc):
        return self.head(self.drop(self.trunk(**enc).last_hidden_state[:, 0]))


def predict_sets(logits):
    """Rejected side: sigmoid >= 0.5, never empty (argmax fallback: a rejection
    always carries at least one family)."""
    p = torch.sigmoid(logits)
    out = []
    for row in p:
        fams = [BORE3[i] for i in range(3) if row[i] >= 0.5]
        if not fams:
            fams = [BORE3[int(row.argmax())]]
        out.append(sorted(fams))
    return out


def prf(n_ok, n_true, n_pred):
    prec = n_ok / max(1, n_pred)
    rec = n_ok / max(1, n_true)
    return {"precision": round(prec, 3), "recall": round(rec, 3),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 3)}


@torch.no_grad()
def run_eval(model, dl, side, device):
    model.eval()
    if side == "accepte":
        conf = {f: Counter() for f in FAMILLES6}
        for enc, y in dl:
            enc = {k: v.to(device) for k, v in enc.items()}
            for p, t in zip(model(enc).argmax(-1).cpu(), y):
                conf[FAMILLES6[int(t)]][FAMILLES6[int(p)]] += 1
        out, ok_tot, n_tot = {}, 0, 0
        for f in FAMILLES6:
            n = sum(conf[f].values())
            ok = conf[f][f]
            ok_tot += ok
            n_tot += n
            out[f] = {"n": n, **prf(ok, n, sum(conf[g][f] for g in FAMILLES6))}
        return {"par_famille": out, "confusion": {f: dict(conf[f]) for f in FAMILLES6},
                "macro_f1": round(float(np.mean([out[f]["f1"] for f in FAMILLES6])), 3),
                "micro_acc": round(ok_tot / max(1, n_tot), 3)}
    ok_ex = n_ex = 0
    st = {f: Counter() for f in BORE3}
    for enc, y in dl:
        enc = {k: v.to(device) for k, v in enc.items()}
        preds = predict_sets(model(enc).float().cpu())
        for pred, t in zip(preds, y):
            true = sorted(BORE3[i] for i in range(3) if t[i] >= 0.5)
            ok_ex += (pred == true)
            n_ex += 1
            for f in BORE3:
                st[f]["true"] += f in true
                st[f]["pred"] += f in pred
                st[f]["ok"] += (f in true and f in pred)
    out = {f: {"n": st[f]["true"], **prf(st[f]["ok"], st[f]["true"], st[f]["pred"])} for f in BORE3}
    return {"par_famille": out,
            "macro_f1": round(float(np.mean([out[f]["f1"] for f in BORE3])), 3),
            "exact": round(ok_ex / max(1, n_ex), 3)}


@torch.no_grad()
def predict_texts(model, encode, texts, side, device, bs=16):
    model.eval()
    outs = []
    for i in range(0, len(texts), bs):
        enc = encode(texts[i:i + bs])
        enc = {k: v.to(device) for k, v in enc.items()}
        outs.append(model(enc).float().cpu())
    n_classes = len(FAMILLES6) if side == "accepte" else len(BORE3)
    return torch.cat(outs) if outs else torch.empty(0, n_classes)


def gold_eval(model, encode, data_dir, side, device, smoke):
    """Final evaluation against the gold set this repository ships, already joined
    to its text and machine label by build_training_data.py."""
    gold = load_jsonl(data_dir / f"gold_eval_{side}.jsonl")
    if smoke:
        gold = gold[:12]
    if side == "accepte":
        rows = [g for g in gold if g["scorable"]]
        logits = predict_texts(model, encode, [g["text"] for g in rows], side, device)
        preds = [FAMILLES6[int(i)] for i in logits.argmax(-1)]
        res = {"n": len(rows), "excluded_out_of_nomenclature": len(gold) - len(rows)}
        for who, labels in [("model", preds), ("rules", [g["family_machine"] for g in rows])]:
            ok = sum(p == g["family_annotated"] for p, g in zip(labels, rows))
            res[f"agreement_{who}"] = [ok, len(rows)]
        conf = {f: Counter() for f in FAMILLES6}
        for p, g in zip(preds, rows):
            conf[g["family_annotated"]][p] += 1
        res["par_famille"] = {f: {"n": sum(conf[f].values()),
                                   **prf(conf[f][f], sum(conf[f].values()),
                                         sum(conf[g][f] for g in FAMILLES6))}
                              for f in FAMILLES6}
        uni = [(p, g) for p, g in zip(preds, rows) if g.get("stratum") == "uniforme"]
        res["agreement_model_uniforme"] = [sum(p == g["family_annotated"] for p, g in uni), len(uni)]
        res["agreement_rules_uniforme"] = [sum(g["family_machine"] == g["family_annotated"]
                                               for _, g in uni), len(uni)]
        # every model miss, with the gold and rules labels
        res["errors"] = [{"pid": g["pid"], "stratum": g.get("stratum"),
                          "gold": g["family_annotated"], "rules": g["family_machine"],
                          "model": p}
                         for p, g in zip(preds, rows) if p != g["family_annotated"]]
    else:
        rows = [g for g in gold if not g["indetermine"]]
        logits = predict_texts(model, encode, [g["text"] for g in rows], side, device)
        preds = predict_sets(logits)
        res = {"n": len(rows)}
        for who, labels in [("model", preds), ("rules", [g["bore_machine"] for g in rows])]:
            ok = sum(p == g["bore_annotated"] for p, g in zip(labels, rows))
            res[f"agreement_{who}"] = [ok, len(rows)]
        st = {f: Counter() for f in BORE3}
        for p, g in zip(preds, rows):
            for f in BORE3:
                st[f]["true"] += f in g["bore_annotated"]
                st[f]["pred"] += f in p
                st[f]["ok"] += (f in g["bore_annotated"] and f in p)
        res["par_famille"] = {f: {"n": st[f]["true"], **prf(st[f]["ok"], st[f]["true"], st[f]["pred"])}
                              for f in BORE3}
        uni = [(p, g) for p, g in zip(preds, rows) if g.get("stratum") == "uniforme"]
        res["agreement_model_uniforme"] = [sum(p == g["bore_annotated"] for p, g in uni), len(uni)]
        res["agreement_rules_uniforme"] = [sum(g["bore_machine"] == g["bore_annotated"]
                                               for _, g in uni), len(uni)]
        res["errors"] = [{"pid": g["pid"], "stratum": g.get("stratum"),
                          "gold": g["bore_annotated"], "rules": g["bore_machine"],
                          "model": p}
                         for p, g in zip(preds, rows) if p != g["bore_annotated"]]
    return res


def main():
    args = parse_args()
    cfg = load_config()["baselines"]
    model_name = cfg["models"][args.model]
    max_len = 1024 if args.model == "camembertv2" else 512
    data_dir = Path(args.data_dir) if args.data_dir else resolve(cfg["out_dir"])
    ckpt_dir = Path(args.checkpoints_dir) if args.checkpoints_dir else resolve(cfg["checkpoints_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    epochs = args.epochs if args.epochs is not None else (4 if args.side == "accepte" else 3)
    batch, grad_accum = BATCH, GRAD_ACCUM
    if args.smoke:
        epochs, batch, grad_accum, max_len = 1, 2, 1, min(max_len, 128)

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    device = "cpu" if args.smoke else ("cuda" if torch.cuda.is_available() else "cpu")
    classes = FAMILLES6 if args.side == "accepte" else BORE3
    capture_version = hashlib.md5(open(__file__, "rb").read()).hexdigest()[:12]
    print(f"capture_version={capture_version} model={args.model} ({model_name}) "
          f"side={args.side} device={device} max_len={max_len} epochs={epochs} smoke={args.smoke}")

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    encode, n_trunc = make_encode(tok, max_len)
    collate = make_collate(encode, args.side)

    rng = random.Random(SEED)
    train_all = load_jsonl(data_dir / f"train_{args.side}.jsonl")
    dev = load_jsonl(data_dir / f"dev_{args.side}.jsonl")
    if args.smoke:
        dev = dev[:12]

    if args.side == "accepte":
        by_fam = {f: [r for r in train_all if r["famille"] == f] for f in FAMILLES6}
        n_min = min(len(v) for v in by_fam.values())
        if args.smoke:
            n_min = min(n_min, 4)
        train = []
        for f in FAMILLES6:
            rows = sorted(by_fam[f], key=lambda r: r["pid"])
            rng.shuffle(rows)
            train += rows[:n_min]
        rng.shuffle(train)
        print(f"class-balanced train: {len(train)} ({n_min} per family) | dev: {len(dev)}")
        print("available per family:", {f: len(v) for f, v in by_fam.items()})
    else:
        train = sorted(train_all, key=lambda r: r["pid"])
        rng.shuffle(train)
        if args.smoke:
            train = train[:24]
        print(f"train: {len(train)} | dev: {len(dev)} | signatures:",
              dict(Counter("+".join(r["bore"]) for r in train)))

    model = Model(model_name, len(classes)).to(device)
    dl_tr = DataLoader(GroundsDataset(train, args.side), batch_size=batch, shuffle=True,
                        collate_fn=collate, drop_last=len(train) > batch)
    dl_dev = DataLoader(GroundsDataset(dev, args.side), batch_size=batch * 2, shuffle=False,
                         collate_fn=collate)
    steps = max(1, len(dl_tr) // grad_accum) * epochs
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss() if args.side == "accepte" else nn.BCEWithLogitsLoss()

    tag = args.model
    ckpt_path = ckpt_dir / f"best_{args.side}_{tag}.pt"
    best = -1.0
    for ep in range(epochs):
        model.train()
        for i, (enc, y) in enumerate(dl_tr):
            enc = {k: v.to(device) for k, v in enc.items()}
            with torch.amp.autocast("cuda", enabled=device == "cuda"):
                loss = loss_fn(model(enc), y.to(device))
            scaler.scale(loss / grad_accum).backward()
            if (i + 1) % grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            if i % 100 == 0:
                print(f"ep{ep} step {i}/{len(dl_tr)} loss {loss.item():.3f}", flush=True)
        m = run_eval(model, dl_dev, args.side, device)
        metric_name = "micro_acc" if args.side == "accepte" else "exact"
        print(f"== epoch {ep} dev: macro_f1 {m['macro_f1']} {metric_name} {m[metric_name]}")
        print("   per family:", {f: m["par_famille"][f]["f1"] for f in classes})
        if m["macro_f1"] > best:
            best = m["macro_f1"]
            torch.save({"model": model.state_dict(), "capture_version": capture_version,
                        "model_name": model_name, "side": args.side, "epoch": ep,
                        "dev_metrics": m}, ckpt_path)
            with open(ckpt_dir / f"metrics_dev_{args.side}_{tag}.json", "w", encoding="utf-8") as fh:
                json.dump(m, fh, indent=1)
    print(f"best dev macro-F1 ({args.side}, {tag}): {round(best, 3)}")

    ck = torch.load(ckpt_path, map_location=device)
    assert ck["capture_version"] == capture_version, "checkpoint from a different code version"
    model.load_state_dict(ck["model"])
    res = gold_eval(model, encode, data_dir, args.side, device, args.smoke)
    res.update({"model": model_name, "side": args.side, "dev_macro_f1": best,
                "capture_version": capture_version,
                "truncation": {"max_len": max_len, **dict(n_trunc)}})
    with open(ckpt_dir / f"gold_metrics_{args.side}_{tag}.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)

    print(f"\n=== gold evaluation ({args.side}, {tag}) ===")
    a, b = res["agreement_model"]
    c, d = res["agreement_rules"]
    print(f"agreement with the gold annotation: model {a}/{b} = {100 * a / max(1, b):.1f}% | "
          f"rule-based classifier {c}/{d} = {100 * c / max(1, d):.1f}%")
    print("per family (truth = gold annotation):", {f: res["par_famille"][f]["f1"] for f in classes})
    print("done.")


if __name__ == "__main__":
    main()
