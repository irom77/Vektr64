"""Time-budgeted training with checkpoints and resume."""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from . import vknn
from .model import Net, quantise
from .prep import NO_RESULT, read_shard
from .vknn import SCALE


@dataclass
class Config:
    minutes: float
    batch: int = 16384
    lr: float = 1e-3
    lr_end: float = 1e-5
    score_weight: float = 0.75
    holdout: float = 0.01
    checkpoint_s: float = 600
    seed: int = 1


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def targets(score: torch.Tensor, result: torch.Tensor, score_weight: float) -> torch.Tensor:
    """Blend of the squashed score and the game result (0, 0.5, 1); unknown results use the score alone."""
    squashed = torch.sigmoid(score.float() / SCALE)
    known = result != NO_RESULT
    outcome = result.float() / 2.0
    blended = score_weight * squashed + (1.0 - score_weight) * outcome
    return torch.where(known, blended, squashed)


def _load(shards: Path, device: str) -> tuple[torch.Tensor, ...]:
    paths = sorted(Path(shards).glob("*.npz"))
    if not paths:
        raise ValueError(f"no shards (*.npz) in {shards}")
    parts = [read_shard(p) for p in paths]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in ("features", "stm", "score", "result")}
    return (
        torch.from_numpy(cat["features"].astype(np.int32)).to(device),
        torch.from_numpy(cat["stm"].astype(np.int64)).to(device),
        torch.from_numpy(cat["score"].astype(np.int64)).to(device),
        torch.from_numpy(cat["result"].astype(np.int64)).to(device),
    )


def _loss(net: Net, features, stm, score, result, weight: float) -> torch.Tensor:
    out = net(features.long(), stm)
    return ((torch.sigmoid(out) - targets(score, result, weight)) ** 2).mean()


@torch.no_grad()
def _validation_loss(net: Net, data, weight: float, batch: int) -> float | None:
    n = len(data[0])
    if n == 0:
        return None
    total = 0.0
    for i in range(0, n, batch):
        part = [d[i : i + batch] for d in data]
        total += _loss(net, *part, weight).item() * len(part[0])
    return total / n


def _fingerprint(shards: Path, total: int, n_train: int, holdout: float) -> str:
    """Identifies the data split, so a resume cannot train on what an earlier run held out."""
    files = [[p.name, p.stat().st_size] for p in sorted(Path(shards).glob("*.npz"))]
    return json.dumps({"shards": files, "total": total, "n_train": n_train, "holdout": holdout})


def _save_checkpoint(path: Path, net, opt, steps: int, elapsed: float, train_loss, fingerprint: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(
        {"model": net.state_dict(), "optimiser": opt.state_dict(), "steps": steps, "elapsed": elapsed, "train_loss": train_loss, "fingerprint": fingerprint},
        tmp,
    )
    os.replace(tmp, path)


def train(shards: Path, out: Path, cfg: Config, *, device: str | None = None, clock=time.monotonic) -> dict:
    device = device or pick_device()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    features, stm, score, result = _load(shards, device)
    total = len(features)
    held = min(int(total * cfg.holdout), total - 1) if cfg.holdout > 0 else 0
    if cfg.holdout > 0 and held == 0 and total > 1:
        held = 1
    n_train = total - held
    train_data = (features[:n_train], stm[:n_train], score[:n_train], result[:n_train])
    val_data = (features[n_train:], stm[n_train:], score[n_train:], result[n_train:])

    fingerprint = _fingerprint(shards, total, n_train, cfg.holdout)
    torch.manual_seed(cfg.seed)
    net = Net().to(device)
    net.clip_()
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr)
    steps, elapsed, train_loss = 0, 0.0, None
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        if state["fingerprint"] != fingerprint:
            raise ValueError(
                f"checkpoint {checkpoint} was made with different data or holdout: "
                f"{state['fingerprint']} != {fingerprint}"
            )
        net.load_state_dict(state["model"])
        opt.load_state_dict(state["optimiser"])
        steps, elapsed, train_loss = int(state["steps"]), float(state["elapsed"]), state["train_loss"]

    budget = cfg.minutes * 60.0
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed + steps)
    recent: list[torch.Tensor] = []
    start = clock()
    last_checkpoint = start
    perm, pos = None, 0
    while elapsed < budget:
        if perm is None or pos >= n_train:
            perm, pos = torch.randperm(n_train, device=device, generator=generator), 0
        idx = perm[pos : pos + cfg.batch]
        pos += cfg.batch
        frac = min(elapsed / budget, 1.0)
        lr = cfg.lr_end + (cfg.lr - cfg.lr_end) * 0.5 * (1.0 + math.cos(math.pi * frac))
        for group in opt.param_groups:
            group["lr"] = lr
        loss = _loss(net, *(d[idx] for d in train_data), cfg.score_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        net.clip_()
        steps += 1
        recent = (recent + [loss.detach()])[-100:]
        now = clock()
        elapsed += now - start
        start = now
        if now - last_checkpoint >= cfg.checkpoint_s:
            train_loss = torch.stack(recent).mean().item()
            _save_checkpoint(checkpoint, net, opt, steps, elapsed, train_loss, fingerprint)
            last_checkpoint = now
    if recent:
        train_loss = torch.stack(recent).mean().item()
    _save_checkpoint(checkpoint, net, opt, steps, elapsed, train_loss, fingerprint)

    qnet = quantise(net)
    vknn.write(out / "net.vknn", qnet)
    vknn.write_check(out / "net.check", qnet)
    info = {
        "net": vknn.net_hash(out / "net.vknn"),
        "positions": total,
        "steps": steps,
        "epochs": steps * cfg.batch / n_train,
        "elapsed_s": elapsed,
        "train_loss": train_loss,
        "val_loss": _validation_loss(net, val_data, cfg.score_weight, cfg.batch),
        "device": device,
        "config": asdict(cfg),
    }
    (out / "run.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="vektr_trainer.train")
    parser.add_argument("--shards", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--minutes", type=float, required=True)
    parser.add_argument("--checkpoint-s", type=float, default=600)
    args = parser.parse_args(argv)
    info = train(args.shards, args.out, Config(minutes=args.minutes, checkpoint_s=args.checkpoint_s))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
