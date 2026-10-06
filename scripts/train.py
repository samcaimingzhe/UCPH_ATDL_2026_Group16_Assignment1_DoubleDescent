#!/usr/bin/env python3
"""Train the width/trial grid for Figure 1. One resumable .pt file per job."""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model.resnet import make_resnet18  # noqa: E402


def parse_range(text: str) -> list[int]:
    values = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            start, end = (int(x) for x in part.split("-", 1))
            values.extend(range(start, end + 1))
        else:
            values.append(int(part))
    values = sorted(set(values))
    if not values:
        raise ValueError("empty integer range")
    return values


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def noisy_labels(clean: np.ndarray, probability: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random(len(clean)) < probability
    offsets = rng.integers(1, 10, size=len(clean))
    noisy = clean.copy()
    noisy[mask] = (noisy[mask] + offsets[mask]) % 10
    return noisy


class TrainData(Dataset):
    def __init__(self, images, labels, augmentation_seed: int):
        self.images = images
        self.labels = labels
        self.augmentation_seed = augmentation_seed
        self.epoch = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        generator = torch.Generator().manual_seed(
            self.augmentation_seed + 1_000_003 * self.epoch + 9_176 * index
        )
        x = TF.pad(x, [4, 4, 4, 4])
        top, left = torch.randint(0, 9, (2,), generator=generator).tolist()
        x = TF.crop(x, top, left, 32, 32)
        if torch.rand((), generator=generator).item() < 0.5:
            x = TF.hflip(x)
        return x, int(self.labels[index])


class EvalData(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float() / 255.0
        return x, int(self.labels[index])


def channel_stats(images: np.ndarray) -> tuple[list[float], list[float]]:
    pixels = images.astype(np.float64) / 255.0
    return pixels.mean(axis=(0, 1, 2)).tolist(), pixels.std(axis=(0, 1, 2)).tolist()


@torch.inference_mode()
def evaluate(model, loader, device) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    errors = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        errors += (logits.argmax(1) != labels).sum().item()
        count += len(labels)
    return loss_sum / count, errors / count


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train_one(args, width: int, trial: int, arrays, mean, std) -> None:
    result = args.output / f"width_{width:02d}_trial_{trial}.pt"
    clean_train = arrays["train_labels"]
    train_labels = noisy_labels(clean_train, args.noise, 2000 + trial)
    train_data = TrainData(arrays["train_images"], train_labels, 4000 + trial)
    train_eval = EvalData(arrays["train_images"], train_labels)
    test_eval = EvalData(arrays["test_images"], arrays["test_labels"])

    seed_everything(trial)
    model = make_resnet18(width, mean, std).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []
    start_epoch = 0

    if result.exists():
        saved = torch.load(result, map_location="cpu", weights_only=False)
        expected = {"width": width, "trial": trial, "epochs": args.epochs, "noise": args.noise}
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"settings do not match existing checkpoint: {result}")
        if saved.get("complete"):
            print(f"skip complete {result.name}", flush=True)
            return
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
        start_epoch = saved["epoch"]

    eval_train_loader = DataLoader(train_eval, batch_size=args.eval_batch_size, shuffle=False,
                                   num_workers=args.workers)
    eval_test_loader = DataLoader(test_eval, batch_size=args.eval_batch_size, shuffle=False,
                                  num_workers=args.workers)
    started = time.time()

    for epoch in range(start_epoch, args.epochs):
        train_data.epoch = epoch
        generator = torch.Generator().manual_seed(3000 + trial + epoch)
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                  generator=generator, num_workers=args.workers,
                                  persistent_workers=False)
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(images), labels)
            loss.backward()
            optimizer.step()

        train_loss, train_error = evaluate(model, eval_train_loader, args.device)
        test_loss, clean_test_error = evaluate(model, eval_test_loader, args.device)
        noisy_test_error = args.noise + (1.0 - 10.0 * args.noise / 9.0) * clean_test_error
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_error": train_error,
            "test_loss": test_loss,
            "clean_test_error": clean_test_error,
            "noisy_test_error": noisy_test_error,
        }
        history.append(row)
        payload = {
            "width": width,
            "trial": trial,
            "epochs": args.epochs,
            "noise": args.noise,
            "epoch": epoch + 1,
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "history": history,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "complete": epoch + 1 == args.epochs,
        }
        save_checkpoint(result, payload)
        print(
            f"width={width:02d} trial={trial} epoch={epoch + 1:04d}/{args.epochs} "
            f"train_error={train_error:.4f} noisy_test_error={noisy_test_error:.4f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--widths", default="1-64")
    parser.add_argument("--trials", default="0-4")
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if not 0 <= args.shard < args.num_shards:
        raise ValueError("shard must satisfy 0 <= shard < num-shards")
    args.output.mkdir(parents=True, exist_ok=True)

    train = CIFAR10(args.data, train=True, download=True)
    test = CIFAR10(args.data, train=False, download=True)
    arrays = {
        "train_images": np.asarray(train.data),
        "train_labels": np.asarray(train.targets, dtype=np.int64),
        "test_images": np.asarray(test.data),
        "test_labels": np.asarray(test.targets, dtype=np.int64),
    }
    mean, std = channel_stats(arrays["train_images"])
    jobs = [(width, trial) for width in parse_range(args.widths) for trial in parse_range(args.trials)]
    jobs = [job for index, job in enumerate(jobs) if index % args.num_shards == args.shard]
    print(f"jobs={len(jobs)} shard={args.shard}/{args.num_shards} mean={mean} std={std}")
    for width, trial in jobs:
        train_one(args, width, trial, arrays, mean, std)


if __name__ == "__main__":
    main()

