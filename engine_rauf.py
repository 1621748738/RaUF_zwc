"""Training and validation loops for the RaUF path."""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict, Iterable

import torch
from tqdm import tqdm

import utils.misc as misc


class _SparseLogger:
    """Print progress every *log_every* steps with loss values."""

    def __init__(self, loader, epoch: int, log_every: int, enabled: bool):
        self.loader = loader
        self.epoch = epoch
        self.log_every = max(1, log_every)
        self.enabled = enabled
        self._postfix = {}

    def __iter__(self):
        for step, item in enumerate(self.loader):
            yield item
            if self.enabled and (
                step % self.log_every == 0 or step == len(self.loader) - 1
            ):
                parts = " ".join(
                    f"{k}={v}" for k, v in self._postfix.items()
                )
                print(
                    f"[Epoch {self.epoch}] step {step + 1}/{len(self.loader)} {parts}",
                    flush=True,
                )

    def __len__(self):
        return len(self.loader)

    def set_postfix(self, **kwargs):
        self._postfix = {k: f"{v:.3f}" if isinstance(v, float) else str(v)
                         for k, v in kwargs.items()}


def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _learning_rate(optimizer, progress: float, config) -> float:
    warmup = float(config.get("warmup_epochs", 0))
    epochs = float(config.epochs)
    minimum = float(config.get("min_lr", 0.0))
    base = float(config.lr)
    if warmup > 0 and progress < warmup:
        value = base * progress / warmup
    else:
        denominator = max(epochs - warmup, 1.0)
        cosine_progress = min(1.0, max(0.0, (progress - warmup) / denominator))
        value = minimum + (base - minimum) * 0.5 * (
            1.0 + math.cos(math.pi * cosine_progress)
        )
    for group in optimizer.param_groups:
        group["lr"] = value * group.get("lr_scale", 1.0)
    return value


def _reduced_stats(totals: Dict[str, float], steps: int) -> Dict[str, float]:
    if not steps:
        return {}
    return {
        key: misc.all_reduce_mean(value / steps) for key, value in totals.items()
    }


def train_one_epoch(
    model,
    criterion,
    data_loader: Iterable,
    optimizer,
    device: torch.device,
    epoch: int,
    scaler,
    config,
) -> Dict[str, float]:
    model.train()
    criterion.train()
    optimizer.zero_grad(set_to_none=True)
    totals = defaultdict(float)
    accumulation = int(config.get("accum_iter", 1))
    amp_enabled = bool(config.get("amp", True)) and device.type == "cuda"
    steps = 0

    import sys
    import utils.misc as misc

    def _make_progress(loader, epoch, total):
        """tqdm for TTY, sparse log lines for redirected output."""
        if sys.stdout.isatty():
            return tqdm(loader, desc=f"Epoch {epoch}", mininterval=1.0,
                        disable=not misc.is_main_process())
        # When output is redirected (e.g. to a log file), print a line
        # every ~10 % of the epoch to avoid log explosion from \r updates.
        log_every = max(1, total // 10)
        return _SparseLogger(loader, epoch, log_every, misc.is_main_process())

    total_steps = len(data_loader)
    pbar = _make_progress(data_loader, epoch, total_steps)
    for step, raw_batch in enumerate(pbar):
        progress = epoch + step / max(len(data_loader), 1)
        learning_rate = _learning_rate(optimizer, progress, config)
        batch = move_batch_to_device(raw_batch, device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast:
            prediction = model(batch["radar_cube"], batch["radar_validity"])
            losses = criterion(prediction, batch)
            loss = losses["loss"] / accumulation

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite RaUF loss at epoch {epoch}, step {step}")
        scaler.scale(loss).backward()
        update = (step + 1) % accumulation == 0 or step + 1 == len(data_loader)
        if update:
            scaler.unscale_(optimizer)
            clip = config.get("clip_grad", None)
            if clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(clip))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        totals["lr"] += learning_rate
        for key, value in losses.items():
            totals[key] += float(value.detach().item())
        steps += 1
        pbar.set_postfix(
            loss=f"{float(losses['loss'].detach().item()):.2f}",
            spatial=f"{float(losses['spatial_nll'].detach().item()):.1f}",
            occ=f"{float(losses['occupancy_loss'].detach().item()):.4f}",
            lr=f"{learning_rate:.2e}",
        )
    return _reduced_stats(totals, steps)


@torch.no_grad()
def evaluate(model, criterion, data_loader, device: torch.device):
    model.eval()
    criterion.eval()
    totals = defaultdict(float)
    steps = 0
    for raw_batch in data_loader:
        batch = move_batch_to_device(raw_batch, device)
        prediction = model(batch["radar_cube"], batch["radar_validity"])
        losses = criterion(prediction, batch)
        for key, value in losses.items():
            totals[key] += float(value.detach().item())

        predicted = prediction["occupancy_logits"].sigmoid() >= 0.5
        target = batch["frustum_occupancy"] > 0.5
        intersection = (predicted & target).sum().float()
        union = (predicted | target).sum().float()
        true_positive = intersection
        totals["occupancy_iou"] += float((intersection / union.clamp_min(1)).item())
        totals["occupancy_precision"] += float(
            (true_positive / predicted.sum().clamp_min(1)).item()
        )
        totals["occupancy_recall"] += float(
            (true_positive / target.sum().clamp_min(1)).item()
        )
        steps += 1
    return _reduced_stats(totals, steps)
