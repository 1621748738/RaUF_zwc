"""Standalone RaUF training entry point built on the RaLD repository."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from torch.utils.tensorboard import SummaryWriter

import utils.misc as misc
from datasets.rauf_dataset import get_rauf_dataset
from engine_rauf import evaluate, train_one_epoch
from model.models_rauf import build_rauf
from model.rauf_loss import RaUFLoss


def parse_args():
    parser = argparse.ArgumentParser("RaUF training/evaluation")
    parser.add_argument(
        "--config", default="configs/rauf/rauf_coloradar_sc.yml", help="YAML config"
    )
    return parser.parse_args()


def make_loader(dataset, sampler, config, training: bool):
    workers = int(config.num_workers if training else config.eval_num_workers)
    kwargs = dict(
        dataset=dataset,
        sampler=sampler,
        batch_size=int(config.batch_size if training else config.eval_batch_size),
        num_workers=workers,
        pin_memory=bool(config.pin_mem),
        drop_last=training,
    )
    if workers:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return torch.utils.data.DataLoader(**kwargs)


def load_checkpoint(path, model, optimizer=None, scaler=None):
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", -1)) + 1


def save_checkpoint(path, epoch, model, optimizer, scaler):
    misc.save_on_master(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
        },
        path,
    )


def main(config):
    misc.init_distributed_mode(config.train)
    rank = misc.get_rank()
    seed = int(config.system.seed) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if config.train.distributed:
        device = torch.device("cuda", config.train.gpu)
    else:
        device = torch.device(config.system.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_dataset = get_rauf_dataset(config.dataset, "train")
    val_dataset = get_rauf_dataset(config.dataset, "val")
    if config.train.distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=True, drop_last=True
        )
        val_sampler = torch.utils.data.DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        val_sampler = torch.utils.data.SequentialSampler(val_dataset)
    train_loader = make_loader(train_dataset, train_sampler, config.dataset, True)
    val_loader = make_loader(val_dataset, val_sampler, config.dataset, False)

    model = build_rauf(config.model).to(device)
    criterion = RaUFLoss(**dict(config.loss)).to(device)
    model_without_ddp = model
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.train.lr),
        weight_decay=float(config.train.weight_decay),
    )
    amp_enabled = bool(config.train.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    start_epoch = int(config.train.start_epoch)
    resume = config.train.get("resume", None)
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer, scaler)

    if config.train.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[config.train.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module

    output_dir = Path(config.system.output_dir) / config.system.expname
    if misc.is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_dir / "tensorboard")
    else:
        writer = None

    if config.system.mode == "eval":
        checkpoint = config.eval.get("ckpt", None)
        if not checkpoint:
            raise ValueError("eval.ckpt is required in evaluation mode")
        load_checkpoint(checkpoint, model_without_ddp)
        stats = evaluate(model, criterion, val_loader, device)
        if misc.is_main_process():
            print(json.dumps(stats, indent=2))
        return

    parameters = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
    if misc.is_main_process():
        print(f"RaUF trainable parameters: {parameters / 1e6:.2f} M")
    started = time.time()
    for epoch in range(start_epoch, int(config.train.epochs)):
        if config.train.distributed:
            train_sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            scaler,
            config.train,
        )
        should_evaluate = (
            (epoch + 1) % int(config.train.eval_freq) == 0
            or epoch + 1 == int(config.train.epochs)
        )
        val_stats = (
            evaluate(model, criterion, val_loader, device)
            if should_evaluate
            else {}
        )
        if misc.is_main_process():
            for key, value in train_stats.items():
                writer.add_scalar(f"train/{key}", value, epoch)
            for key, value in val_stats.items():
                writer.add_scalar(f"val/{key}", value, epoch)
            record = {"epoch": epoch, "train": train_stats, "val": val_stats}
            with (output_dir / "log.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        if (
            (epoch + 1) % int(config.train.save_ckpt_freq) == 0
            or epoch + 1 == int(config.train.epochs)
        ):
            save_checkpoint(
                output_dir / f"checkpoint-{epoch:04d}.pth",
                epoch,
                model_without_ddp,
                optimizer,
                scaler,
            )
    if writer is not None:
        writer.close()
    if misc.is_main_process():
        print(f"Training time: {(time.time() - started) / 3600:.2f} h")


if __name__ == "__main__":
    cli = parse_args()
    config = edict(yaml.safe_load(Path(cli.config).read_text(encoding="utf-8")))
    main(config)
