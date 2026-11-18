#!/usr/bin/env python
"""
Simplified training script for Depth Anything 3 on Toulouse dataset.

This is a minimal version of train.py with fewer dependencies and simpler logic.
Use this for debugging and initial testing.

Usage:
    python train_simple.py --dataset-root /path/to/dataset --model da3-small

Resume a previous run (model + optimizer + scheduler + scaler + epoch):
    python train_simple.py --dataset-root /path/to/dataset --epochs 20 \
        --resume ./outputs_simple/epoch_10.pt
    python train_simple.py --dataset-root /path/to/dataset --epochs 20 --resume auto

Start a NEW training from previous weights only (fresh optimizer/scheduler, epoch 0):
    python train_simple.py --dataset-root /path/to/dataset --init-from ./outputs_simple/epoch_10.pt
"""
import argparse, os, sys, time, gc, datetime, math, re
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

# Add paths
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from canopy_dataset import CanopyDataset, DataField
from depth_anything_3.cfg import load_config, create_object
from depth_anything_3.registry import MODEL_REGISTRY


def find_latest_checkpoint(output_dir):
    """Return the epoch_N.pt with the highest N in output_dir, or None."""
    best_epoch, best_path = -1, None
    for p in Path(output_dir).glob('epoch_*.pt'):
        m = re.fullmatch(r'epoch_(\d+)\.pt', p.name)
        if m and int(m.group(1)) > best_epoch:
            best_epoch, best_path = int(m.group(1)), p
    return best_path


def extract_model_state(checkpoint):
    """Accept either a full training checkpoint or a bare state_dict."""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        return checkpoint['state_dict']
    return checkpoint


def compute_relative_extrinsics(extrinsics_c2w):
    """Compute relative extrinsics with respect to the first (parent/query) image.

    For a batch of S images, the parent is the first image (index 0).
    All other images are represented relative to the parent.

    The relative transform is computed as:
        R_rel = R_parent^T * R_i
        t_rel = R_parent^T * (t_i - t_parent)

    The parent image gets:
        R_rel = identity matrix
        t_rel = [0, 0, 0]

    Args:
        extrinsics_c2w: [B, S, 4, 4] or [S, 4, 4] camera-to-world matrices

    Returns:
        relative_extrinsics: [B, S, 4, 4] or [S, 4, 4] relative extrinsics
            where the first image has identity rotation and zero translation
    """
    if extrinsics_c2w.dim() == 4:  # [B, S, 4, 4]
        B, S = extrinsics_c2w.shape[:2]
        device = extrinsics_c2w.device
        dtype = extrinsics_c2w.dtype

        # Get parent (first in sequence) extrinsics
        parent_extrinsics = extrinsics_c2w[:, 0:1, :, :]  # [B, 1, 4, 4]

        # Extract parent rotation and translation
        R_parent = parent_extrinsics[:, :, :3, :3]  # [B, 1, 3, 3]
        t_parent = parent_extrinsics[:, :, :3, 3]   # [B, 1, 3]

        # Extract all rotations and translations
        R_all = extrinsics_c2w[:, :, :3, :3]  # [B, S, 3, 3]
        t_all = extrinsics_c2w[:, :, :3, 3]   # [B, S, 3]

        # Compute relative rotation: R_rel = R_parent^T * R_i
        # For parent: R_parent^T * R_parent = I
        R_rel = torch.matmul(R_parent.transpose(-2, -1), R_all)  # [B, S, 3, 3]

        # Compute relative translation: t_rel = R_parent^T * (t_i - t_parent)
        # For parent: R_parent^T * (t_parent - t_parent) = 0
        t_diff = t_all - t_parent  # [B, S, 3]
        t_rel = torch.matmul(R_parent.transpose(-2, -1), t_diff.unsqueeze(-1)).squeeze(-1)  # [B, S, 3]

        # Build relative extrinsics matrices directly (avoid expand + assign issue)
        # Create 4x4 matrices from rotation and translation
        # Top-left 3x3: R_rel
        # Top-right 3x1: t_rel
        # Bottom row: [0, 0, 0, 1]
        relative_extrinsics = torch.cat([
            torch.cat([R_rel, t_rel.unsqueeze(-1)], dim=-1),  # [B, S, 3, 4]
            torch.tensor([[[0., 0., 0., 1.]]], device=device, dtype=dtype).expand(B, S, 1, 4)
        ], dim=-2)  # [B, S, 4, 4]

        return relative_extrinsics

    else:  # [S, 4, 4]
        S = extrinsics_c2w.shape[0]
        device = extrinsics_c2w.device
        dtype = extrinsics_c2w.dtype

        # Get parent (first in sequence) extrinsics
        parent_extrinsics = extrinsics_c2w[0:1, :, :]  # [1, 4, 4]

        # Extract parent rotation and translation
        R_parent = parent_extrinsics[:, :3, :3]  # [1, 3, 3]
        t_parent = parent_extrinsics[:, :3, 3]   # [1, 3]

        # Extract all rotations and translations
        R_all = extrinsics_c2w[:, :3, :3]  # [S, 3, 3]
        t_all = extrinsics_c2w[:, :3, 3]   # [S, 3]

        # Compute relative rotation: R_rel = R_parent^T * R_i
        R_rel = torch.matmul(R_parent.transpose(-2, -1), R_all)  # [S, 3, 3]

        # Compute relative translation: t_rel = R_parent^T * (t_i - t_parent)
        t_diff = t_all - t_parent  # [S, 3]
        t_rel = torch.matmul(R_parent.transpose(-2, -1), t_diff.unsqueeze(-1)).squeeze(-1)  # [S, 3]

        # Build relative extrinsics matrices directly
        relative_extrinsics = torch.cat([
            torch.cat([R_rel, t_rel.unsqueeze(-1)], dim=-1),  # [S, 3, 4]
            torch.tensor([[[0., 0., 0., 1.]]], device=device, dtype=dtype).expand(S, 1, 4)
        ], dim=-2)  # [S, 4, 4]

        return relative_extrinsics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True, help='Dataset root')
    parser.add_argument('--subset', type=str, default='synthetic_test_672x448')
    parser.add_argument('--model', type=str, default='da3-small', choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=10,
                        help='Total number of epochs (when resuming, training continues up to this epoch)')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--warmup-ratio', type=float, default=0.0,
                        help='Fraction of total steps used for linear LR warmup')
    parser.add_argument('--min-lr-ratio', type=float, default=0.1,
                        help='Final LR as a fraction of --lr at the end of the cosine decay')
    parser.add_argument('--grad-clip', type=float, default=0.0,
                        help='Max gradient norm for clipping (<= 0 disables clipping)')
    parser.add_argument('--num-pairs', type=int, default=3)
    parser.add_argument('--output', type=str,
                        default=str(Path(__file__).resolve().parent / 'checkpoints'),
                        help='Directory where checkpoints are saved (default: ./checkpoints next to this script)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint to resume from (restores model, optimizer, scheduler, scaler '
                             'and epoch). Use "auto" to pick the latest epoch_N.pt in --output.')
    parser.add_argument('--init-from', type=str, default=None,
                        help='Checkpoint (or bare state_dict) to load model weights from, then start a '
                             'fresh training (new optimizer/scheduler, epoch 0). Ignored if --resume is set.')

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"Model: {args.model}")

    # Load dataset
    print("\nLoading dataset...")
    dataset = CanopyDataset(
        dataset_root=args.dataset_root,
        subset=args.subset,
        img_size=672,
        num_pairs=args.num_pairs,
        use_depth=True,
    )

    # Collate function
    def collate_fn(batch):
        parent_name = [b[DataField.PARENT_NAME] for b in batch]
        images = [b[DataField.IMAGES] for b in batch]
        extrinsics = [b[DataField.EXTRINSICS_C2W] for b in batch]
        intrinsics = [b[DataField.INTRINSICS] for b in batch]
        depths = [b[DataField.DEPTHS] for b in batch]

        return {
            'parent_name': parent_name,
            'images': torch.stack(images),
            'extrinsics': torch.stack(extrinsics),
            'intrinsics': torch.stack(intrinsics),
            'depths': torch.stack(depths),
        }

    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Load model
    print("\nLoading model...")
    config_path = MODEL_REGISTRY[args.model]
    config = load_config(config_path)
    model = create_object(config)
    model = model.to(device)
    model.train()

    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=True)

    # Scheduler: linear warmup then cosine decay, stepped once per iteration
    total_steps = max(1, args.epochs * len(train_loader))
    warmup_steps = int(args.warmup_ratio * total_steps)

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return args.min_lr_ratio + (1.0 - args.min_lr_ratio) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: warmup {warmup_steps} steps, cosine over {total_steps} steps "
          f"(lr {args.lr:g} -> {args.lr * args.min_lr_ratio:g})")

    # Resume / initialise from a previous checkpoint
    start_epoch = 0
    resume_path = args.resume
    if resume_path == 'auto':
        latest = find_latest_checkpoint(output_dir)
        if latest is None:
            print(f"\n--resume auto: no epoch_*.pt found in {output_dir}, starting from scratch")
            resume_path = None
        else:
            resume_path = str(latest)

    if resume_path is not None:
        print(f"\nResuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(extract_model_state(ckpt))
        if isinstance(ckpt, dict) and 'optimizer' in ckpt:
            optimizer.load_state_dict(ckpt['optimizer'])
        else:
            print("  Warning: no optimizer state in checkpoint, using a fresh optimizer")
        if isinstance(ckpt, dict) and 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        else:
            print("  Warning: no scheduler state in checkpoint, LR schedule restarts")
        if isinstance(ckpt, dict) and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt.get('epoch', 0)) if isinstance(ckpt, dict) else 0
        del ckpt
        print(f"  Restored state, continuing at epoch {start_epoch + 1}/{args.epochs} "
              f"(lr {optimizer.param_groups[0]['lr']:.6g})")
        if start_epoch >= args.epochs:
            print(f"  Checkpoint is already at epoch {start_epoch} >= --epochs {args.epochs}; "
                  f"increase --epochs to train further.")
            return
    elif args.init_from is not None:
        print(f"\nInitialising weights from {args.init_from}")
        ckpt = torch.load(args.init_from, map_location='cpu')
        result = model.load_state_dict(extract_model_state(ckpt), strict=False)
        print(f"  Missing keys: {len(result.missing_keys)}, unexpected keys: {len(result.unexpected_keys)}")
        del ckpt

    # Loss
    def depth_l1_loss(pred, gt, mask=None):
        return torch.abs(pred[:,:,:,0] - gt).mean()

    def depth_l2_loss(pred, gt, mask=None):
        return ((pred[:,:,:,0] - gt) ** 2).mean()

    def depth_huber_loss(pred, gt, mask=None):
        return torch.sqrt((pred[:,:,:,0] - gt) ** 2 + 1e-6).mean()

    def compute_train_loss(pred, gt, mask=None):
        return depth_l1_loss(pred, gt) + 0.5 * depth_l2_loss(pred, gt) + 0.5 * depth_huber_loss(pred, gt)

    # Training loop
    print("\nTraining...")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_train_loss = 0.0
        count = 0

        batch_idx = 0
        for batch in train_loader:
            images = batch['images'].to(device).float()
            extrinsics = batch['extrinsics'].to(device)
            intrinsics = batch['intrinsics'].to(device)
            depths = batch['depths'].to(device)

            relative_extrinsics = compute_relative_extrinsics(extrinsics)

            B, S, C, H, W = images.shape

            optimizer.zero_grad()

            start_time = time.time()
            with autocast(enabled=True):

                output = model(
                    x=images,
                    extrinsics=relative_extrinsics,
                    intrinsics=intrinsics
                )

                pred_depth = output.get('depth')
                if pred_depth is not None:
                    if len(pred_depth.shape) == 3:
                        pred_depth = pred_depth.view(B, S, H, W)

                    # This is average loss over all samples in batch (maybe I need more juste th eloss on the parent image?)
                    train_loss = 0.0
                    for s in range(S):
                        train_loss += compute_train_loss(
                            pred_depth[:, s],
                            depths[:, s],
                        )
                    train_loss /= S

                    depth_loss = 0.0
                    for s in range(S):
                        depth_loss += depth_l1_loss(
                            pred_depth[:, s],
                            depths[:, s],
                        )
                    depth_loss /= S

                else:
                    continue
            print(
               "Epoch {}/{}, Iter {}/{}, {}, lr {:.6f}, depth loss = {:.9f}, train loss = {:.9f}, time = {:.3f}".format(
                   epoch, args.epochs, batch_idx, len(train_loader), batch['parent_name'],
                   optimizer.param_groups[0]["lr"], depth_loss, train_loss,
                   time.time() - start_time))

            scaler.scale(train_loss).backward()
            if args.grad_clip > 0:
                # Unscale first so the clipping threshold applies to the true gradients
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_train_loss += train_loss.item()

            count += 1
            batch_idx += 1

        epoch_train_loss /= count if count > 0 else 1


        print(f"Epoch {epoch + 1}/{args.epochs} | Train: {epoch_train_loss:.4f}")

        # Save checkpoint
        checkpoint = {
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'train_loss': epoch_train_loss,
            }
        torch.save(checkpoint, output_dir / f'epoch_{epoch + 1}.pt')
        print(f"  Saved checkpoint: epoch_{epoch + 1}.pt")

    print("\nDone!")


if __name__ == '__main__':
    main()
