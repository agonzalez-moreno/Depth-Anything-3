#!/usr/bin/env python
"""
Minimal inference for Depth Anything 3 on the Toulouse canopy dataset.
Runs the model and saves each predicted depth map as a float32 .tif.

Usage:
    python infer_simple.py --dataset-root /path/to/dataset --checkpoint outputs_simple/epoch_10.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from canopy_dataset import CanopyDataset, DataField
from depth_anything_3.cfg import load_config, create_object
from depth_anything_3.registry import MODEL_REGISTRY
from train_simple import compute_relative_extrinsics


def collate_fn(batch):
    return {
    	'parent_name': [b[DataField.PARENT_NAME] for b in batch],
        'images': torch.stack([b[DataField.IMAGES] for b in batch]),
        'extrinsics': torch.stack([b[DataField.EXTRINSICS_C2W] for b in batch]),
        'intrinsics': torch.stack([b[DataField.INTRINSICS] for b in batch]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-root', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--subset', type=str, default='synthetic_test_672x448')
    parser.add_argument('--model', type=str, default='da3-small', choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument('--num-pairs', type=int, default=3)
    parser.add_argument('--output', type=str, default='./predictions')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = CanopyDataset(
        dataset_root=args.dataset_root,
        subset=args.subset,
        img_size=672,
        num_pairs=args.num_pairs,
        use_depth=False,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    model = create_object(load_config(MODEL_REGISTRY[args.model])).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(loader):
            images = batch['images'].to(device).float()
            extrinsics = compute_relative_extrinsics(batch['extrinsics'].to(device))
            intrinsics = batch['intrinsics'].to(device)
            B, S, C, H, W = images.shape

            with torch.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
                out = model(x=images, extrinsics=extrinsics, intrinsics=intrinsics)

            depth = out['depth'].float().reshape(B, S, H, W).cpu().numpy() * 1400.0


            Image.fromarray(depth[0, 0].astype(np.float32), mode='F').save(
                output_dir / f'{batch["parent_name"][0]}.tif')

            print(f'{i + 1}/{len(loader)}')

    print(f'Done. Depth maps saved to {output_dir}')


if __name__ == '__main__':
    main()
