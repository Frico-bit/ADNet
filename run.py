import argparse
import glob
import os
import re

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from dataset.dataset import SegDataset
from model.adnet import ADNet

IMG_SIZE = 352
# MEAN = (0.485, 0.456, 0.406)
# STD = (0.229, 0.224, 0.225)


def build_dataframe(data_dir):
    images_dir = os.path.join(data_dir, "images")
    masks_dir = os.path.join(data_dir, "masks")

    rows = []
    for image_path in sorted(glob.glob(os.path.join(images_dir, "*"))):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        mask_matches = glob.glob(os.path.join(masks_dir, stem + ".*"))
        if not mask_matches:
            continue
        rows.append({"image": image_path, "mask": mask_matches[0]})

    if not rows:
        raise FileNotFoundError(
            f"No matching image/mask pairs found under {data_dir} "
            f"(expected an 'images/' and a 'masks/' subfolder with matching filenames)"
        )
    return pd.DataFrame(rows)


def dice_score(pred, target, eps=1e-7):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    return (2 * intersection + eps) / (pred.sum() + target.sum() + eps)


def iou_score(pred, target, eps=1e-7):
    pred = pred.reshape(-1)
    target = target.reshape(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + eps) / (union + eps)


def remap_legacy_state_dict_keys(state_dict):
    """Rename keys from older checkpoints to match the current ADNet module names:
    'bilateral_fusion{i}' -> 'cska{i}', and 'shared_bara.{discovery_net,refinement_h,
    refinement_l}' -> 'shared_bara.rsm.{...}'. Also drops the now-unused cached
    '*.attn' debug buffers that older checkpoints saved.
    """
    remapped = {}
    for key, value in state_dict.items():
        if key.endswith(".attn"):
            continue
        new_key = key.replace("bilateral_fusion", "cska")
        new_key = re.sub(
            r"(shared_bara)\.(discovery_net|refinement_h|refinement_l)",
            r"\1.rsm.\2",
            new_key,
        )
        remapped[new_key] = value
    return remapped


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "state_dict" not in checkpoint and "model" not in checkpoint:
        state_dict = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint.get("model"))
    state_dict = remap_legacy_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Run ADNet inference/evaluation on a folder of images.")
    parser.add_argument("--checkpoint", required=True, help="Path to a trained ADNet checkpoint (.pth).")
    parser.add_argument("--data_dir", required=True, help="Folder containing 'images/' and 'masks/' subfolders.")
    parser.add_argument("--output_dir", default="outputs", help="Where predicted masks are saved.")
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    transform = A.Compose([
        # A.Resize(args.img_size, args.img_size),
        A.Normalize(normalization="min_max"),
        ToTensorV2(),
    ])

    dataframe = build_dataframe(args.data_dir)
    dataset = SegDataset(dataframe, transform=transform)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = ADNet(pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    dice_scores = []
    iou_scores = []

    with torch.no_grad():
        for idx, (image, mask) in enumerate(loader):
            image = image.to(device)
            mask = mask.to(device)

            logits = model(image)
            pred = (torch.sigmoid(logits) > 0.5).float()

            dice = dice_score(pred.cpu().numpy(), mask.cpu().numpy())
            iou = iou_score(pred.cpu().numpy(), mask.cpu().numpy())
            dice_scores.append(dice)
            iou_scores.append(iou)

            image_name = os.path.splitext(os.path.basename(dataframe.loc[idx, "image"]))[0]
            pred_mask = (pred[0, 0].cpu().numpy() * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.output_dir, f"{image_name}_pred.png"), pred_mask)

            print(f"[{idx + 1}/{len(dataset)}] {image_name} - Dice: {dice:.4f}, IoU: {iou:.4f}")

    print(f"\nMean Dice: {np.mean(dice_scores):.4f}")
    print(f"Mean IoU:  {np.mean(iou_scores):.4f}")


if __name__ == "__main__":
    main()
