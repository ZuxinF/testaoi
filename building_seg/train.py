from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from building_seg.dataset import PatchSegDataset
from building_seg.models import build_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a building function segmentation model.")
    parser.add_argument("--dataset", required=True, help="Prepared dataset directory")
    parser.add_argument("--out", required=True, help="Output checkpoint path")
    parser.add_argument("--model", default="tiny_unet")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def pixel_accuracy(logits, target):
    pred = torch.argmax(logits, dim=1)
    return (pred == target).float().mean().item()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    count = 0
    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device)
            mask = mask.to(device)
            logits = model(image)
            loss = criterion(logits, mask)
            total_loss += loss.item()
            total_acc += pixel_accuracy(logits, mask)
            count += 1
    return total_loss / max(count, 1), total_acc / max(count, 1)


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset)
    metadata = json.loads((dataset_dir / "metadata" / "dataset.json").read_text())
    class_names = metadata["class_names"]

    train_ds = PatchSegDataset(dataset_dir, "train")
    val_ds = PatchSegDataset(dataset_dir, "val")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_kwargs = {"base_channels": args.base_channels}
    model = build_model(args.model, num_classes=len(class_names), **model_kwargs).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        for image, mask in train_loader:
            image = image.to(args.device)
            mask = mask.to(args.device)
            optimizer.zero_grad()
            logits = model(image)
            loss = criterion(logits, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += pixel_accuracy(logits, mask)
            n_batches += 1

        val_loss, val_acc = evaluate(model, val_loader, criterion, args.device)
        print(
            f"Epoch {epoch}/{args.epochs} "
            f"train_loss={train_loss/max(n_batches,1):.4f} "
            f"train_acc={train_acc/max(n_batches,1):.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss <= best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_name": args.model,
                    "model_kwargs": model_kwargs,
                    "model_state": model.state_dict(),
                    "class_names": class_names,
                    "metadata": metadata,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                },
                out_path,
            )
            print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
