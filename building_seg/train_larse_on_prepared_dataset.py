from __future__ import annotations

import argparse
import distutils.version  # noqa: F401 - needed by older torch/lightning imports.
import json
import os
import random
import sys
import time
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LaRSE on a prepared image/mask dataset.")
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--larse-dir", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--backbone", default="clip_vitb32_384")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-val-samples", type=int, default=None)
    parser.add_argument("--freeze-clip", action="store_true", help="Freeze RemoteCLIP text/image model weights.")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze ViT image backbone and train only segmentation scratch/head layers.")
    parser.add_argument("--save-every-epochs", type=int, default=1)
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_class_names(dataset_dir: Path) -> list[str]:
    metadata_path = dataset_dir / "metadata" / "dataset.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    class_names = metadata.get("class_names")
    if not class_names or class_names[0] != "background":
        raise ValueError(f"{metadata_path} must contain class_names with background at index 0")
    return [str(name) for name in class_names]


def resolve_sample_ids(dataset_dir: Path, split: str, limit: int | None) -> list[str]:
    split_path = dataset_dir / "splits" / f"{split}.txt"
    if split_path.exists():
        sample_ids = [line.strip() for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        sample_ids = sorted(path.stem for path in (dataset_dir / "images").glob("*") if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    sample_ids = [Path(sample_id).stem for sample_id in sample_ids]
    if limit is not None:
        sample_ids = sample_ids[:limit]
    return sample_ids


def find_image_path(dataset_dir: Path, sample_id: str) -> Path:
    image_dir = dataset_dir / "images"
    for suffix in [".png", ".jpg", ".jpeg"]:
        path = image_dir / f"{sample_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Missing image for sample {sample_id} under {image_dir}")


def find_mask_path(dataset_dir: Path, sample_id: str) -> Path:
    mask_dir = dataset_dir / "masks"
    path = mask_dir / f"{sample_id}.png"
    if path.exists():
        return path
    raise FileNotFoundError(f"Missing mask for sample {sample_id}: {path}")


class PreparedBuildingSegDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root,
        split="train",
        mode=None,
        transform=None,
        target_transform=None,
        class_names=None,
        max_train_samples=None,
        max_val_samples=None,
        **kwargs,
    ):
        self.root = Path(root)
        self.split = split
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.class_names = class_names or load_class_names(self.root)
        self.num_class = len(self.class_names)
        self.NUM_CLASS = self.num_class
        limit = max_train_samples if split == "train" else max_val_samples
        self.sample_ids = resolve_sample_ids(self.root, split, limit)
        if not self.sample_ids:
            raise RuntimeError(f"Found 0 samples for split={split} under {self.root}")

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        image = Image.open(find_image_path(self.root, sample_id)).convert("RGB")
        mask = Image.open(find_mask_path(self.root, sample_id))
        if self.transform is not None:
            image = self.transform(image)
        mask_arr = np.asarray(mask, dtype=np.int64)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[..., 0]
        target = torch.from_numpy(mask_arr)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target, sample_id

    def __len__(self):
        return len(self.sample_ids)

    @property
    def pred_offset(self):
        return 0


def install_larse_runtime_patches(args, class_names: list[str]):
    larse_dir = Path(args.larse_dir).resolve()
    if not (larse_dir / "modules" / "lseg_module.py").exists():
        raise FileNotFoundError(f"Invalid LaRSE dir: {larse_dir}")
    if str(larse_dir) not in sys.path:
        sys.path.insert(0, str(larse_dir))

    import encoding.datasets as enc_datasets
    import data as larse_data

    dataset_kwargs = {
        "class_names": class_names,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
    }

    class RegisteredPreparedBuildingSegDataset(PreparedBuildingSegDataset):
        def __init__(self, *dargs, **dkwargs):
            dkwargs.update(dataset_kwargs)
            super().__init__(*dargs, **dkwargs)

    RegisteredPreparedBuildingSegDataset.NUM_CLASS = len(class_names)
    enc_datasets.datasets["buff1w"] = RegisteredPreparedBuildingSegDataset
    if "buff1w" not in larse_data.encoding_datasets:
        import functools

        larse_data.encoding_datasets["buff1w"] = functools.partial(enc_datasets.get_dataset, "buff1w")

    from modules.lseg_module import LSegModule

    def get_labels(self, dataset):
        print("Use prepared dataset labels:", class_names)
        return class_names

    LSegModule.get_labels = get_labels
    return LSegModule


def make_remoteclip_redirect_torch_load(original_torch_load, remoteclip_path: Path):
    def redirected_torch_load(f, *args, **kwargs):
        if isinstance(f, (str, os.PathLike)):
            path = Path(f)
            if path.name == "RemoteCLIP-ViT-B-32.pt" and not path.exists():
                print(f"Redirect RemoteCLIP checkpoint: {path} -> {remoteclip_path}")
                f = str(remoteclip_path)
        return original_torch_load(f, *args, **kwargs)

    return redirected_torch_load


def instantiate_larse_module(args, LSegModule, class_names: list[str]):
    larse_dir = Path(args.larse_dir).resolve()
    remoteclip_path = larse_dir / "checkpoints" / "RemoteCLIP-ViT-B-32.pt"
    if not remoteclip_path.exists():
        raise FileNotFoundError(f"RemoteCLIP checkpoint not found: {remoteclip_path}")

    old_cwd = Path.cwd()
    original_torch_load = torch.load
    os.chdir(larse_dir)
    try:
        torch.load = make_remoteclip_redirect_torch_load(original_torch_load, remoteclip_path)
        module = LSegModule(
            data_path=str(Path(args.dataset_dir).resolve()),
            dataset="buff1w",
            backbone=args.backbone,
            aux=False,
            num_features=256,
            aux_weight=0,
            se_loss=False,
            se_weight=0,
            base_lr=args.lr,
            batch_size=args.batch_size,
            max_epochs=args.epochs,
            ignore_index=-1,
            dropout=0.0,
            scale_inv=True,
            augment=False,
            no_batchnorm=False,
            widehead=False,
            widehead_hr=False,
            arch_option=0,
            block_depth=0,
            activation="lrelu",
            weight_decay=args.weight_decay,
            midasproto=False,
        )
    finally:
        torch.load = original_torch_load
        os.chdir(old_cwd)
    module.num_classes = len(class_names)
    module.nclass = len(class_names)
    return module


def load_filtered_checkpoint(module, checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model_state = module.state_dict()
    loadable = {}
    skipped = {}
    for key, value in state_dict.items():
        if key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            loadable[key] = value
        else:
            skipped[key] = {
                "checkpoint_shape": list(value.shape) if hasattr(value, "shape") else None,
                "model_shape": list(model_state[key].shape) if key in model_state else None,
            }
    missing, unexpected = module.load_state_dict(loadable, strict=False)
    print(f"Filtered checkpoint load: loaded={len(loadable)} skipped={len(skipped)} missing={len(missing)} unexpected={len(unexpected)}")
    return {
        "loaded_keys": sorted(loadable),
        "skipped_keys": skipped,
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def apply_freezing(module, freeze_clip: bool, freeze_backbone: bool):
    if freeze_clip and hasattr(module.net, "clip_pretrained"):
        for param in module.net.clip_pretrained.parameters():
            param.requires_grad = False
        print("Frozen module.net.clip_pretrained")
    if freeze_backbone and hasattr(module.net, "pretrained"):
        for param in module.net.pretrained.parameters():
            param.requires_grad = False
        print("Frozen module.net.pretrained")


def install_quiet_steps(module):
    def training_step(self, batch, batch_nb):
        img, target, _name = batch
        out = self(img)
        multi_loss = isinstance(out, tuple)
        loss = self.criterion(*out, target) if multi_loss else self.criterion(out, target)
        final_output = out[0] if multi_loss else out
        train_pred, train_gt = self._filter_invalid(final_output, target)
        if train_gt.nelement() != 0:
            self.train_accuracy(train_pred, train_gt)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_nb):
        img, target, _name = batch
        out = self(img)
        multi_loss = isinstance(out, tuple)
        val_loss = self.criterion(*out, target) if multi_loss else self.criterion(out, target)
        final_output = out[0] if multi_loss else out
        valid_pred, valid_gt = self._filter_invalid(final_output, target)
        self.val_iou.update(target, final_output)
        pix_acc, iou = self.val_iou.get()
        self.log("val_loss", val_loss, prog_bar=True, on_epoch=True)
        self.log("val_iou", iou, prog_bar=True, on_epoch=True)
        self.log("pix_acc", pix_acc, prog_bar=True, on_epoch=True)
        if valid_gt.nelement() != 0:
            self.log("val_acc", self.val_accuracy(valid_pred, valid_gt), prog_bar=True, on_epoch=True)

    module.training_step = MethodType(training_step, module)
    module.validation_step = MethodType(validation_step, module)


def install_dataloaders(module, num_workers: int):
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            worker_init_fn=lambda worker_id: random.seed(time.time() + worker_id),
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    module.train_dataloader = MethodType(train_dataloader, module)
    module.val_dataloader = MethodType(val_dataloader, module)


def save_lightweight_copy(src: Path, dst: Path):
    if src.exists():
        data = torch.load(src, map_location="cpu")
        torch.save(data, dst)


def main():
    args = parse_args()
    seed_everything(args.seed)

    dataset_dir = Path(args.dataset_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    class_names = load_class_names(dataset_dir)
    (out_dir / "class_names.json").write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    LSegModule = install_larse_runtime_patches(args, class_names)
    module = instantiate_larse_module(args, LSegModule, class_names)
    load_report = load_filtered_checkpoint(module, Path(args.init_checkpoint).resolve())
    (out_dir / "checkpoint_load_report.json").write_text(json.dumps(load_report, ensure_ascii=False, indent=2), encoding="utf-8")
    apply_freezing(module, args.freeze_clip, args.freeze_backbone)
    install_quiet_steps(module)
    install_dataloaders(module, args.num_workers)

    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    logger = CSVLogger(save_dir=str(out_dir), name="logs")
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(out_dir),
        filename="best",
        monitor="val_iou",
        mode="max",
        save_last=True,
        save_top_k=1,
        every_n_epochs=args.save_every_epochs,
    )

    trainer_kwargs = {
        "max_epochs": args.epochs,
        "logger": logger,
        "callbacks": [checkpoint_callback],
        "log_every_n_steps": 20,
    }
    if args.device.startswith("cuda") and torch.cuda.is_available():
        trainer_kwargs["gpus"] = 1
    else:
        trainer_kwargs["gpus"] = 0

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module)

    best_path = Path(checkpoint_callback.best_model_path) if checkpoint_callback.best_model_path else out_dir / "best.ckpt"
    last_path = out_dir / "last.ckpt"
    if best_path.exists() and best_path.name != "best.ckpt":
        save_lightweight_copy(best_path, out_dir / "best.ckpt")
    if last_path.exists():
        print(f"Last checkpoint: {last_path}")
    print(f"Best checkpoint: {best_path}")
    print(f"Training outputs written to {out_dir}")


if __name__ == "__main__":
    main()
