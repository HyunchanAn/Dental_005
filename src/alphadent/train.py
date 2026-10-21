# coding: utf-8
__author__ = "ZFTurbo: https://github.com/ZFTurbo"

import argparse
import os
import torch
from ultralytics import YOLO

os.environ["WANDB_DISABLED"] = "true"


def _resolve_yolo_device():
    env_dev = os.getenv("DEVICE", "auto").strip().lower()
    if env_dev and env_dev != "auto":
        return env_dev
    return [0] if torch.cuda.is_available() else "cpu"


def train_seg_yolo(args):
    model = YOLO("yolov8x-seg.pt")
    device = _resolve_yolo_device()

    results = model.train(
        data=args.dataset_config,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch_size,
        project="yolo_seg_x_proj_{}".format(args.image_size),
        deterministic=False,
        plots=True,
        device=device,
    )

    print(results)


if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_use = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    print("Torch: {} Cuda is available: {}".format(torch.__version__, torch.cuda.is_available()))
    code_path = os.path.dirname(os.path.abspath(__file__)) + "/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config",
        type=str,
        default=code_path + "AlphaDent/yolo_seg_train.yaml",
        help="Path to yolo_seg_train.yaml for AlphaDent dataset",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train")
    parser.add_argument("--image_size", type=int, default=640, help="Image size in pixels for model input")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for model. Set according to your GPU memory"
    )

    args = parser.parse_args()
    print("Input arguments:", args)

    train_seg_yolo(args)
