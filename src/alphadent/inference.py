# coding: utf-8
__author__ = "ZFTurbo: https://github.com/ZFTurbo"

import argparse
import glob
import os
import time
import torch
from ultralytics import YOLO


def _resolve_yolo_device():
    env_dev = os.getenv("DEVICE", "auto").strip().lower()
    if env_dev and env_dev != "auto":
        return env_dev
    return "0" if torch.cuda.is_available() else "cpu"


def predict_test_seg_yolo(args):
    BATCH_SIZE = args.batch_size
    device = _resolve_yolo_device()
    use_half = device != "cpu"
    model = YOLO(args.weights)  # load a custom model
    images = glob.glob(args.input_path + "/**/*.jpg", recursive=True) + glob.glob(
        args.input_path + "/**/*.png", recursive=True
    )
    print("Folder location: {}".format(args.input_path))
    print("Images found in folder: {}".format(len(images)))

    output_path = args.output_path
    print("Output path: {}".format(output_path))

    for part_id in range(0, len(images), BATCH_SIZE):
        images_part = images[part_id : part_id + BATCH_SIZE]

        start_time = time.time()
        results = model.predict(
            images_part,
            project=output_path,
            name="preds",
            imgsz=args.image_size,
            conf=args.conf,
            iou=args.iou,
            half=use_half,
            device=device,
            max_det=300,
            save=True,
        )
        print("Processed images: {} Time: {:.2f} sec".format(len(results), time.time() - start_time))
        for i, result in enumerate(results):
            image_name = images[part_id + i]
            file_name = "{}".format(os.path.basename(image_name)[:-4])
            print(file_name)
            result.save_txt(output_path + file_name + ".txt")


if __name__ == "__main__":
    if torch.cuda.is_available():
        gpu_use = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
    print("Torch: {} Cuda is available: {}".format(torch.__version__, torch.cuda.is_available()))
    code_path = os.path.dirname(os.path.abspath(__file__)) + "/"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights",
        type=str,
        default=code_path + "weights/yolov8x_AlphaDent_9_classes_640px.pt",
        help="Path to file with weights (.pt)",
    )
    parser.add_argument(
        "--input_path", type=str, default=code_path + "input/", help="Path to directory with images to process"
    )
    parser.add_argument(
        "--output_path", type=str, default=code_path + "output/", help="Path to directory where to store results"
    )
    parser.add_argument("--image_size", type=int, default=640, help="Input image size for model")
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for model (how many images process at once)"
    )
    parser.add_argument("--iou", type=float, default=0.6, help="Intersection over Union for NMS")
    parser.add_argument("--conf", type=float, default=0.1, help="Save all boxes with confidence larger than this value")
    args = parser.parse_args()
    print("Input arguments:", args)

    predict_test_seg_yolo(args)
