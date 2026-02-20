#!/usr/bin/env python3
"""
quantize.py
Vitis AI INT8 quantization for DPU deployment.

"""
import argparse
import os
import sys
import glob
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Vitis AI INT8 Quantization')
    parser.add_argument('-w', '--weights', type=str, required=True,
                        help='Trained FP32 weights (.pt)')
    parser.add_argument('-d', '--dataset', type=str, required=True,
                        help='Calibration dataset directory (images)')
    parser.add_argument('-q', '--quant-mode', type=str, required=True,
                        choices=['calib', 'test'],
                        help='Quantization mode: calib (calibrate) or test (export)')
    parser.add_argument('-b', '--batch-size', type=int, default=1,
                        help='Batch size for calibration')
    parser.add_argument('-o', '--output-dir', type=str, default='build/quant_model',
                        help='Output directory for quantized model')
    return parser.parse_args()


def load_calibration_images(dataset_dir, img_size=640, max_images=200):
    """
    Load calibration images for quantization.
    """
    import cv2
    import numpy as np
    import torch

    image_files = sorted(
        glob.glob(os.path.join(dataset_dir, 'images', '*.jpg')) +
        glob.glob(os.path.join(dataset_dir, 'images', '*.png')) +
        glob.glob(os.path.join(dataset_dir, '*.jpg')) +
        glob.glob(os.path.join(dataset_dir, '*.png'))
    )[:max_images]

    if not image_files:
        raise FileNotFoundError(f"No images found in {dataset_dir}")

    logger.info(f"Loading {len(image_files)} calibration images...")
    images = []
    for f in image_files:
        img = cv2.imread(f)
        if img is None:
            continue
        img = cv2.resize(img, (img_size, img_size))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC → CHW
        images.append(img)

    return torch.tensor(np.array(images))


def quantize(args):
    """
    Run Vitis AI quantization.
    """
    import torch
    from pytorch_nndct.apis import torch_quantizer

    # Load trained model — preserves original model loading
    logger.info(f"Loading model from {args.weights}")
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.nn.autobackend import DetectMultiBackend
    model = DetectMultiBackend(weights=args.weights)
    model.eval()

    # Create random input for quantizer initialization
    rand_input = torch.randn(1, 3, 640, 640)

    # Create quantizer — preserves NNDCT behavior
    logger.info(f"Creating quantizer (mode={args.quant_mode})")
    os.makedirs(args.output_dir, exist_ok=True)

    quantizer = torch_quantizer(
        args.quant_mode,
        model,
        rand_input,
        output_dir=args.output_dir,
    )
    quant_model = quantizer.quant_model

    if args.quant_mode == 'calib':
        # Calibration: run inference on validation images
        logger.info("Running calibration...")
        calib_images = load_calibration_images(args.dataset)

        with torch.no_grad():
            for i in range(0, len(calib_images), args.batch_size):
                batch = calib_images[i:i + args.batch_size]
                quant_model(batch)
                if (i // args.batch_size) % 20 == 0:
                    logger.info(f"  Calibrated {i + len(batch)}/{len(calib_images)} images")

        quantizer.export_quant_config()
        logger.info(f"Calibration complete. Config saved to {args.output_dir}")

    elif args.quant_mode == 'test':
        # Export quantized xmodel
        logger.info("Exporting quantized xmodel...")
        calib_images = load_calibration_images(args.dataset, max_images=10)
        with torch.no_grad():
            quant_model(calib_images[:1])

        quantizer.export_xmodel(output_dir=args.output_dir)
        logger.info(f"xmodel exported to {args.output_dir}")
        logger.info("Next: compile with vai_c_xir for ZCU104 DPU")


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Vitis AI INT8 Quantization")
    logger.info("=" * 60)
    logger.info(f"  Weights:    {args.weights}")
    logger.info(f"  Dataset:    {args.dataset}")
    logger.info(f"  Mode:       {args.quant_mode}")
    logger.info(f"  Output:     {args.output_dir}")

    if not os.path.isfile(args.weights):
        logger.error(f"Weights file not found: {args.weights}")
        sys.exit(1)

    quantize(args)


if __name__ == '__main__':
    main()