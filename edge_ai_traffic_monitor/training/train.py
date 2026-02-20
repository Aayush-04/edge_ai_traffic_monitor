#!/usr/bin/env python3
"""
train.py
YOLOv3 training script using Ultralytics.

"""
import argparse
import os
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOv3 Training for FPGA Deployment')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to data.yaml (dataset config)')
    parser.add_argument('--weights', type=str, default='yolov3-tiny.pt',
                        help='Pretrained weights path or model name')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image resolution')
    parser.add_argument('--device', type=str, default='',
                        help='CUDA device (e.g., "0" or "cpu")')
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Output directory for training results')
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("YOLOv3 Training for FPGA Deployment")
    logger.info("=" * 60)
    logger.info(f"  Dataset:    {args.data}")
    logger.info(f"  Weights:    {args.weights}")
    logger.info(f"  Epochs:     {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Image size: {args.img_size}")
    logger.info(f"  Device:     {args.device or 'auto'}")

    # Validate dataset path
    if not os.path.isfile(args.data):
        logger.error(f"Dataset config not found: {args.data}")
        sys.exit(1)

    # Import Ultralytics 
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("Ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    # Create model and train 
    model = YOLO(args.weights)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        device=args.device if args.device else None,
        project=args.project,
        verbose=True,
    )

    logger.info("Training complete.")
    logger.info(f"Best weights: {args.project}/exp/weights/best.pt")
    logger.info("Next step: quantize with training/quantize.py")


if __name__ == '__main__':
    main()