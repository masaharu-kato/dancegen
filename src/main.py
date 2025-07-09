import os
from config import get_args
from train import train_model
from generate import generate_images

def main():
    args = get_args()

    print(f"Running in mode: {args.mode}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data directory: {args.data_dir}")

    if args.mode == 'train':
        train_model(args)
    elif args.mode == 'generate':
        generate_images(args)
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == "__main__":
    main()