import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Diffusion Model for Human Image Generation")

    # General
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'generate'],
                        help='Mode to run: train or generate')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save checkpoints and generated images')
    parser.add_argument('-seed', '--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Dataset
    parser.add_argument('-d', '--data_dir', type=str, required=True,
                        help='Directory containing the dataset (images and corresponding data)')
    parser.add_argument('--image_size', type=int, default=128,
                        help='Input image size (e.g., 128x128)')
    parser.add_argument('--channels', type=int, default=3,
                        help='Number of image channels (e.g., 3 for RGB)')
    parser.add_argument("-ct", "--cond_type", type=str, default="face_pose",
                        choices=["face_pose", "pose_only"],
                        help="Type of conditional input to use. 'face_pose' (1536 dim) or 'pose_only' (132 dim).")


    # Model Parameters (for UNet)
    parser.add_argument('-ts', '--timesteps', type=int, default=1000,
                        help='Number of diffusion timesteps')
    parser.add_argument('--beta_start', type=float, default=0.0001,
                        help='Start beta value for diffusion schedule')
    parser.add_argument('--beta_end', type=float, default=0.02,
                        help='End beta value for diffusion schedule')
    parser.add_argument('--beta_schedule', type=str, default='linear', choices=['linear', 'cosine'],
                        help='Beta schedule for diffusion process.')
    parser.add_argument('-md', '--model_dim', type=int, required=True, # U-Netの最初の層の次元
                        help='Base dimension for the UNet model (channels).')
    parser.add_argument('--time_emb_dim', type=int, default=256,
                        help='Dimension of the time embedding.')
    parser.add_argument('--num_down_blocks', type=int, default=3,
                        help='Number of downsampling blocks in UNet.')
    parser.add_argument('--attention_depths', nargs='+', type=int, default=[1, 2],
                        help='Depths at which to apply attention blocks (0-indexed downsampling blocks).')


    # Training
    parser.add_argument('-e', '--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('-bs', '--batch_size', type=int, required=True,
                        help='Batch size for training')
    parser.add_argument('-gas', '--gradient_accumulation_steps', type=int, default=16, # デフォルト値を設定
                        help='Number of steps to accumulate gradients before updating model parameters.')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001,
                        help='Learning rate for optimizer')
    parser.add_argument('-nv', '--n-val-samples', type=int, default=None, 
                        help='Number of validation data. None for use --n-val-samples-rate.')
    parser.add_argument('-nvr', '--n-val-samples-rate', type=float, default=0.1,
                        help='Rate of validation data (available if --n-val-samples is None)')
    parser.add_argument('-n', '--n-samples', type=int, default=None,
                        help='Number of random samples to use per epoch. If None, use all available data.')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of workers for data loading')

    # Generation
    parser.add_argument('-ngi', '--num-generate-images', type=int, default=4,
                        help='Number of images to generate')
    parser.add_argument('-gbs', '--generate-batch-size', type=int, default=2,
                        help='Batch size in generation')
    parser.add_argument('--checkpoint_path', type=str,
                        help='Path to the model checkpoint for generation')

    args = parser.parse_args()

    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'generated_images'), exist_ok=True)

    return args