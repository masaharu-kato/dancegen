from pathlib import Path
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.utils import save_image
from tqdm import tqdm
import os

from model import UNet, Diffusion
from dataset import CustomDanceDataset
from utils import setup_seed # Assuming utils.py has these

def train_model(args):
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    pose_only = str(args.cond_type) == "pose_only"
    num_workers = args.num_workers or (os.cpu_count() or 2) - 1
    num_generate_images = int(args.num_generate_images or 1)
    generate_batch_size = int(args.generate_batch_size or 1)

    # Output directory setup
    output_dir.mkdir(exist_ok=True, parents=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)
    images_dir = output_dir / "generated_images"
    images_dir.mkdir(exist_ok=True)

    # Check for existing training state and resume
    last_checkpoint = None
    if os.path.exists(checkpoints_dir):
        checkpoints = sorted([f for f in os.listdir(checkpoints_dir) if f.startswith('model_epoch_') and f.endswith('.pt')])
        if checkpoints:
            last_checkpoint = os.path.join(checkpoints_dir, checkpoints[-1])
            print(f"Resuming from checkpoint: {last_checkpoint}")


    full_dataset = CustomDanceDataset(
        data_dir=data_dir, # 全データを含むルートディレクトリを指定
        # image_size=args.image_size,
        pose_only=pose_only,
    )

    # --- シード値に基づいてトレーニングと検証に分割 ---
    dataset_size = len(full_dataset)
    val_size = min(int(args.n_val_samples or dataset_size * args.n_val_samples_rate), dataset_size * 0.5)
    train_size = dataset_size - val_size
    n_samples = min(args.n_samples or train_size, train_size)
    print(f"dataset_size: {dataset_size}, val_size: {val_size}, train_size(all): {train_size}, n_samples: {n_samples}")

    # PyTorch の乱数生成器を使用 (シード値に基づいて再現可能)
    g = torch.Generator().manual_seed(args.seed) 
    
    # 全インデックスのランダムな順列を生成
    all_indices = torch.randperm(dataset_size, generator=g).tolist()

    # インデックスを分割
    train_indices = all_indices[val_size:]
    val_indices = all_indices[:val_size]

    print("Images to be generated: ", ', '.join(full_dataset.image_files[i] for i in val_indices[:num_generate_images]))

    # Model, Optimizer, etc.
    unet_model = UNet(
        model_dim=args.model_dim,
        cond_dim=full_dataset.cond_dim,
        time_emb_dim=args.time_emb_dim,
        num_down_blocks=args.num_down_blocks,
        att_depths=args.attention_depths,
    ).to(device)

    diffusion = Diffusion(
        model=unet_model,
        image_size=args.image_size,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        beta_schedule=args.beta_schedule,
    ).to(device)

    optimizer = AdamW(diffusion.parameters(), lr=args.learning_rate)
    scaler = torch.amp.grad_scaler.GradScaler("cuda")

    start_epoch = 0
    if last_checkpoint:
        checkpoint = torch.load(last_checkpoint, map_location=device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict']) # Load scaler state
        start_epoch = checkpoint['epoch'] + 1
        print(f"Loaded model from epoch {checkpoint['epoch']}, continuing training from epoch {start_epoch}")

    # --- Gradient Accumulation setting ---
    # 勾配を何ステップ蓄積するか。これを大きくすると実質的なバッチサイズが大きくなる。
    # 例: batch_size=4, gradient_accum_steps=8 なら、実質バッチサイズは 32
    gradient_accum_steps = args.gradient_accumulation_steps # argsに追加する
    
    # バリデーション用 DataLoader (各エポックで再構築する必要はないが、コードの一貫性のため)
    val_sampler = SubsetRandomSampler(val_indices) # バリデーションはシャッフルしない
    val_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):

        train_subset_idxes = torch.tensor(train_indices, dtype=torch.long)[torch.randperm(len(train_indices), generator=g)[:n_samples]].tolist()
        train_sampler = SubsetRandomSampler(train_subset_idxes, generator=g)
        train_dataloader = DataLoader(full_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)
        
        # Training
        diffusion.train()
        total_loss = 0

        optimizer.zero_grad() # エポックの開始時に一度だけ勾配をクリア

        dataloader_with_tqdm = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}", dynamic_ncols=True)
        for i, (images, _cond_data) in enumerate(dataloader_with_tqdm):
            images = images.to(device)
            cond_data = _cond_data.to(device)

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                loss = diffusion(images, cond_data)
            total_loss += loss.item()

            scaler.scale(loss).backward() # スケーリングされた勾配でバックプロパゲーション

            if (i + 1) % gradient_accum_steps == 0 or i == len(train_dataloader) - 1:
                scaler.step(optimizer) # optimizer.step() の代わりに scaler.step()
                scaler.update() # スケーラーの更新
                optimizer.zero_grad() # 勾配をクリア

            loss_v = loss.item()
            dataloader_with_tqdm.set_postfix(loss=f"{loss_v:.4f}")
            

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoints_dir, f"model_epoch_{epoch:03d}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': diffusion.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(), # Scalerの状態も保存
            'loss': avg_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")


        # Validation
        diffusion.eval()
        total_val_loss = 0
        all_val_cond_data: list[torch.Tensor] = []
        val_pbar = tqdm(val_dataloader, desc=f"Epoch {epoch}/{args.epochs} [Validation]", dynamic_ncols=True)
        with torch.no_grad():
            for images, _cond_data in val_pbar:
                images = images.to(device)
                cond_data = _cond_data.to(device)

                all_val_cond_data.append(cond_data.cpu()) 

                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    val_loss = diffusion(images, cond_data)
                
                val_loss_v = val_loss.item()
                total_val_loss += val_loss_v
                val_pbar.set_postfix(val_loss=f"{val_loss_v:.4f}")
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Epoch {epoch} finished. Average Validation Loss: {avg_val_loss:.4f}")

        diffusion.eval()
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            generate_cond = torch.cat(all_val_cond_data, dim=0).to(device)[:num_generate_images]
            generated_images = []
            for i in tqdm(range(0, len(generate_cond), generate_batch_size), desc="Generating images", dynamic_ncols=True):
                # diffusion.sample の第一引数は生成する画像の枚数、第二引数は条件データ
                c_generate_cond = generate_cond[i:i+generate_batch_size]
                generated_images.append(diffusion.sample(len(c_generate_cond), c_generate_cond))

        final_generated_images = torch.cat(generated_images, dim=0)
        image_path = os.path.join(images_dir, f"generated_epoch_{epoch:03d}.png")
        save_image(final_generated_images, image_path, nrow=int(num_generate_images**0.5) or 1) # 例えば、ルートで正方形に近づける
        print(f"Generated images saved to {image_path}")

    print("Training finished.")