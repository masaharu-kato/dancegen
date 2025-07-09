import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Helper Functions for U-Net ---
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Swish(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, time):
        # sinusoidal positional embedding
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return self.mlp(embeddings)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, cond_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU() # Default inplace=False, which is good.
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.cond_mlp = nn.Linear(cond_emb_dim, out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb, cond_emb):
        h = self.relu(self.bn1(self.conv1(x)))
        
        # 修正: インプレース操作を防ぐため、明示的な加算に
        time_add = self.time_mlp(time_emb)[:, :, None, None]
        h = h + time_add 

        cond_add = self.cond_mlp(cond_emb)[:, :, None, None]
        h = h + cond_add # 同様に修正
        
        h = self.relu(self.bn2(self.conv2(h)))
        return h + self.shortcut(x) # こちらは最終的なreturnなので問題ないことが多い

class AttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.q = nn.Conv2d(dim, dim, 1)
        self.k = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj_out = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(-1, -2) # B, (H*W), C
        k = self.k(h).flatten(2) # B, C, (H*W)
        v = self.v(h).flatten(2).transpose(-1, -2) # B, (H*W), C

        attn = (q @ k).softmax(dim=-1) # B, (H*W), (H*W)
        out = (attn @ v).transpose(-1, -2).reshape_as(x) # B, C, H, W
        return x + self.proj_out(out)

# --- U-Net Model ---
class UNet(nn.Module):
    def __init__(self,
                 model_dim,
                 cond_dim,
                 time_emb_dim,
                 num_down_blocks,
                 image_channels=3,
                 att_depths=[1, 2],
                 ):
        super().__init__()

        print("UNet cond_dim=", cond_dim)

        self.image_channels = image_channels
        self.time_emb = TimeEmbedding(time_emb_dim)
        self.cond_emb = nn.Linear(cond_dim, cond_dim)

        self.initial_conv = nn.Conv2d(image_channels, model_dim, 3, padding=1)

        # Encoder (Downsampling)
        self.downs = nn.ModuleList()
        # 各ダウンサンプリングブロックの後にスキップ接続として保存されるチャネル数を記録
        # (ダウンサンプリング層の出力チャネルではない点に注意)
        self.encoder_channels_for_skips = [] 
        
        curr_channels = model_dim
        for i in range(num_down_blocks):
            out_channels_res_block = curr_channels * 2 if i < num_down_blocks - 1 else curr_channels
            self.downs.append(nn.ModuleList([
                ResidualBlock(curr_channels, out_channels_res_block, time_emb_dim, cond_dim),
                AttentionBlock(out_channels_res_block) if i in att_depths else nn.Identity(),
                nn.Conv2d(out_channels_res_block, out_channels_res_block, 4, 2, 1) # Downsample layer
            ]))
            self.encoder_channels_for_skips.append(out_channels_res_block) # ここで保存されるチャネル数
            curr_channels = out_channels_res_block # Update for the next block's input

        # Middle
        self.mid_res1 = ResidualBlock(curr_channels, curr_channels, time_emb_dim, cond_dim)
        self.mid_att = AttentionBlock(curr_channels)
        self.mid_res2 = ResidualBlock(curr_channels, curr_channels, time_emb_dim, cond_dim)

        # Decoder (Upsampling)
        self.ups = nn.ModuleList()
        # curr_channels at this point is the channels of the deepest layer from encoder/middle
        
        # encoder_channels_for_skips は [64, 128, 256] (model_dim=64, num_down_blocks=3の場合)
        # デコーダーはこれを逆順にポップするので、このリストも逆順にしておく
        decoder_skip_channels = list(reversed(self.encoder_channels_for_skips)) 
        
        for i in range(num_down_blocks): # i は 0, 1, 2 ...
            # ConvTranspose2dの出力チャネル (アップサンプリング後の x のチャネル)
            # これは、次の ResidualBlock の最終的な出力チャネルになる
            decoder_upsample_out_channels = curr_channels // 2 if i < num_down_blocks - 1 else model_dim # Ensure final layer matches model_dim

            # スキップ接続のチャネル数
            # decoder_skip_channels[i] は、現在のデコーダー層に対応するスキップ接続のチャネル数
            skip_ch = decoder_skip_channels[i]

            # ResidualBlockへの入力チャネルは、アップサンプリング後の x とスキップ接続のチャネルの合計
            res_block_in_channels = decoder_upsample_out_channels + skip_ch
            
            self.ups.append(nn.ModuleList([
                nn.ConvTranspose2d(curr_channels, decoder_upsample_out_channels, 4, 2, 1), # Upsample
                ResidualBlock(res_block_in_channels, decoder_upsample_out_channels, time_emb_dim, cond_dim), # Input: upsampled_x + skip_x
                AttentionBlock(decoder_upsample_out_channels) if i in att_depths else nn.Identity(),
            ]))
            curr_channels = decoder_upsample_out_channels # Update for the next up block

        self.final_conv = nn.Conv2d(model_dim, image_channels, 3, padding=1)

    def forward(self, x, t, conditional_input):
        time_emb = self.time_emb(t)
        cond_emb = self.cond_emb(conditional_input)

        x = self.initial_conv(x)
        skips = []

        # Encoder
        for res_block, att_block, downsample in self.downs: # type: ignore
            x = res_block(x, time_emb, cond_emb)
            x = att_block(x)
            skips.append(x) # Save features *before* downsampling
            x = downsample(x)

        # Middle
        x = self.mid_res1(x, time_emb, cond_emb)
        x = self.mid_att(x) # Ensure this is called correctly
        x = self.mid_res2(x, time_emb, cond_emb)

        # Decoder
        for i, (upsample_layer, res_block, att_block) in enumerate(self.ups): # type: ignore
            # Upsample x first
            x = upsample_layer(x)

            # Pop the corresponding skip connection
            skip_connection = skips.pop()

            # --- CRITICAL FIX: Ensure spatial dimensions match ---
            if x.shape[2:] != skip_connection.shape[2:]:
                # Calculate the size difference (height, width)
                diff_h = skip_connection.shape[2] - x.shape[2]
                diff_w = skip_connection.shape[3] - x.shape[3]

                if diff_h > 0 or diff_w > 0:
                    # Crop symmetrically from the center of the larger tensor (skip_connection)
                    skip_connection = skip_connection[:, :, 
                                                      diff_h // 2 : skip_connection.shape[2] - (diff_h - diff_h // 2),
                                                      diff_w // 2 : skip_connection.shape[3] - (diff_w - diff_w // 2)]
                # If x is larger than skip_connection, this indicates an issue with ConvTranspose2d settings.
                # For (k=4, s=2, p=1), output_size = (input_size - 1)*stride - 2*padding + kernel_size = (input_size - 1)*2 + 2
                # This should always exactly double the input_size if input_size is a power of 2 minus 1 (e.g. 16 -> 32)
                # or if it's a power of 2, like 32.
                # E.g., for input 16, output is (16-1)*2+2 = 30+2 = 32. This is what we want.
                # So if x.shape[2:] is larger than skip_connection.shape[2:], it's a critical error.
                # For now, let's just make sure the print statement catches it clearly.
                elif diff_h < 0 or diff_w < 0:
                    print(f"ERROR: Upsampled X is larger than skip_connection! X: {x.shape}, Skip: {skip_connection.shape}")
                    raise RuntimeError("Upsampling resulted in larger image than skip connection. Check ConvTranspose2d parameters.")

            x = torch.cat((x, skip_connection), dim=1) # Concatenate along the channel dimension

            # Pass through residual and attention blocks
            x = res_block(x, time_emb, cond_emb)
            x = att_block(x)

        return self.final_conv(x)

# --- Diffusion Process ---
class Diffusion(nn.Module):
    def __init__(self, model, image_size, timesteps, beta_start, beta_end, beta_schedule):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps

        if beta_schedule == 'linear':
            betas = self.prepare_betas_linear(beta_start, beta_end)
        elif beta_schedule == 'cosine':
            betas = self.prepare_betas_cosine()
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.betas: torch.Tensor
        self.alphas: torch.Tensor
        self.alphas_cumprod: torch.Tensor
        self.sqrt_alphas_cumprod: torch.Tensor
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor
        self.posterior_variance: torch.Tensor

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # Calculations for diffusion sampling
        # posterior_variance の計算で F.pad を使用するために、beta_t と alpha_cumprod_t の次元を合わせる
        # self.alphas_cumprod の先頭に 1.0 を追加して、alphas_cumprod_prev を作成
        alphas_cumprod_prev = F.pad(self.alphas_cumprod, (1, 0), value=1.0)[:-1]
        posterior_variance = self.betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        posterior_variance[0] = self.betas[0] # 最初のステップの分散は beta_0 と定義される
        self.register_buffer('posterior_variance', posterior_variance)


    def prepare_betas_linear(self, beta_start, beta_end):
        return torch.linspace(beta_start, beta_end, self.timesteps, dtype=torch.float32)

    def prepare_betas_cosine(self, s=0.008):
        timesteps = self.timesteps
        t = torch.arange(timesteps + 1, dtype=torch.float32)
        f_t = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = f_t / f_t[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0, 0.999)

    def noise_images(self, x_0, t):
        noise = torch.randn_like(x_0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        return x_t, noise

    def forward(self, images, conditional_input):
        b = images.shape[0]
        t = torch.randint(0, self.timesteps, (b,), device=images.device).long()
        
        x_t, noise = self.noise_images(images, t)
        predicted_noise = self.model(x_t, t, conditional_input)
        
        loss = F.mse_loss(noise, predicted_noise)
        return loss

    @torch.no_grad()
    def sample(self, n_samples, conditional_input):
        self.model.eval()
        
        # x_t の初期化は、モデルの image_channels を使う
        x_t = torch.randn((n_samples, self.model.image_channels, self.image_size, self.image_size), device=self.betas.device)

        for i in reversed(range(1, self.timesteps)):
            t = torch.full((n_samples,), i, device=self.betas.device, dtype=torch.long)
            
            predicted_noise = self.model(x_t, t, conditional_input)

            alpha_t = self.alphas[t][:, None, None, None]
            alpha_cumprod_t = self.alphas_cumprod[t][:, None, None, None]
            beta_t = self.betas[t][:, None, None, None]
            sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]

            # DDPMの平均の計算
            # x_0_pred = (x_t - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_alphas_cumprod_t
            # mean = (beta_t * x_0_pred / sqrt_one_minus_alphas_cumprod_t) + (torch.sqrt(alpha_t) * x_t) / sqrt_one_minus_alphas_cumprod_t
            # 簡略化された平均の計算 (DDPM公式)
            mean = (x_t - beta_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t) / torch.sqrt(alpha_t)
            
            if i > 1:
                # variance をテンソルとして定義
                variance = self.posterior_variance[t][:, None, None, None]
                noise = torch.randn_like(x_t)
            else:
                # t=1 のステップではノイズを追加しない（または分散が非常に小さい）
                # variance をテンソルとして定義
                variance = torch.tensor(0.0, device=self.betas.device) # ここを修正
                noise = 0 # noise は 0 になるが、torch.sqrt(variance) * noise は 0 になるので問題ない
            
            # x_t の更新
            x_t = mean + torch.sqrt(variance) * noise
        
        self.model.train() # サンプリング後、モデルを訓練モードに戻す
        
        x_t = (x_t.clamp(-1, 1) + 1) / 2
        return x_t
