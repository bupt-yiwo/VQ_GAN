import os
import numpy as np
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from model.transformer_vqgan import VQGANTransformer
from model.torch_utils import load_data


class TrainTransformer:
    def __init__(self, args):
        self.model = VQGANTransformer(args).to(device=args.device)
        self.optim = self.configure_optimizers()

        self.train(args)

    def configure_optimizers(self):
        decay, no_decay = set(), set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, nn.Embedding)

        for mn, m in self.model.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith("bias"):
                    no_decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)

                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        no_decay.add("pos_emb")

        param_dict = {pn: p for pn, p in self.model.transformer.named_parameters()}

        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=4.5e-06, betas=(0.9, 0.95))
        return optimizer

    def train(self, args):
        train_loader, val_loader = load_data(args)

        for epoch in range(args.epochs):
            print(f"\n[TRAIN] Starting Epoch {epoch}\n", flush=True)
            self.model.transformer.train()
            for param in self.model.vqgan.parameters():
                param.requires_grad = False
            os.makedirs(f"results/train/epoch_{epoch}", exist_ok=True)

            for i, (imgs, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                self.optim.zero_grad()
                imgs = imgs.to(device=args.device)
                logits, targets = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                loss.backward()
                self.optim.step()

                tqdm.write(f"[TRAIN] Iter {i} - Loss: {loss.item():.4f}")

                if i < 4:
                    log, sampled_imgs = self.model.log_images(imgs[:4])
                    vutils.save_image(
                        sampled_imgs,
                        os.path.join(f"results/train/epoch_{epoch}", f"train_sample_{i}.jpg"),
                        nrow=4
                    )

            self.validate(val_loader, epoch)
            torch.save(self.model.state_dict(), os.path.join("checkpoints", f"transformer_{epoch}.pt"))
            
    def validate(self, val_loader, epoch):
        os.makedirs(f"results/val/epoch_{epoch}", exist_ok=True)
        self.model.eval()  
        val_loss = 0
        with torch.no_grad():
            for i, (imgs, _) in enumerate(val_loader):
                imgs = imgs.to(device=args.device)
                logits, targets = self.model(imgs)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
                val_loss += loss.item()

                if i < 8:
                    log, sampled_imgs = self.model.log_images(imgs[0][None])
                    vutils.save_image(
                        sampled_imgs,
                        os.path.join(f"results/val/epoch_{epoch}", f"val_sample_{i}.jpg"),
                        nrow=4
                    )
        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch}] - Validation Loss: {avg_val_loss}")

        # 将验证损失写入txt文件
        with open('val_loss.txt', 'a') as f:
            f.write(f"Epoch {epoch} - Validation Loss: {avg_val_loss}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--latent-dim', type=int, default=256, help='Latent dimension n_z.')
    parser.add_argument('--image-size', type=int, default=256, help='Image height and width.)')
    parser.add_argument('--num-codebook-vectors', type=int, default=1024, help='Number of codebook vectors.')
    parser.add_argument('--beta', type=float, default=0.25, help='Commitment loss scalar.')
    parser.add_argument('--image-channels', type=int, default=3, help='Number of channels of images.')
    parser.add_argument('--dataset-path', type=str, default='./data', help='Path to data.')
    parser.add_argument('--checkpoint-path', type=str, default='./checkpoints/last_ckpt.pt', help='Path to checkpoint.')
    parser.add_argument('--device', type=str, default="cuda", help='Which device the training is on')
    parser.add_argument('--train-batch-size', type=int, default=20, help='Input batch size for training (default: 6)')
    parser.add_argument('--val-batch-size', type=int, default=20, help='Input batch size for validation (default: 6)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--learning-rate', type=float, default=2.25e-05, help='Learning rate.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam beta param.')
    parser.add_argument('--beta2', type=float, default=0.9, help='Adam beta param.')
    parser.add_argument('--disc-start', type=int, default=10000, help='When to start the discriminator.')
    parser.add_argument('--disc-factor', type=float, default=1., help='Weighting factor for the Discriminator.')
    parser.add_argument('--l2-loss-factor', type=float, default=1., help='Weighting factor for reconstruction loss.')
    parser.add_argument('--perceptual-loss-factor', type=float, default=1., help='Weighting factor for perceptual loss.')

    parser.add_argument('--pkeep', type=float, default=0.5, help='Percentage for how much latent codes to keep.')
    parser.add_argument('--sos-token', type=int, default=0, help='Start of Sentence token.')

    args = parser.parse_args()
    args.dataset_path = "/wangxiao/VQGAN/data/psi"
    args.checkpoint_path = "/wangxiao/VQGAN/checkpoints/vqgan_epoch_45.pt"

    train_transformer = TrainTransformer(args)
