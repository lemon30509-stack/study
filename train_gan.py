import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image

from config import GANConfig
from utils import ensure_dir, seed_everything


class Generator(nn.Module):
    def __init__(self, z_dim=100, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(z)
        return nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0),
        )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1).mean(dim=1)


def gradient_penalty(D, real, fake, device):
    b = real.size(0)
    alpha = torch.rand(b, 1, 1, 1, device=device)
    interpolates = alpha * real + (1 - alpha) * fake
    interpolates.requires_grad_(True)
    d_inter = D(interpolates)
    grads = torch.autograd.grad(
        outputs=d_inter,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gp = ((grads.view(b, -1).norm(2, dim=1) - 1) ** 2).mean()
    return gp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", type=str, default="outputs/gan")
    args = parser.parse_args()

    cfg = GANConfig(epochs=args.epochs, batch_size=args.batch_size)
    seed_everything(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = ensure_dir(args.output_dir)
    sample_dir = ensure_dir(out / "samples")

    tfm = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    ds = datasets.ImageFolder(args.data_dir, transform=tfm)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2, drop_last=True)

    G, D = Generator(cfg.z_dim, cfg.img_channels).to(device), Discriminator(cfg.img_channels).to(device)
    opt_g = optim.Adam(G.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))
    opt_d = optim.Adam(D.parameters(), lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))

    fixed_z = torch.randn(64, cfg.z_dim, 1, 1, device=device)

    for epoch in range(1, cfg.epochs + 1):
        for i, (real, _) in enumerate(loader):
            real = real.to(device)

            for _ in range(cfg.n_critic):
                z = torch.randn(real.size(0), cfg.z_dim, 1, 1, device=device)
                fake = G(z).detach()

                d_real, d_fake = D(real), D(fake)
                gp = gradient_penalty(D, real, fake, device)
                loss_d = -(d_real.mean() - d_fake.mean()) + cfg.lambda_gp * gp

                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()

            z = torch.randn(real.size(0), cfg.z_dim, 1, 1, device=device)
            fake = G(z)
            loss_g = -D(fake).mean()

            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()

        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                samples = G(fixed_z).cpu()
                grid = make_grid(samples, nrow=8, normalize=True)
                save_image(grid, sample_dir / f"epoch_{epoch:03d}.png")
            print(f"Epoch {epoch}/{cfg.epochs} | D: {loss_d.item():.4f} | G: {loss_g.item():.4f}")

    torch.save(G.state_dict(), out / "generator.pth")
    torch.save(D.state_dict(), out / "discriminator.pth")


if __name__ == "__main__":
    main()
