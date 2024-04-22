import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def preprocess_data(log_ret_df):
    # Assuming log_ret_df is already preprocessed
    return torch.tensor(log_ret_df.values, dtype=torch.float32)

# Model Architecture
class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

class GAN:
    def __init__(self, latent_dim, output_dim):
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = Generator(latent_dim, output_dim).to(self.device)
        self.discriminator = Discriminator(output_dim).to(self.device)
        self.criterion = nn.BCELoss()
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    def fit(self, data_loader, epochs, batch_size):
        for epoch in range(epochs):
            for i, data in enumerate(data_loader, 0):
                real_batch_size = data[0].size(0)  # Get the actual batch size of real data

                # Update Discriminator
                self.discriminator.zero_grad()
                real_data = data[0].to(self.device)
                real_target = torch.full((real_batch_size, 1), 1, dtype=torch.float32, device=self.device)
                output = self.discriminator(real_data)
                d_loss_real = self.criterion(output, real_target)

                noise = torch.randn(real_batch_size, self.latent_dim, device=self.device)
                fake_data = self.generator(noise)
                fake_target = torch.full((real_batch_size, 1), 0, dtype=torch.float32, device=self.device)
                output = self.discriminator(fake_data.detach())
                d_loss_fake = self.criterion(output, fake_target)

                d_loss = d_loss_real + d_loss_fake
                d_loss.backward()
                self.d_optimizer.step()

                # Update Generator
                self.generator.zero_grad()
                output = self.discriminator(fake_data)
                g_loss = self.criterion(output, real_target)  # Generator tries to fool the discriminator
                g_loss.backward()
                self.g_optimizer.step()

            # Print progress
            if epoch % 100 == 0 or epoch < 3:
                print(f"Epoch [{epoch}/{epochs}], Batch [{i}/{len(data_loader)}], "
                        f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    def generate_samples(self, n_samples):
        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim, device=self.device)
            generated_samples = self.generator(noise)
            generated_samples = generated_samples.cpu().numpy()
        self.generator.train()
        return generated_samples