import torch
import torch.nn.functional as F
from unet import AudioUNet

device = torch.device("cpu")

class DiffusionModel(torch.nn.Module):
    def __init__(self, timesteps=1000):
        super(DiffusionModel, self).__init__()
        self.unet = AudioUNet(in_channels=1, out_channels=1)
        self.timesteps = timesteps
        self.beta_start = 1e-4
        self.beta_end = 0.02
        self.beta_schedule = torch.linspace(self.beta_start, self.beta_end, timesteps)

    def forward_diffusion(self, x_0):
        noise = torch.randn_like(x_0).to(device)
        beta_schedule = self.beta_schedule.view(-1, 1, 1, 1)
        x_t = x_0
        for t in range(self.timesteps):
            noise_scale = torch.sqrt(beta_schedule[t]).to(device)
            x_t = x_t + noise_scale * noise
        return x_t

    def reverse_diffusion(self, x_t):
        for t in range(self.timesteps-1, -1, -1):
            x_t = self.unet(x_t)
            noise_scale = torch.sqrt(self.beta_schedule[t])
            x_t = x_t - noise_scale
        return x_t
