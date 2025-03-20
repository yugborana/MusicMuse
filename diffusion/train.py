import torch
import torch.optim as optim
from diffusion_model import DiffusionModel
from torch.utils.data import DataLoader, Dataset
import sys
import os
import librosa
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
save_dir = "checkpoints"
os.makedirs(save_dir, exist_ok=True)
save_path = os.path.join(save_dir, "diffusion_model.pth")
# Check if GPU is available
device = torch.device("cpu")
print(f"Training on: {device}")

# Example Dataset class for Mel spectrograms
class MelSpectrogramDataset(Dataset):
    def __init__(self, mel_spectrogram_dir):
        self.mel_spectrogram_dir = mel_spectrogram_dir
        self.files = [f for f in os.listdir(mel_spectrogram_dir) if f.endswith('.npy')]  # Assuming .npy files for Mel spectrograms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.mel_spectrogram_dir, self.files[idx])
        mel_spectrogram = np.load(file_path)
        return torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)  # Adding a channel dimension

# Setup the dataset and dataloaders
mel_spectrogram_dir = 'data/mel_spectrograms/'
dataset = MelSpectrogramDataset(mel_spectrogram_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Initialize the model and optimizer
model = DiffusionModel().to(device)  # Move model to GPU
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Loss function
def diffusion_loss(pred, target):
    return torch.mean((pred - target) ** 2)  # MSE loss

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for i, batch in enumerate(dataloader):
        # Move the batch to the GPU
        batch = batch.to(device)

        # Add noise to the mel spectrogram (forward diffusion)
        noisy_spectrograms = model.forward_diffusion(batch)

        # Reconstruct the clean spectrogram using the reverse diffusion process
        reconstructed_spectrograms = model.reverse_diffusion(noisy_spectrograms)

        # Calculate the loss
        loss = diffusion_loss(reconstructed_spectrograms, batch)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}")

    # Save model checkpoint
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint saved for epoch {epoch + 1}")    

