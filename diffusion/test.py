import torch
from diffusion_model import DiffusionModel

device = torch.device("cpu")
model = DiffusionModel().to(device)

checkpoint = torch.load("checkpoints/diffusion_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("hehe")