import sys
import os

# Print current working directory
print("Current working directory:", os.getcwd())

# Print Python path
print("Python path:", sys.path)

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Try importing the module again
try:
    from diffusion.diffusion_model import DiffusionModel
    print("Successfully imported DiffusionModel!")
except ModuleNotFoundError as e:
    print("Module not found:", e)
