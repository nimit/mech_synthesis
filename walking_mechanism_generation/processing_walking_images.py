import os
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from vae import VAE  # your existing VAE implementation

# ---------------- CONFIG ----------------

image_dir = "motiongen"          # directory with generated .jpg images
output_dir = "vae_latents"      # where to save .npy files
vae_checkpoint = "./weights/lat_50.ckpt"
latent_dim = 50
image_size = 64

os.makedirs(output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- LOAD VAE ----------------

input_dim = image_size * image_size
checkpoint = torch.load(vae_checkpoint, map_location=device)

vae = VAE(latent_dim, input_dim).to(device)
vae.load_state_dict(checkpoint["state_dict"])
vae.eval()

print("✅ Loaded VAE")

# ---------------- LOAD IMAGES ----------------

image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.lower().endswith((".jpg"))
])

if len(image_files) == 0:
    raise RuntimeError("No images found")

all_mu = []
all_sigma = []
all_filenames = []
print(image_files)
# ---------------- ENCODE ----------------

with torch.no_grad():
    for fname in tqdm(image_files, desc="Encoding images"):
        path = os.path.join(image_dir, fname)

        # Force grayscale + normalize
        img = Image.open(path)
        img = np.array(img, dtype=np.float32) / 255.0

        if img.shape != (image_size, image_size):
            raise ValueError(f"{fname} has shape {img.shape}, expected (64,64)")

        img_tensor = torch.from_numpy(img).view(1, 1, image_size, image_size).to(device)

        # Encode
        encoded = vae.encoder(img_tensor)
        mu = encoded[:, :latent_dim]
        logvar = encoded[:, latent_dim:]
        sigma = torch.exp(0.5 * logvar)

        all_mu.append(mu.cpu().numpy().squeeze(0))
        all_sigma.append(sigma.cpu().numpy().squeeze(0))
        all_filenames.append(fname)

# ---------------- SAVE ----------------

np.save(os.path.join(output_dir, "vae_mu.npy"), np.array(all_mu))
np.save(os.path.join(output_dir, "vae_sigma.npy"), np.array(all_sigma))
np.save(os.path.join(output_dir, "filenames.npy"), np.array(all_filenames, dtype=object))

print(f"\n✅ Saved latents to: {output_dir}")
print(f"μ shape: {np.array(all_mu).shape}")
print(f"σ shape: {np.array(all_sigma).shape}")
