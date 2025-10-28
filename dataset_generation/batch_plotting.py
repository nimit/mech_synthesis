import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import BarLinkageDataset  # Your dataset

# Load dataset
dataset = BarLinkageDataset(data_dir='/home/anurizada/Documents/processed_dataset')
loader = DataLoader(dataset, batch_size=1024, shuffle=False)

all_tokens = []

for batch in loader:
    target_tokens = batch["labels_discrete"]  # shape [B, T]
    # Flatten and convert to list, ignore pad tokens
    tokens = target_tokens[target_tokens != 2].flatten().tolist()
    all_tokens.extend(tokens)

# Convert to tensor
all_tokens = torch.tensor(all_tokens)

# Plot histogram
plt.figure(figsize=(12, 6))
plt.hist(all_tokens.cpu().numpy(), bins=204, color='skyblue', edgecolor='black')
plt.title("Token Frequency Histogram")
plt.xlabel("Token Index")
plt.ylabel("Count")
plt.show()
