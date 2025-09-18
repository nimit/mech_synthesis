# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from dataset import SingleTransformerDataset  # adjust if in same file

# dataset = SingleTransformerDataset(
#     node_features_path='/home/anurizada/Documents/nobari_10_joints/node_features.npy',
#     edge_index_path='/home/anurizada/Documents/nobari_10_joints/edge_index.npy',
#     curves_path='/home/anurizada/Documents/nobari_10_joints/curves.npy'
# )

# # Output directory
# output_dir = "/home/anurizada/Documents/nobari_10_transformer"
# os.makedirs(output_dir, exist_ok=True)

# # Create empty lists to collect preprocessed samples
# all_adj = []
# all_curves = []
# all_dec_in = []
# all_labels = []
# all_masks = []

# print("🚀 Preprocessing full dataset...")
# for i in tqdm(range(len(dataset))):
#     sample = dataset[i]
#     all_adj.append(sample["adjacency"].numpy())
#     all_curves.append(sample["curve_numerical"].numpy())
#     all_dec_in.append(sample["decoder_input"].numpy())
#     all_labels.append(sample["label"].numpy())
#     all_masks.append(sample["decoder_mask"].numpy())

# # Save each as a separate file
# np.save(os.path.join(output_dir, "adjacency.npy"), np.stack(all_adj))
# np.save(os.path.join(output_dir, "curves.npy"), np.stack(all_curves))
# np.save(os.path.join(output_dir, "decoder_input.npy"), np.stack(all_dec_in))
# np.save(os.path.join(output_dir, "labels.npy"), np.stack(all_labels))
# np.save(os.path.join(output_dir, "masks.npy"), np.stack(all_masks))

# print("✅ Done. All preprocessed tensors saved.")

# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# from dataset import TransformerDataset  # adjust if in same file

# dataset = TransformerDataset(
#     node_features_path='/home/anurizada/Documents/nobari_20_joints/node_features.npy',
#     edge_index_path='/home/anurizada/Documents/nobari_20_joints/edge_index.npy',
#     curves_path='/home/anurizada/Documents/nobari_20_joints/curves.npy'
# )

# # Output directory
# output_dir = "/home/anurizada/Documents/nobari_20_transformer"
# os.makedirs(output_dir, exist_ok=True)

# # Create empty lists to collect preprocessed samples
# all_adj = []
# all_curves = []
# all_dec_in_first, all_dec_in_second = [], []
# all_lbl_first, all_lbl_second = [], []
# all_mask_first, all_mask_second = [], []

# print("🚀 Preprocessing full dataset...")
# for i in tqdm(range(len(dataset))):
#     sample = dataset[i]
#     all_adj.append(sample["adjacency"].numpy())
#     all_curves.append(sample["curve_numerical"].numpy())
#     all_dec_in_first.append(sample["decoder_input_first"].numpy())
#     all_dec_in_second.append(sample["decoder_input_second"].numpy())
#     all_lbl_first.append(sample["label_first"].numpy())
#     all_lbl_second.append(sample["label_second"].numpy())
#     all_mask_first.append(sample["decoder_mask_first"].numpy())
#     all_mask_second.append(sample["decoder_mask_second"].numpy())

# # Save each as a separate file
# np.save(os.path.join(output_dir, "adjacency.npy"), np.stack(all_adj))
# np.save(os.path.join(output_dir, "curves.npy"), np.stack(all_curves))
# np.save(os.path.join(output_dir, "decoder_input_first.npy"), np.stack(all_dec_in_first))
# np.save(os.path.join(output_dir, "decoder_input_second.npy"), np.stack(all_dec_in_second))
# np.save(os.path.join(output_dir, "label_first.npy"), np.stack(all_lbl_first))
# np.save(os.path.join(output_dir, "label_second.npy"), np.stack(all_lbl_second))
# np.save(os.path.join(output_dir, "mask_first.npy"), np.stack(all_mask_first))
# np.save(os.path.join(output_dir, "mask_second.npy"), np.stack(all_mask_second))

# print("✅ Done. All preprocessed tensors saved.")

from torch.utils.data import DataLoader
from dataset import SingleTransformerDataset

# ------------------------------
# Load Dataset
# ------------------------------
dataset = SingleTransformerDataset(
    data_dir='/home/anurizada/Documents/nobari_10_transformer'
)

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# ------------------------------
# Inspect the first 5 samples
# ------------------------------
print("=== Dataset Inspection ===\n")
for i, sample in enumerate(dataloader):
    print(f"--- Sample {i} ---")
    for key, value in sample.items():
        if key == 'curve_numerical':
            continue
        print(f"{key}:")
        print(f"  Shape: {tuple(value.shape)}")
        print(f"  Dtype: {value.dtype}")
        # Print only first 5 values for brevity
        flat_val = value
        print(f"  Sample values (first 5 entries): {flat_val}")
    print("\n")
    
    if i == 4:  # Stop after 5 samples
        break

# import numpy as np

# # Load the .npy file
# data = np.load('adjacency.npy', mmap_mode='r')

# # Check current dtype and memory usage
# print("Original dtype:", data.dtype)
# print("Original itemsize (bytes per element):", data.itemsize)
# print("Original shape:", data.shape)
# print("Original memory usage (MB):", data.nbytes / (1024 * 1024))

# # Convert to smallest appropriate integer type
# if np.all(data == 0) or np.all(data == 1):  # If strictly binary values
#     converted = data.astype(np.uint8)  # Smallest standard integer type (1 byte)
#     # Alternative: np.bool_ if you only need True/False (1 bit per element)
# else:
#     converted = data.astype(np.int8)  # If you might have other small integers

# # Verify conversion
# print("\nConverted dtype:", converted.dtype)
# print("Converted itemsize:", converted.itemsize)
# print("Converted memory usage (MB):", converted.nbytes / (1024 * 1024))

# # Save the converted file
# np.save('converted_file.npy', converted)
