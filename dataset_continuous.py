import torch
from torch.utils.data import Dataset
import numpy as np
import os


class BarLinkageDataset(Dataset):
    def __init__(self, data_dict, mean=None, std=None):
        self.images = data_dict["images"]
        self.attn_mask = data_dict["attn_mask"]  # (B, 8)

        # 1. Transform flat (B, 16) -> (B, 8, 3) with stop bits
        self.raw_inputs = self._process_continuous_data(
            data_dict["inputs"], self.attn_mask
        )
        self.raw_labels = self._process_continuous_data(
            data_dict["labels"], self.attn_mask
        )

        self.causal_mask = data_dict["causal_mask"]
        self.encoded_labels = data_dict["enc_labels"]
        self.vae_mu = data_dict.get("vae_mu")

        # Stats should only be for the 2D coordinates (x, y)
        # not needed for pre-normalized data
        # self.mean = mean if mean is not None else 0.0
        # self.std = std if std is not None else 1.0

    def _process_continuous_data(self, data, mask):
        """
        Converts (B, 16) -> (B, 8, 3)
        The 3rd dimension is the 'Stop Bit'
        """
        B = data.shape[0]
        # Reshape to (B, 8, 2)
        coords = data.reshape(B, 8, 2)

        # Create stop bits: 1.0 at the index of the last valid joint
        stop_bits = np.zeros((B, 8, 1), dtype=np.float32)

        # Find the last 'True' in each row of the attention mask
        # mask shape is (B, 8)
        last_joint_indices = np.sum(mask, axis=1) - 1
        for i, last_idx in enumerate(last_joint_indices):
            if last_idx >= 0:
                stop_bits[i, int(last_idx), 0] = 1.0

        # Concatenate to get (B, 8, 3)
        return np.concatenate([coords, stop_bits], axis=-1)

    @classmethod
    def from_folder(cls, data_dir, split_ratio=0.8, seed=42):
        # 1. Load raw files
        # Data is (N, 17). Slicing [:, 1:] removes SOS. [:, :-1] removes EOS.
        raw_inputs = np.load(f"{data_dir}/decoder_input_continuous.npy")[:, 1:]
        raw_labels = np.load(f"{data_dir}/labels_continuous.npy")[:, :-1]

        # Attention mask must be sliced to match the 8 joints (from 17 down to 8)
        # Assuming the original mask included SOS/EOS, we take the middle 8
        full_attn_mask = np.load(f"{data_dir}/attention_masks.npy")
        # If original was 17, indices 1-16 were the 8 pairs.
        # We need a boolean mask for the 8 joint positions.
        attn_mask_8 = full_attn_mask[
            :, 1:17:2
        ]  # Take every other to get 8 joint-level masks

        raw_data = {
            "images": np.load(f"{data_dir}/images.npy"),
            "inputs": raw_inputs.astype(np.float32),
            "labels": raw_labels.astype(np.float32),
            "attn_mask": attn_mask_8.astype(bool),
            "causal_mask": np.load(f"{data_dir}/causal_masks.npy"),
            "enc_labels": np.load(f"{data_dir}/encoded_labels.npy"),
        }

        vae_path = f"{data_dir}/vae_mu.npy"
        if os.path.exists(vae_path):
            raw_data["vae_mu"] = np.load(vae_path)

        # 2. Split
        num_samples = len(raw_data["images"])
        indices = np.arange(num_samples)
        np.random.seed(seed)
        np.random.shuffle(indices)
        split_idx = int(split_ratio * num_samples)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        train_set_raw = {k: v[train_idx] for k, v in raw_data.items()}
        val_set_raw = {k: v[val_idx] for k, v in raw_data.items()}

        # 3. Calculate Stats with Masking
        # mean, std = cls.calculate_stats(
        #     train_set_raw["inputs"], train_set_raw["labels"], train_set_raw["attn_mask"]
        # )

        return cls(train_set_raw), cls(val_set_raw)

    # @staticmethod
    # def calculate_stats(inputs, labels, attn_mask):
    #     """
    #     Calculate mean/std only for valid (non-padded) joints.
    #     """
    #     # Reshape to (B, 8, 2) to separate x and y
    #     in_coords = inputs.reshape(-1, 8, 2)
    #     lab_coords = labels.reshape(-1, 8, 2)
    #     pool = np.concatenate([in_coords, lab_coords], axis=0)  # (2*B, 8, 2)

    #     # Duplicate mask for the pool
    #     mask_pool = np.concatenate([attn_mask, attn_mask], axis=0)  # (2*B, 8)

    #     # Apply mask: keep only valid joints
    #     # valid_coords shape: (Total_Valid_Joints, 2)
    #     valid_coords = pool[mask_pool]

    #     mean = np.mean(valid_coords, axis=0)  # [mean_x, mean_y]
    #     std = np.std(valid_coords, axis=0)  # [std_x, std_y]

    #     return mean, std

    # def _normalize(self, x):
    #     """
    #     x: (8, 3) tensor
    #     We only normalize the first two columns (x, y).
    #     The 3rd column is the stop bit and stays 0 or 1.
    #     """
    #     x_norm = x.copy()
    #     x_norm[:, :2] = (x[:, :2] - self.mean) / (self.std + 1e-8)
    #     return x_norm

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Normalize only the coordinate part of the (8, 3) array
        # dec_in = self._normalize(self.raw_inputs[idx])
        # targets = self._normalize(self.raw_labels[idx])

        # skip normalization for pre-normalized data
        dec_in = self.raw_inputs[idx]
        targets = self.raw_labels[idx]

        sample = {
            "images": torch.tensor(self.images[idx], dtype=torch.float32),
            "decoder_input_continuous": torch.tensor(dec_in, dtype=torch.float32),
            "labels_continuous": torch.tensor(targets, dtype=torch.float32),
            "attn_mask": torch.tensor(self.attn_mask[idx], dtype=torch.bool),
            "encoded_labels": torch.tensor(self.encoded_labels[idx], dtype=torch.long),
        }

        if self.vae_mu is not None:
            sample["vae_mu"] = torch.tensor(self.vae_mu[idx], dtype=torch.float32)
        return sample
