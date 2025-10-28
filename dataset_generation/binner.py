import numpy as np
import torch
from torch.utils.data import Dataset
import os

class CoordinateBinner:
    def __init__(self, kappa=1.0, num_bins=200):
        """
        Discretize continuous coordinate space [-κ, κ] into uniform bins.
        """
        self.kappa = kappa
        self.num_bins = num_bins
        self.bin_edges = np.linspace(-kappa, kappa, num_bins + 1)
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        
    def value_to_bin(self, value):
        """
        Convert continuous coordinate value to bin index.
        """
        # Clip values to [-kappa, kappa] range
        clipped_value = np.clip(value, -self.kappa, self.kappa)
        
        # Apply the formula: b(v) = floor((v + κ) * (B - 1) / (2κ))
        bin_index = np.floor((clipped_value + self.kappa) * (self.num_bins - 1) / (2 * self.kappa))
        
        return bin_index.astype(int)
    
    def bin_to_value(self, bin_index):
        """
        Convert bin index back to continuous coordinate value (using bin center).
        """
        # Ensure bin index is within valid range
        bin_index = np.clip(bin_index, 0, self.num_bins - 1)
        
        # Return the center of the bin
        return self.bin_centers[bin_index]
    
    def value_to_bin_torch(self, value_tensor):
        """
        PyTorch version of value_to_bin.
        """
        # Clip values to [-kappa, kappa] range
        clipped_value = torch.clamp(value_tensor, -self.kappa, self.kappa)
        
        # Apply the formula
        bin_index = torch.floor((clipped_value + self.kappa) * (self.num_bins - 1) / (2 * self.kappa))
        
        return bin_index.long()
    
    def bin_to_value_torch(self, bin_index_tensor):
        """
        PyTorch version of bin_to_value.
        """
        # Ensure bin index is within valid range
        bin_index_tensor = torch.clamp(bin_index_tensor, 0, self.num_bins - 1)
        
        # Convert to numpy for indexing, then back to tensor
        bin_centers_tensor = torch.tensor(self.bin_centers, device=bin_index_tensor.device)
        return bin_centers_tensor[bin_index_tensor]


class DiscretizedTransformerDataset(Dataset):
    def __init__(self, data_dir, kappa=1.0, num_bins=200):
        """
        Dataset that loads and discretizes continuous coordinates.
        """
        self.images = np.load(os.path.join(data_dir, "images.npy"), mmap_mode='r')
        self.dec_in = np.load(os.path.join(data_dir, "decoder_input.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')
        self.masks = np.load(os.path.join(data_dir, "masks.npy"), mmap_mode='r')
        
        # Load text labels without memory mapping (since they contain Python objects)
        self.text_labels = np.load(os.path.join(data_dir, "text_labels.npy"), allow_pickle=True)
        
        self.binner = CoordinateBinner(kappa=kappa, num_bins=num_bins)
        self.num_bins = num_bins
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Get continuous values
        continuous_decoder = torch.tensor(self.dec_in[idx], dtype=torch.float32)
        continuous_label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        # Discretize coordinates (but keep special tokens as is)
        discretized_decoder = self._discretize_with_special_tokens(continuous_decoder)
        discretized_label = self._discretize_with_special_tokens(continuous_label)
        
        return {
            "image": torch.tensor(self.images[idx], dtype=torch.float32),
            "decoder_input": discretized_decoder,
            "label": discretized_label,
            "decoder_mask": torch.tensor(self.masks[idx], dtype=torch.bool),
            "text_label": self.text_labels[idx],
            # Also return continuous values for reference
            "continuous_decoder": continuous_decoder,
            "continuous_label": continuous_label,
        }
    
    def _discretize_with_special_tokens(self, tensor):
        """
        Discretize coordinates while preserving special tokens.
        Special tokens: SOS (-2), PAD (-1), EOS (2)
        """
        # Create mask for special tokens
        sos_mask = (tensor == -2.0).all(dim=-1)
        pad_mask = (tensor == -1.0).all(dim=-1)
        eos_mask = (tensor == 2.0).all(dim=-1)
        special_mask = sos_mask | pad_mask | eos_mask
        
        # Discretize non-special tokens
        discretized = torch.zeros_like(tensor, dtype=torch.long)
        
        # Handle x coordinates
        x_discrete = self.binner.value_to_bin_torch(tensor[:, 0])
        discretized[:, 0] = torch.where(special_mask, tensor[:, 0].long(), x_discrete)
        
        # Handle y coordinates
        y_discrete = self.binner.value_to_bin_torch(tensor[:, 1])
        discretized[:, 1] = torch.where(special_mask, tensor[:, 1].long(), y_discrete)
        
        return discretized


# Test the binning implementation
def test_binning(kappa=1.0, num_bins=200):
    """Test the binning implementation with example values."""
    binner = CoordinateBinner(kappa=kappa, num_bins=num_bins)
    
    print(f"Binning parameters: κ={kappa}, B={num_bins}")
    print(f"Bin range: [-{kappa}, {kappa}]")
    print(f"Bin edges: {binner.bin_edges[:5]} ... {binner.bin_edges[-5:]}")
    print(f"Bin centers: {binner.bin_centers[:5]} ... {binner.bin_centers[-5:]}")
    
    # Test some values
    test_values = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, -1.5]  # Last two should be clipped
    
    print("\nValue to bin conversion:")
    for val in test_values:
        bin_idx = binner.value_to_bin(val)
        recovered_val = binner.bin_to_value(bin_idx)
        print(f"  {val:6.2f} → bin {bin_idx:3d} → {recovered_val:6.3f}")
    
    # Test edge cases
    print("\nEdge cases:")
    edge_values = [-kappa, 0.0, kappa]
    for val in edge_values:
        bin_idx = binner.value_to_bin(val)
        recovered_val = binner.bin_to_value(bin_idx)
        print(f"  {val:6.3f} → bin {bin_idx:3d} → {recovered_val:6.3f}")


# Function to find optimal kappa from your data
def find_optimal_kappa(data_dir, margin=0.0):
    """Find the optimal kappa value from your dataset with some margin."""
    decoder_inputs = np.load(os.path.join(data_dir, "decoder_input.npy"))
    
    # Reshape and filter special tokens
    flat_decoder = decoder_inputs.reshape(-1, 2)
    valid_mask = ~((flat_decoder == -2.0).all(axis=1) | (flat_decoder == -1.0).all(axis=1))
    joint_coords = flat_decoder[valid_mask]
    
    # Find maximum absolute values
    max_abs_x = np.max(np.abs(joint_coords[:, 0]))
    max_abs_y = np.max(np.abs(joint_coords[:, 1]))
    
    # Use the larger of the two with margin
    optimal_kappa = max(max_abs_x, max_abs_y) * (1 + margin)
    
    print(f"Max absolute x: {max_abs_x:.6f}")
    print(f"Max absolute y: {max_abs_y:.6f}")
    print(f"Optimal kappa (with {margin*100:.0f}% margin): {optimal_kappa:.6f}")
    
    return optimal_kappa


# Usage example
if __name__ == "__main__":
    data_dir = "/home/anurizada/Documents/processed_dataset"
    
    # Find optimal kappa for your dataset
    kappa = find_optimal_kappa(data_dir, margin=0.0)
    num_bins = 200
    
    # Test the binning
    test_binning(kappa=kappa, num_bins=num_bins)
    
    # Create discretized dataset
    discretized_dataset = DiscretizedTransformerDataset(
        data_dir=data_dir, 
        kappa=kappa, 
        num_bins=num_bins
    )
    
    # Test a sample
    sample = discretized_dataset[0]
    print(f"\nSample 0:")
    print(f"Continuous decoder: {sample['continuous_decoder'][:5]}")
    print(f"Discretized decoder: {sample['decoder_input'][:5]}")
    print(f"Continuous label: {sample['continuous_label'][:5]}")
    print(f"Discretized label: {sample['label'][:5]}")