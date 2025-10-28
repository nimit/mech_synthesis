import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
from PIL import Image
import os

class ImageLinkageDataset(Dataset):
    def __init__(self, root_dir="/home/anurizada/Documents/bar_linkages", image_extensions=None, shuffle=True, max_joints=10, max_sequence_length=10):
        """
        Dataset for loading images with joint coordinates from filenames.
        
        Args:
            root_dir (str): Root directory containing subfolders with images
            image_extensions (list): List of valid image extensions
            shuffle (bool): Whether to shuffle the data
            max_joints (int): Maximum number of joints to consider
            max_sequence_length (int): Maximum sequence length for decoder (including <s> and </s> tokens)
        """
        self.root_dir = Path(root_dir)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        self.max_joints = max_joints
        self.max_sequence_length = max_sequence_length
        self.image_paths = []
        self.metadata = []  # Will store (joint_coords, text_label) tuples
        
        # Collect all image paths and parse metadata
        self._collect_image_paths()
        
        if shuffle:
            self._shuffle_data()
    
    def _collect_image_paths(self):
        """Collect all image paths and parse their metadata"""
        if not self.root_dir.exists():
            raise ValueError(f"Directory '{self.root_dir}' does not exist!")
        
        for folder in self.root_dir.iterdir():
            if folder.is_dir():
                for file_path in folder.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in self.image_extensions:
                        self.image_paths.append(file_path)
                        # Parse metadata from filename
                        joint_coords, text_label = self._parse_filename(file_path.name)
                        self.metadata.append((joint_coords, text_label))
    
    def _parse_filename(self, filename):
        """
        Parse filename to extract joint coordinates and text label.
        Format: x0_y0_x1_y1_..._xn_yn_textlabel_..._.ext
        """
        # Remove file extension
        name_without_ext = Path(filename).stem
        
        # Use regex to find the text part (typically contains letters and numbers with hyphens)
        text_match = re.search(r'([A-Za-z]+[A-Za-z0-9-]*)', name_without_ext)
        
        if text_match:
            text_part = text_match.group(1)
            # Split the filename around the text part
            parts = name_without_ext.split(text_part)
            
            # Extract numeric values from the part before text
            pre_text = parts[0].rstrip('_')
            numeric_values = []
            
            if pre_text:
                # Split by underscore and convert to float
                numeric_strs = pre_text.split('_')
                for num_str in numeric_strs:
                    if num_str:  # Skip empty strings
                        try:
                            numeric_values.append(float(num_str))
                        except ValueError:
                            print(f"Warning: Could not convert '{num_str}' to float in filename: {filename}")
            
            # Reshape into (x, y) pairs
            if len(numeric_values) % 2 != 0:
                print(f"Warning: Odd number of numeric values ({len(numeric_values)}) in filename: {filename}")
                # Remove last value to make it even
                numeric_values = numeric_values[:-1]
            
            # Reshape to (n_joints, 2)
            joint_coords = np.array(numeric_values).reshape(-1, 2) if numeric_values else np.empty((0, 2))
            
            return joint_coords, text_part
        else:
            # If no text part found, try to extract all numeric values
            print(f"Warning: No text pattern found in filename: {filename}")
            all_parts = name_without_ext.split('_')
            numeric_values = []
            for part in all_parts:
                if part:
                    try:
                        numeric_values.append(float(part))
                    except ValueError:
                        # This part is not numeric
                        pass
            
            # Reshape into (x, y) pairs
            if len(numeric_values) % 2 != 0:
                print(f"Warning: Odd number of numeric values ({len(numeric_values)}) in filename: {filename}")
                numeric_values = numeric_values[:-1]
            
            joint_coords = np.array(numeric_values).reshape(-1, 2) if numeric_values else np.empty((0, 2))
            
            return joint_coords, ""
    
    def _shuffle_data(self):
        """Shuffle the dataset"""
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]
    
    def _create_decoder_data(self, pos):
        """
        Create decoder input and label with proper padding.
        
        Args:
            pos: Tensor of shape [num_nodes, 2] containing joint coordinates
            
        Returns:
            decoder_input: Tensor of shape [max_sequence_length, 2]
            label: Tensor of shape [max_sequence_length, 2]
        """
        num_nodes = pos.size(0)
        pad_len = self.max_sequence_length - num_nodes - 1
        
        # Create decoder input: <s> + joints + padding
        decoder_input = torch.cat([
            torch.ones(1, pos.size(1)) * -2.0,                  # <s> token
            pos,
            torch.full((pad_len, pos.size(1)), -1.0)            # Padding
        ], dim=0)
        
        # Create label: joints + </s> + padding
        label = torch.cat([
            pos,
            torch.ones(1, pos.size(1)) * 2.0,                   # </s> token
            torch.full((pad_len, pos.size(1)), -1.0)            # Padding
        ], dim=0)
        
        return decoder_input, label
    
    def _create_combined_mask(self, decoder_input):
        """
        Create combined mask for decoder.
        
        Args:
            decoder_input: Tensor of shape [max_sequence_length, 2]
            
        Returns:
            mask: Boolean tensor of shape [max_sequence_length, max_sequence_length]
        """
        size = decoder_input.size(0)
        # Create causal mask (upper triangular)
        causal_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        # Create padding mask (False for padding tokens)
        pad_mask = ~(decoder_input == -1.0).all(dim=1)
        pad_mask = pad_mask.unsqueeze(0).expand(size, -1)
        # Combine masks
        combined_mask = causal_mask | ~pad_mask | ~pad_mask.T
        return ~combined_mask  # Invert so True means "attend"
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Get image path and metadata
        image_path = self.image_paths[idx]
        joint_coords, text_label = self.metadata[idx]
        joint_coords = joint_coords / 10
        # Load image
        try:
            image = Image.open(image_path)
            # Convert to tensor and normalize
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
            # If grayscale, add channel dimension
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(-1)
            # Convert to CHW format
            image_tensor = image_tensor.permute(2, 0, 1)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image if loading fails
            image_tensor = torch.zeros((3, 64, 64), dtype=torch.float32)
        
        # Convert joint coordinates to tensor and handle padding
        n_joints = min(len(joint_coords), self.max_joints)
        joint_tensor = torch.full((self.max_joints, 2), 0.0, dtype=torch.float32)
        
        if n_joints > 0:
            joint_tensor[:n_joints] = torch.tensor(joint_coords[:n_joints], dtype=torch.float32)
        
        # Create decoder components using the actual joint coordinates
        # Use only the actual joints (not padded ones) for decoder sequence
        actual_joints = joint_tensor[:n_joints]
        decoder_input, label = self._create_decoder_data(actual_joints)
        decoder_mask = self._create_combined_mask(decoder_input)
        
        return {
            "image": image_tensor,
            "joint_coordinates": joint_tensor,  # Shape: [max_joints, 2]
            "num_joints": torch.tensor(n_joints, dtype=torch.long),  # Actual number of joints
            "text_label": text_label,
            "filename": str(image_path.name),
            "decoder_input": decoder_input,      # Shape: [max_sequence_length, 2]
            "label": label,                      # Shape: [max_sequence_length, 2]
            "decoder_mask": decoder_mask,        # Shape: [max_sequence_length, max_sequence_length]
        }


# Create dataset and dataloader
def test_dataloader():
    # Create dataset
    dataset = ImageLinkageDataset(root_dir="/home/anurizada/Documents/bar_linkages", max_joints=10, max_sequence_length=10)
    
    # Create dataloader with batch size 1
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    
    print(f"Total images in dataset: {len(dataset)}")
    print(f"Max sequence length: {dataset.max_sequence_length}")
    print("=" * 80)
    
    # Print first 10 entries
    for i, batch in enumerate(dataloader):
        if i >= 10:  # Only show first 10
            break
        
        print(f"\n--- Entry {i+1} ---")
        print(f"Filename: {batch['filename'][0]}")
        print(f"Text Label: {batch['text_label'][0]}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Joint coordinates shape: {batch['joint_coordinates'].shape}")
        print(f"Number of joints: {batch['num_joints'].item()}")
        print(f"Joint values (first 3): {batch['joint_coordinates'][0]}")
        print(f"Decoder input shape: {batch['decoder_input'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Mask shape: {batch['decoder_mask'].shape}")
        
        # Show special tokens in decoder input and label
        print(f"Decoder input special tokens: {batch['decoder_input'][0, :]}")
        print(f"Label special tokens: {batch['label'][0, :]}")
        print(f"Mask: {batch['decoder_mask']}")


if __name__ == "__main__":
    test_dataloader()
