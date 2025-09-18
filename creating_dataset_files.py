import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import re
from PIL import Image
import os
from tqdm import tqdm

class ImageLinkageDataset(Dataset):
    def __init__(self, root_dir="bar_linkages", image_extensions=None, shuffle=True, max_joints=10, max_sequence_length=10):
        """
        Dataset for loading images with joint coordinates from filenames.
        """
        self.root_dir = Path(root_dir)
        self.image_extensions = image_extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
        self.max_joints = max_joints
        self.max_sequence_length = max_sequence_length
        self.image_paths = []
        self.metadata = []
        
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
                        joint_coords, text_label = self._parse_filename(file_path.name)
                        self.metadata.append((joint_coords, text_label))
    
    def _parse_filename(self, filename):
        """Parse filename to extract joint coordinates and text label."""
        name_without_ext = Path(filename).stem
        text_match = re.search(r'([A-Za-z]+[A-Za-z0-9-]*)', name_without_ext)
        
        if text_match:
            text_part = text_match.group(1)
            parts = name_without_ext.split(text_part)
            pre_text = parts[0].rstrip('_')
            numeric_values = []
            
            if pre_text:
                numeric_strs = pre_text.split('_')
                for num_str in numeric_strs:
                    if num_str:
                        try:
                            numeric_values.append(float(num_str))
                        except ValueError:
                            pass
            
            if len(numeric_values) % 2 != 0:
                numeric_values = numeric_values[:-1]
            
            joint_coords = np.array(numeric_values).reshape(-1, 2) if numeric_values else np.empty((0, 2))
            return joint_coords, text_part
        else:
            all_parts = name_without_ext.split('_')
            numeric_values = []
            for part in all_parts:
                if part:
                    try:
                        numeric_values.append(float(part))
                    except ValueError:
                        pass
            
            if len(numeric_values) % 2 != 0:
                numeric_values = numeric_values[:-1]
            
            joint_coords = np.array(numeric_values).reshape(-1, 2) if numeric_values else np.empty((0, 2))
            return joint_coords, ""
    
    def _shuffle_data(self):
        """Shuffle the dataset"""
        indices = np.random.permutation(len(self.image_paths))
        self.image_paths = [self.image_paths[i] for i in indices]
        self.metadata = [self.metadata[i] for i in indices]
    
    def _create_decoder_data(self, pos):
        """Create decoder input and label with proper padding."""
        num_nodes = pos.size(0)
        pad_len = self.max_sequence_length - num_nodes - 1
        
        decoder_input = torch.cat([
            torch.ones(1, pos.size(1)) * -2.0,  # SOS
            pos,
            torch.full((pad_len, pos.size(1)), -1.0)  # PAD
        ], dim=0)
        
        label = torch.cat([
            pos,
            torch.ones(1, pos.size(1)) * 2.0,   # EOS
            torch.full((pad_len, pos.size(1)), -1.0)  # PAD
        ], dim=0)
        
        return decoder_input, label
    
    def _create_combined_mask(self, decoder_input):
        """Create combined mask for decoder."""
        size = decoder_input.size(0)
        causal_mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        pad_mask = ~(decoder_input == -1.0).all(dim=1)
        pad_mask = pad_mask.unsqueeze(0).expand(size, -1)
        combined_mask = causal_mask | ~pad_mask | ~pad_mask.T
        return ~combined_mask
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        joint_coords, text_label = self.metadata[idx]
        
        joint_coords = joint_coords / 10
        
        try:
            image = Image.open(image_path)
            image_tensor = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
            if len(image_tensor.shape) == 2:
                image_tensor = image_tensor.unsqueeze(-1)
            image_tensor = image_tensor.permute(2, 0, 1)
        except Exception as e:
            image_tensor = torch.zeros((1, 64, 64), dtype=torch.float32)
        
        n_joints = min(len(joint_coords), self.max_joints)
        joint_tensor = torch.full((self.max_joints, 2), 0.0, dtype=torch.float32)
        
        if n_joints > 0:
            joint_tensor[:n_joints] = torch.tensor(joint_coords[:n_joints], dtype=torch.float32)
        
        actual_joints = joint_tensor[:n_joints]
        decoder_input, label = self._create_decoder_data(actual_joints)
        decoder_mask = self._create_combined_mask(decoder_input)
        
        return {
            "image": image_tensor,
            "text_label": text_label,
            "decoder_input": decoder_input,
            "label": label,
            "decoder_mask": decoder_mask,
        }


def create_npy_files(dataset, output_dir):
    """
    Process the dataset and save individual components as .npy files.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize lists to store all data
    all_images = []
    all_text_labels = []
    all_decoder_inputs = []
    all_labels = []
    all_masks = []
    
    print(f"Processing {len(dataset)} samples...")
    
    # Process each sample
    for i in tqdm(range(len(dataset))):
        sample = dataset[i]
        
        all_images.append(sample["image"].numpy())
        all_text_labels.append(sample["text_label"])
        all_decoder_inputs.append(sample["decoder_input"].numpy())
        all_labels.append(sample["label"].numpy())
        all_masks.append(sample["decoder_mask"].numpy())
    
    # Convert to numpy arrays
    all_images = np.array(all_images)
    all_decoder_inputs = np.array(all_decoder_inputs)
    all_labels = np.array(all_labels)
    all_masks = np.array(all_masks)
    
    # Save as .npy files
    print("Saving .npy files...")
    np.save(os.path.join(output_dir, "images.npy"), all_images)
    np.save(os.path.join(output_dir, "decoder_input.npy"), all_decoder_inputs)
    np.save(os.path.join(output_dir, "labels.npy"), all_labels)
    np.save(os.path.join(output_dir, "masks.npy"), all_masks)
    np.save(os.path.join(output_dir, "text_labels.npy"), np.array(all_text_labels, dtype=object))
    
    print(f"Files saved to {output_dir}:")
    print(f"  - images.npy: {all_images.shape}")
    print(f"  - decoder_input.npy: {all_decoder_inputs.shape}")
    print(f"  - labels.npy: {all_labels.shape}")
    print(f"  - masks.npy: {all_masks.shape}")
    print(f"  - text_labels.npy: {len(all_text_labels)} mechanism types")


class SingleTransformerDataset(Dataset):
    def __init__(self, data_dir):
        """
        Dataset that loads individual .npy files for efficient memory mapping.
        """
        self.images = np.load(os.path.join(data_dir, "images.npy"), mmap_mode='r')
        self.dec_in = np.load(os.path.join(data_dir, "decoder_input.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_dir, "labels.npy"), mmap_mode='r')
        self.masks = np.load(os.path.join(data_dir, "masks.npy"), mmap_mode='r')
        self.text_labels = np.load(os.path.join(data_dir, "text_labels.npy"), allow_pickle=True)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {
            "image": torch.tensor(self.images[idx], dtype=torch.float32),
            "decoder_input": torch.tensor(self.dec_in[idx], dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
            "decoder_mask": torch.tensor(self.masks[idx], dtype=torch.bool),
            "text_label": self.text_labels[idx],
        }


def test_saved_dataset(data_dir):
    """Test the saved dataset by loading a few samples"""
    dataset = SingleTransformerDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"\nTesting saved dataset with {len(dataset)} samples:")
    print("=" * 80)
    
    for i, batch in enumerate(dataloader):
        if i >= 3:  # Show first 3 samples
            break
        
        print(f"\nSample {i+1}:")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Decoder input shape: {batch['decoder_input'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"Mask shape: {batch['decoder_mask'].shape}")
        print(f"Text label: {batch['text_label'][0]}")
        
        # Remove batch dimension for easier viewing
        decoder_input = batch['decoder_input'][0]  # [seq_len, 2]
        label = batch['label'][0]  # [seq_len, 2]
        mask = batch['decoder_mask'][0]  # [seq_len, seq_len]
        
        print(f"\nComplete Decoder Input ({decoder_input.shape}):")
        for j in range(decoder_input.shape[0]):
            print(f"  Position {j:2d}: {decoder_input[j].numpy()}")
        
        print(f"\nComplete Label ({label.shape}):")
        for j in range(label.shape[0]):
            print(f"  Position {j:2d}: {label[j].numpy()}")
        
        print(f"\nComplete Mask ({mask.shape}):")
        print("     " + " ".join([f"{k:2d}" for k in range(mask.shape[1])]))
        for j in range(mask.shape[0]):
            row_str = " ".join([" T " if val else " F " for val in mask[j].numpy()])
            print(f"  {j:2d}: {row_str}")
        
        # Show special token analysis
        print(f"\nSpecial Token Analysis:")
        sos_positions = torch.where((decoder_input == -2.0).all(dim=1))[0]
        eos_positions = torch.where((label == 2.0).all(dim=1))[0]
        pad_positions_dec = torch.where((decoder_input == -1.0).all(dim=1))[0]
        pad_positions_lab = torch.where((label == -1.0).all(dim=1))[0]
        
        print(f"  SOS positions in decoder input: {sos_positions.tolist()}")
        print(f"  EOS positions in labels: {eos_positions.tolist()}")
        print(f"  PAD positions in decoder input: {pad_positions_dec.tolist()}")
        print(f"  PAD positions in labels: {pad_positions_lab.tolist()}")
        
        # Count actual joints (non-pad, non-special tokens in labels before EOS)
        actual_joints = 0
        for j in range(label.shape[0]):
            if (label[j] == 2.0).all():  # EOS found
                break
            if not (label[j] == -1.0).all():  # Not PAD
                actual_joints += 1
        
        print(f"  Actual number of joints: {actual_joints}")
        
        print("=" * 80)

if __name__ == "__main__":
    # Create and process the dataset
    dataset = ImageLinkageDataset(root_dir="/home/anurizada/Documents/bar_linkages", max_joints=12, max_sequence_length=12)
    
    # Specify output directory
    output_dir = "/home/anurizada/Documents/processed_dataset"
    
    # Create .npy files
    create_npy_files(dataset, output_dir)
    
    # Test the saved dataset
    test_saved_dataset(output_dir)