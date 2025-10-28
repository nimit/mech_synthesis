import numpy as np
import os

def find_max_joint_values_with_examples(data_dir):
    """
    Find the maximum absolute values for x and y coordinates from decoder inputs,
    and show the specific arrays that contain these maximum values.
    """
    decoder_inputs = np.load(os.path.join(data_dir, "decoder_input.npy"))
    print(f"Decoder inputs shape: {decoder_inputs.shape}")
    
    # Reshape to 2D array for easier processing
    flat_decoder = decoder_inputs.reshape(-1, 2)
    
    # Create mask for non-special tokens
    not_sos = ~(flat_decoder == -2.0).all(axis=1)
    not_pad = ~(flat_decoder == -1.0).all(axis=1)
    valid_mask = not_sos & not_pad
    
    joint_coords = flat_decoder[valid_mask]
    
    print(f"Total joint coordinates found: {len(joint_coords):,}")
    
    if len(joint_coords) == 0:
        return {"max_abs_x": 0.0, "max_abs_y": 0.0}
    
    # Find maximum absolute values and their positions
    abs_x = np.abs(joint_coords[:, 0])
    abs_y = np.abs(joint_coords[:, 1])
    
    max_abs_x = np.max(abs_x)
    max_abs_y = np.max(abs_y)
    
    # Find the actual coordinates that have these maximum absolute values
    max_x_coords = joint_coords[abs_x == max_abs_x]
    max_y_coords = joint_coords[abs_y == max_abs_y]
    
    # Now we need to find which original arrays these came from
    # Get the indices in the flattened array where max values occur
    max_x_indices_flat = np.where(abs_x == max_abs_x)[0]
    max_y_indices_flat = np.where(abs_y == max_abs_y)[0]
    
    # Convert flat indices back to original array indices
    valid_indices_flat = np.where(valid_mask)[0]  # Indices in flat_decoder that are valid
    max_x_orig_indices = valid_indices_flat[max_x_indices_flat]
    max_y_orig_indices = valid_indices_flat[max_y_indices_flat]
    
    # Convert flat indices to original (sample, position) indices
    max_x_sample_indices = max_x_orig_indices // decoder_inputs.shape[1]
    max_x_pos_indices = max_x_orig_indices % decoder_inputs.shape[1]
    
    max_y_sample_indices = max_y_orig_indices // decoder_inputs.shape[1]
    max_y_pos_indices = max_y_orig_indices % decoder_inputs.shape[1]
    
    # Get the complete arrays that contain these maximum values
    max_x_arrays = []
    for sample_idx, pos_idx in zip(max_x_sample_indices, max_x_pos_indices):
        max_x_arrays.append({
            'sample_index': sample_idx,
            'position_index': pos_idx,
            'complete_array': decoder_inputs[sample_idx],
            'max_value': decoder_inputs[sample_idx, pos_idx]
        })
    
    max_y_arrays = []
    for sample_idx, pos_idx in zip(max_y_sample_indices, max_y_pos_indices):
        max_y_arrays.append({
            'sample_index': sample_idx,
            'position_index': pos_idx,
            'complete_array': decoder_inputs[sample_idx],
            'max_value': decoder_inputs[sample_idx, pos_idx]
        })
    
    # Calculate statistics
    stats = {
        "max_abs_x": float(max_abs_x),
        "max_abs_y": float(max_abs_y),
        "mean_x": float(np.mean(joint_coords[:, 0])),
        "mean_y": float(np.mean(joint_coords[:, 1])),
        "std_x": float(np.std(joint_coords[:, 0])),
        "std_y": float(np.std(joint_coords[:, 1])),
        "min_x": float(np.min(joint_coords[:, 0])),
        "max_x": float(np.max(joint_coords[:, 0])),
        "min_y": float(np.min(joint_coords[:, 1])),
        "max_y": float(np.max(joint_coords[:, 1])),
        "total_joint_coords": len(joint_coords),
        "max_x_examples": max_x_arrays,
        "max_y_examples": max_y_arrays,
    }
    
    return stats


def print_detailed_statistics_with_examples(stats):
    """Print detailed statistics including the arrays with maximum values."""
    print("\n" + "="*80)
    print("JOINT COORDINATE STATISTICS WITH EXAMPLES")
    print("="*80)
    
    print(f"Total joint coordinates analyzed: {stats['total_joint_coords']:,}")
    print(f"Maximum absolute x value: {stats['max_abs_x']:.6f}")
    print(f"Maximum absolute y value: {stats['max_abs_y']:.6f}")
    print(f"X coordinate range: [{stats['min_x']:.6f}, {stats['max_x']:.6f}]")
    print(f"Y coordinate range: [{stats['min_y']:.6f}, {stats['max_y']:.6f}]")
    
    # Print examples with maximum x values
    print(f"\nArrays containing maximum x value ({stats['max_abs_x']:.6f}):")
    for i, example in enumerate(stats['max_x_examples'][:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"  Sample index: {example['sample_index']}")
        print(f"  Position index: {example['position_index']}")
        print(f"  Max value at position: {example['max_value']}")
        print(f"  Complete array:")
        for j, pos in enumerate(example['complete_array']):
            marker = " ← MAX" if j == example['position_index'] else ""
            print(f"    Position {j:2d}: {pos}{marker}")
    
    # Print examples with maximum y values
    print(f"\nArrays containing maximum y value ({stats['max_abs_y']:.6f}):")
    for i, example in enumerate(stats['max_y_examples'][:3]):  # Show first 3 examples
        print(f"\nExample {i+1}:")
        print(f"  Sample index: {example['sample_index']}")
        print(f"  Position index: {example['position_index']}")
        print(f"  Max value at position: {example['max_value']}")
        print(f"  Complete array:")
        for j, pos in enumerate(example['complete_array']):
            marker = " ← MAX" if j == example['position_index'] else ""
            print(f"    Position {j:2d}: {pos}{marker}")
    
    # Recommended binning ranges
    x_margin = stats['max_abs_x'] * 0.1
    y_margin = stats['max_abs_y'] * 0.1
    
    recommended_x_range = [-stats['max_abs_x'] - x_margin, stats['max_abs_x'] + x_margin]
    recommended_y_range = [-stats['max_abs_y'] - y_margin, stats['max_abs_y'] + y_margin]
    
    print(f"\nRecommended binning ranges:")
    print(f"  X: [{recommended_x_range[0]:.6f}, {recommended_x_range[1]:.6f}]")
    print(f"  Y: [{recommended_y_range[0]:.6f}, {recommended_y_range[1]:.6f}]")
    
    print("="*80)


# Quick function to just get the max values with one example
def get_max_joint_values_with_example(data_dir):
    """Get max values with one example array for each."""
    decoder_inputs = np.load(os.path.join(data_dir, "decoder_input.npy"))
    
    # Reshape and filter
    flat_decoder = decoder_inputs.reshape(-1, 2)
    valid_mask = ~((flat_decoder == -2.0).all(axis=1) | (flat_decoder == -1.0).all(axis=1))
    joint_coords = flat_decoder[valid_mask]
    
    if len(joint_coords) == 0:
        return 0.0, 0.0, None, None
    
    abs_x = np.abs(joint_coords[:, 0])
    abs_y = np.abs(joint_coords[:, 1])
    
    max_abs_x = np.max(abs_x)
    max_abs_y = np.max(abs_y)
    
    # Get one example for each
    max_x_idx_flat = np.where(abs_x == max_abs_x)[0][0]
    max_y_idx_flat = np.where(abs_y == max_abs_y)[0][0]
    
    valid_indices_flat = np.where(valid_mask)[0]
    max_x_orig_idx = valid_indices_flat[max_x_idx_flat]
    max_y_orig_idx = valid_indices_flat[max_y_idx_flat]
    
    max_x_sample = max_x_orig_idx // decoder_inputs.shape[1]
    max_x_pos = max_x_orig_idx % decoder_inputs.shape[1]
    
    max_y_sample = max_y_orig_idx // decoder_inputs.shape[1]
    max_y_pos = max_y_orig_idx % decoder_inputs.shape[1]
    
    max_x_example = decoder_inputs[max_x_sample]
    max_y_example = decoder_inputs[max_y_sample]
    
    return max_abs_x, max_abs_y, (max_x_sample, max_x_pos, max_x_example), (max_y_sample, max_y_pos, max_y_example)


# Usage example
if __name__ == "__main__":
    data_dir = "/home/anurizada/Documents/processed_dataset"
    
    # Quick check
    max_abs_x, max_abs_y, max_x_example, max_y_example = get_max_joint_values_with_example(data_dir)
    print(f"Maximum absolute x value: {max_abs_x:.6f}")
    print(f"Maximum absolute y value: {max_abs_y:.6f}")
    
    if max_x_example:
        sample_idx, pos_idx, array = max_x_example
        print(f"\nExample with max x value (sample {sample_idx}, position {pos_idx}):")
        for j, pos in enumerate(array):
            marker = " ← MAX X" if j == pos_idx else ""
            print(f"  Position {j:2d}: {pos}{marker}")
    
    if max_y_example:
        sample_idx, pos_idx, array = max_y_example
        print(f"\nExample with max y value (sample {sample_idx}, position {pos_idx}):")
        for j, pos in enumerate(array):
            marker = " ← MAX Y" if j == pos_idx else ""
            print(f"  Position {j:2d}: {pos}{marker}")
    
    # Detailed analysis
    print("\n" + "="*80)
    print("RUNNING DETAILED ANALYSIS...")
    print("="*80)
    
    stats = find_max_joint_values_with_examples(data_dir)
    print_detailed_statistics_with_examples(stats)