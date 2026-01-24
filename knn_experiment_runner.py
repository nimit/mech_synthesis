import os
import re
import subprocess
import time

k_values = [1, 2, 4, 8, 16, 32]
results_file = "knn_varying_results.md"
target_script = "dtw_calc_for_knn_continuous.py"

def parse_results(k, output_dir):
    stats_path = os.path.join(output_dir, "final_dtw_stats_continuous.txt")
    if not os.path.exists(stats_path):
        return f"| {k} | Error | Error | Error | Error | Error | Error | Error |"
        
    with open(stats_path, 'r') as f:
        txt = f.read()
    
    try:
        mean = re.search(r"Mean Best DTW: ([\d\.]+)", txt).group(1)
        std = re.search(r"Std Dev Best DTW: ([\d\.]+)", txt).group(1)
        median = re.search(r"Median Best DTW: ([\d\.]+)", txt).group(1)
        min_val = re.search(r"Min Best DTW: ([\d\.]+)", txt).group(1)
        max_val = re.search(r"Max Best DTW: ([\d\.]+)", txt).group(1)
        
        below_1_match = re.search(r"Best < 1.0: \d+ \(([\d\.]+)%\)", txt)
        below_1_pct = below_1_match.group(1) if below_1_match else "0.00"
        
        below_2_match = re.search(r"Best < 2.0: \d+ \(([\d\.]+)%\)", txt)
        below_2_pct = below_2_match.group(1) if below_2_match else "0.00"
        
        below_3_match = re.search(r"Best < 3.0: \d+ \(([\d\.]+)%\)", txt)
        below_3_pct = below_3_match.group(1) if below_3_match else "0.00"
        
        row = f"| {k} | {median} | {float(mean):.4f} ± {std} | {below_3_pct}% | {below_2_pct}% | {below_1_pct}% | {min_val} | {max_val} |"
        return row
    except Exception as e:
        print(f"Error parsing results for k={k}: {e}")
        return f"| {k} | Parse Err | - | - | - | - | - | - |"

output_lines = []
output_lines.append("| knn_neighbors | ηDTW | µDTW+-std | DTW < 3.0 | DTW < 2.0 | DTW < 1.0 | min_best_dtw | max_best_dtw |")
output_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

print("Starting experiment...")
for k in k_values:
    print(f"Running for k={k}...")
    output_dir = f"knn_results_k{k}"
    
    cmd = [
        "python", target_script,
        "--num_neighbors", str(k),
        "--output_dir", output_dir,
        "--num_samples", "100" # Running with 100 samples as default/requested
    ]
    
    subprocess.run(cmd, check=True)
    
    row = parse_results(k, output_dir)
    output_lines.append(row)
    print(f"Result for k={k}: {row}")

# Write to file
with open(results_file, 'w') as f:
    f.write("\n".join(output_lines))

print(f"Done! Written table to {results_file}")
