import os
import json
import glob
import re
import matplotlib.pyplot as plt

MAX_STEPS = 500

# Hardcoded list of target directories (modify these paths)
TARGET_DIRS = [
    "logs/dpsk_distill_qwen3_8b_pacevolve_eplb_grpo_bs1_eplb",
    "logs/dpsk_distill_qwen3_8b_pacevolve_eplb_pkpo"
    # Add more directories as needed
]

def extract_step_number(filepath):
    """Extracts the integer step number from the 'step_xxxxx' folder name."""
    match = re.search(r'step_(\d+)', filepath)
    return int(match.group(1)) if match else -1

def process_directories(target_dirs):
    """Parses JSON files and extracts metrics for all directories."""
    all_data = {}
    
    for t_dir in target_dirs:
        # Find all metrics.json files within the records/step_* subdirectories
        search_pattern = os.path.join(t_dir, "records", "step_*", "metrics.json")
        filepaths = glob.glob(search_pattern)
        
        if not filepaths:
            print(f"Warning: No metrics.json files found in {t_dir}")
            continue
            
        # Sort files chronologically by step number
        filepaths.sort(key=extract_step_number)
        
        steps = []
        metrics_history = {}
        max_reward_history = []
        cumulative_max_reward = -float('inf')
        
        for filepath in filepaths:
            step_num = extract_step_number(filepath)
            
            with open(filepath, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading JSON from {filepath}")
                    continue
            
            steps.append(step_num)
            
            # 1. Process Train Metrics
            train_data = data.get("train", {})
            for key, value in train_data.items():
                if key not in metrics_history:
                    metrics_history[key] = []
                metrics_history[key].append(value)
                
            # 2. Process Cumulative Max Reward
            candidates = data.get("candidates", [])
            if candidates:
                # Find the max reward in the current step
                current_step_max = max(c.get("score", -float('inf')) for c in candidates)
                # Update the running cumulative max
                cumulative_max_reward = max(cumulative_max_reward, current_step_max)
            
            max_reward_history.append(cumulative_max_reward)

        # Limit to max steps
        valid_indices = [i for i, s in enumerate(steps) if s <= MAX_STEPS]
        steps = [steps[i] for i in valid_indices]
        max_reward_history = [max_reward_history[i] for i in valid_indices]
        metrics_history = {
            k: [v[i] for i in valid_indices]
            for k, v in metrics_history.items()
        }
            
        all_data[t_dir] = {
            "steps": steps,
            "train_metrics": metrics_history,
            "max_score_evolution": max_reward_history
        }
        
    return all_data

def plot_metrics(all_data):
    """Generates and saves head-to-head plots for each metric."""
    if not all_data:
        print("No data to plot.")
        return

    # Figure out all unique training metrics across all directories
    all_train_keys = set()
    for dir_data in all_data.values():
        all_train_keys.update(dir_data["train_metrics"].keys())
        
    # Create a plot for each train metric
    for metric in all_train_keys:
        plt.figure(figsize=(10, 6))
        for t_dir, data in all_data.items():
            if metric in data["train_metrics"]:
                # Use a cleaner label name (just the folder name rather than full path)
                label_name = os.path.basename(os.path.normpath(t_dir))
                plt.plot(data["steps"], data["train_metrics"][metric], label=label_name)
        
        plt.title(f"Train Metric: {metric} vs Steps")
        plt.xlabel("Steps")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        
        # Save the plot
        filename = f"plot_{metric}.png"
        plt.savefig(filename)
        print(f"Saved {filename}")
        plt.close()

    # Create the plot for Max Reward Evolution
    plt.figure(figsize=(10, 6))
    for t_dir, data in all_data.items():
        label_name = os.path.basename(os.path.normpath(t_dir))
        plt.plot(data["steps"], data["max_score_evolution"], label=label_name, linewidth=2)
        
    plt.title("Cumulative Max Reward vs Steps")
    plt.xlabel("Steps")
    plt.ylabel("Max Reward")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    filename = "plot_max_score_evolution.png"
    plt.savefig(filename)
    print(f"Saved {filename}")
    plt.close()

if __name__ == "__main__":
    print("Parsing directories...")
    parsed_data = process_directories(TARGET_DIRS)
    print("Generating plots...")
    plot_metrics(parsed_data)
    print("Done!")