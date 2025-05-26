import json
import matplotlib.pyplot as plt
import numpy as np

# --- Configuration (Adjust these as needed) ---
METRICS_FILE_PATH = "results/rd_esi_debug_run/metrics.json" # Or your specific path
LOG_INTERVAL = 10  # The 'log_interval' from your trainer configuration (for training metrics)
DOWNSAMPLE_TRAIN_METRICS = 10 # Plot every 10th point for dense training metrics
MOVING_AVERAGE_WINDOW = 50 # Window size for smoothing training loss (optional)

# --- 1. Load the JSON Data ---
try:
    with open(METRICS_FILE_PATH, 'r') as f:
        metrics = json.load(f)
    print(f"Successfully loaded metrics from {METRICS_FILE_PATH}")
    print(f"Available metric keys: {list(metrics.keys())}")
except FileNotFoundError:
    print(f"Error: Metrics file not found at {METRICS_FILE_PATH}")
    exit()
except json.JSONDecodeError:
    print(f"Error: Could not decode JSON from {METRICS_FILE_PATH}")
    exit()

# --- 2. Extract Data Series & 3. Prepare X-axis Values ---

# Validation metrics
val_loss = metrics.get("val_loss", [])
val_ppl = metrics.get("val_ppl", [])
eval_steps = metrics.get("eval_steps", []) # This is your x-axis for validation metrics

# Training metrics
train_loss_raw = metrics.get("train_loss", [])
expert_load_variance_raw = metrics.get("expert_load_variance", [])
expert_load_cv_raw = metrics.get("expert_load_cv", [])
expert_load_entropy_raw = metrics.get("expert_load_entropy", [])
# Add other training metrics as needed

# Create x-axis for training metrics (assuming they are logged at each log_interval * global_step)
# global_step in your trainer increments after gradient_accumulation_steps
# So, if train_loss is logged every log_interval optimizer steps:
train_steps_raw = [i * LOG_INTERVAL for i in range(len(train_loss_raw))]

# --- 4. Plot the Data & 5. Handle Large Number of Points ---

plt.style.use('seaborn-v0_8-whitegrid') # Using a nice style

# Function to apply moving average
def moving_average(data, window_size):
    if not data or window_size <= 0:
        return []
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Plot 1: Training and Validation Loss
plt.figure(figsize=(12, 6))
# Downsample training loss for plotting
train_loss_plot = train_loss_raw[::DOWNSAMPLE_TRAIN_METRICS]
train_steps_plot = train_steps_raw[::DOWNSAMPLE_TRAIN_METRICS]

# Optional: Smoothed training loss
if len(train_loss_raw) >= MOVING_AVERAGE_WINDOW:
    train_loss_smoothed = moving_average(train_loss_raw, MOVING_AVERAGE_WINDOW)
    # Adjust steps for smoothed data (it will be shorter)
    train_steps_smoothed = train_steps_raw[MOVING_AVERAGE_WINDOW-1:]
    plt.plot(train_steps_smoothed, train_loss_smoothed, label=f'Smoothed Train Loss (window {MOVING_AVERAGE_WINDOW})', alpha=0.7, color='lightblue')

plt.plot(train_steps_plot, train_loss_plot, label='Train Loss (Downsampled)', alpha=0.8)

if eval_steps and val_loss:
    plt.plot(eval_steps, val_loss, label='Validation Loss', marker='o', linestyle='--')
plt.xlabel('Training Steps (Optimizer Steps)')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Steps')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("loss_plot.png") # Save the plot
plt.show()

# Plot 2: Validation Perplexity
if eval_steps and val_ppl:
    plt.figure(figsize=(12, 6))
    plt.plot(eval_steps, val_ppl, label='Validation Perplexity', marker='o', color='green')
    plt.xlabel('Training Steps (Optimizer Steps)')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity Over Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("perplexity_plot.png")
    plt.show()

# Plot 3: Expert Load Variance (Downsampled)
if expert_load_variance_raw:
    variance_plot = expert_load_variance_raw[::DOWNSAMPLE_TRAIN_METRICS]
    # train_steps_plot is already defined and downsampled
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps_plot, variance_plot, label='Expert Load Variance (Downsampled)', color='red', alpha=0.8)
    plt.xlabel('Training Steps (Optimizer Steps)')
    plt.ylabel('Variance')
    plt.title('Expert Load Variance Over Steps')
    plt.yscale('log') # Variance can be very large, log scale might be useful
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("variance_plot.png")
    plt.show()

# Plot 4: Expert Load CV (Downsampled)
if expert_load_cv_raw:
    cv_plot = expert_load_cv_raw[::DOWNSAMPLE_TRAIN_METRICS]
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps_plot, cv_plot, label='Expert Load CV (Downsampled)', color='purple', alpha=0.8)
    plt.xlabel('Training Steps (Optimizer Steps)')
    plt.ylabel('Coefficient of Variation (CV)')
    plt.title('Expert Load CV Over Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("cv_plot.png")
    plt.show()

# Plot 5: Expert Load Entropy (Downsampled)
if expert_load_entropy_raw:
    entropy_plot = expert_load_entropy_raw[::DOWNSAMPLE_TRAIN_METRICS]
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps_plot, entropy_plot, label='Expert Load Entropy (Downsampled)', color='orange', alpha=0.8)
    plt.xlabel('Training Steps (Optimizer Steps)')
    plt.ylabel('Entropy')
    plt.title('Expert Load Entropy Over Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("entropy_plot.png")
    plt.show()

print("Plots generated and saved.")