import sys
from pathlib import Path

# Add parent directory to path so we can import Models
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import matplotlib.pyplot as plt
import numpy as np
import random

from Models.ConvLSTM.AConvLSTM import AConvLSTMLayers

def generate_continuous_blob(total_steps, H, W):
    """
    Generate ONE long, continuous moving square pattern.
    Output shape: (total_steps, 1, H, W)
    """
    seq = torch.zeros(total_steps, 1, H, W)
    
    # Start at a fixed position
    x, y = 0, 0  
    dx, dy = 1, 1  # constant motion

    for t in range(total_steps):
        # Draw the 4x4 square
        seq[t, 0, x:x+12, y:y+12] = torch.rand(12, 12)

        # Move the square for the next frame
        x = (x + dx) % (H - 12)
        y = (y + dy) % (W - 12)

    return seq

def create_sliding_windows(continuous_data, P, Q):
    """
    Slice the continuous timeline into overlapping windows of size P+Q.
    Output shape: (Num_Samples, P+Q, 1, H, W)
    """
    total_steps = continuous_data.shape[0]
    window_size = P + Q
    windows = []

    # Slide the window across the timeline one step at a time
    for i in range(total_steps - window_size + 1):
        window = continuous_data[i : i + window_size]
        windows.append(window)

    # Stack the list of tensors into a single batch-like tensor
    dataset = torch.stack(windows)
    
    return dataset

def main():
    # ================================Inference=================================
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams (must match training)
    P = 8   # input steps
    Q = 20   # predict steps
    H, W = 16, 16

    # Initialize model
    model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=[256, 256],
        kernel_size=[3, 3],
        num_layers=2
    ).to(device)

    # Load trained model
    model_path = Path(__file__).parent.parent / "checkpoints" / "aconvlstm_model.pth"
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    print(f"Model loaded from {model_path}")

    # Inference
    model.eval()

    # Generate test data
    full_dataset = generate_continuous_blob(total_steps=300, H=H, W=W)
    sliding_windows_full_dataset = create_sliding_windows(full_dataset, P=P, Q=Q)

    with torch.no_grad():
        # Example: take one sample from dataset
        sample = sliding_windows_full_dataset[0].unsqueeze(0).to(device)
        
        input_seq = sample[:, :P]   # (1, P, 1, H, W)
        target_seq = sample[:, P:]  # (1, Q, 1, H, W) ← ground truth (optional)

        # Step 1: encode past
        _, last_state_list = model(input_seq)

        # Step 2: predict future
        preds = model.predict_future(
            last_state_list,
            Q=Q,
            first_input=input_seq[:, -1]
        )

    # ======================Plotting======================
    # Making Figure 7: Plot predicted vs ground truth for a random pixel over the Q future frames
    # Ensure preds and target_seq are on CPU for numpy conversion
    preds_cpu = preds.cpu() if preds.is_cuda else preds
    target_seq_cpu = target_seq.cpu() if target_seq.is_cuda else target_seq

    # Randomly select a pixel (h, w) for visualization
    h_rand = random.randint(0, H - 1)
    w_rand = random.randint(0, W - 1)

    # Extract time series for the selected pixel
    # preds_cpu shape: (Batch, Q, C, H, W). Assuming C=1, squeeze it out.
    # We are taking the first sample from the batch (sample_idx=0)
    predicted_pixel_series = preds_cpu[0, :, 0, h_rand, w_rand].numpy()
    ground_truth_pixel_series = target_seq_cpu[0, :, 0, h_rand, w_rand].numpy()

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, Q + 1), ground_truth_pixel_series, label='Ground Truth', marker='o')
    plt.plot(range(1, Q + 1), predicted_pixel_series, label='Predicted', marker='x')

    plt.title(f'Pixel Value Over Time at H={h_rand}, W={w_rand}')
    plt.xlabel('Time Frame (Q)')
    plt.ylabel('Pixel Intensity Value')
    plt.xticks(range(1, Q + 1))
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

if __name__ == "__main__":
    main()
