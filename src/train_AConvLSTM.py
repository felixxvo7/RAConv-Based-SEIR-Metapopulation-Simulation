import sys
from pathlib import Path

# Add parent directory to path so we can import Models
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

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
    # =================================Train=================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparams
    batch_size = 4
    P = 8   # input steps
    Q = 4   # predict steps
    H, W = 16, 16
    epochs = 2

    model = AConvLSTMLayers(
        input_channels=1,
        hidden_channels=[256, 256],
        kernel_size=[3, 3],
        num_layers=2
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()


    # Dummy dataset generation for testing
    # 1. Generate 1000 frames of continuous "historical" data
    full_dataset = generate_continuous_blob(total_steps=300, H=H, W=W)

    # 2. Slice it into training samples of length 12 (P=8, Q=4)
    sliding_windows_full_dataset = create_sliding_windows(full_dataset, P=P, Q=Q)
    
    # 3. sliding_windows_full_dataset shape is (989, 12, 1, 16, 16)
    # Wrap it so PyTorch recognizes it as a Dataset
    train_ds = TensorDataset(sliding_windows_full_dataset)

    # 4. Create a DataLoader
    # batch_size=4: It will take 4 samples at a time
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train() # Set to train mode once per epoch
        epoch_loss = 0.0
        
        for batch in train_loader:
            data = batch[0].to(device)
            input_seq = data[:, :P]
            target_seq = data[:, P:]
        
            # Forward
            _, last_state_list = model(input_seq)

            # Predict
            preds = model.predict_future(
                last_state_list,
                Q=Q,
                first_input=input_seq[:, -1]
            )

            loss = criterion(preds, target_seq)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Print average loss for the whole epoch
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.6f}")

    # Save the trained model
    model_dir = Path(__file__).parent.parent / "checkpoints"
    model_dir.mkdir(exist_ok=True)
    
    model_path = model_dir / "aconvlstm_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()