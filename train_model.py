import os
import struct
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from model import CVAE, loss_function

# --- Load MNIST IDX Files ---
def load_idx_images(path):
    with open(path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, rows, cols)
        return images.astype(np.float32) / 255.0

def load_idx_labels(path):
    with open(path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# --- Main Training Function ---
def train_vae(epochs=20, batch_size=64, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load IDX files
    image_path = os.path.join('MNIST_ORG', 'train-images.idx3-ubyte')
    label_path = os.path.join('MNIST_ORG', 'train-labels.idx1-ubyte')

    if not os.path.exists(image_path) or not os.path.exists(label_path):
        print("MNIST IDX files not found!")
        return

    images = load_idx_images(image_path)
    labels = load_idx_labels(label_path)

    images_tensor = torch.tensor(images)
    labels_tensor = torch.tensor(labels)

    dataset = TensorDataset(images_tensor, labels_tensor)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = CVAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    train_losses = []

    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.to(device).long()  # Ensure correct dtype

            optimizer.zero_grad()
            labels_onehot = F.one_hot(labels, num_classes=10).float().to(device)

            recon_batch, mu, logvar = model(data.view(-1, 784), labels_onehot)
            loss = loss_function(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] "
                      f"Loss: {loss.item() / len(data):.6f}")

        avg_loss = train_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"===> Epoch {epoch} Average loss: {avg_loss:.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    torch.save(model.state_dict(), 'models/cvae_mnist.pth')
    print("✅ Model saved to models/cvae_mnist.pth")

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses)
    plt.title('CVAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.show()

    return model

# --- Entry Point ---
if __name__ == "__main__":
    print("Starting CVAE training using MNIST training data...")
    model = train_vae(epochs=20)
    print("✅ Training completed.")
