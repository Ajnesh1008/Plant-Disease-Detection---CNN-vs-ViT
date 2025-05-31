import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from cnn_model import CNNModel
from vit_model import ViTModel
import time

# Config
batch_size = 32
epochs = 2
num_classes = 15
lr = 0.001
dataset_dir = 'data/plant_disease_dataset'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataloaders
train_data = datasets.ImageFolder(os.path.join(dataset_dir, 'train'), transform=transform)
val_data = datasets.ImageFolder(os.path.join(dataset_dir, 'val'), transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size)


# Training function with progress tracking
def train(model, name):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        start_epoch = time.time()  # Start time for each epoch

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Display progress every 10 batches
            if i % 10 == 0:
                print(f"{name} Epoch {epoch + 1}/{epochs}, Batch {i + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # End of epoch progress
        epoch_duration = time.time() - start_epoch
        avg_loss = running_loss / len(train_loader)
        print(
            f"{name} Epoch {epoch + 1}/{epochs} completed in {epoch_duration:.2f} seconds. Average loss: {avg_loss:.4f}")

    # Save the model after training
    torch.save(model.state_dict(), f'models/{name}_model.pth')
    print(f"{name} model saved to models/{name}_model.pth")


# Train CNN
cnn = CNNModel(num_classes)
train(cnn, 'cnn')

# Train ViT
vit = ViTModel(num_classes)
train(vit, 'vit')
