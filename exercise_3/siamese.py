import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import struct


class BaseNetwork(nn.Module):
    """
    Defines a base neural network architecture for feature extraction.
    """

    def __init__(self):
        super(BaseNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the fully connected layers.
        """
        return self.fc(x)


class SiameseNetwork(nn.Module):
    """
    Siamese Network that uses a shared base network to process two inputs.
    """

    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.base_network = BaseNetwork()

    def forward(self, input1, input2):
        """
        Forward pass for two inputs.

        Args:
            input1 (torch.Tensor): First input tensor.
            input2 (torch.Tensor): Second input tensor.

        Returns:
            tuple: Outputs from the base network for both inputs.
        """
        output1 = self.base_network(input1)
        output2 = self.base_network(input2)
        return output1, output2


class TripletLoss(nn.Module):
    """
    Triplet Loss function to train the Siamese network.
    """

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        """
        Compute the triplet loss.

        Args:
            anchor (torch.Tensor): Embeddings for the anchor sample.
            positive (torch.Tensor): Embeddings for the positive sample.
            negative (torch.Tensor): Embeddings for the negative sample.

        Returns:
            torch.Tensor: Computed triplet loss.
        """
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = torch.mean(torch.clamp(pos_dist - neg_dist + self.margin, min=0.0))
        return loss


class TripletMNIST(Dataset):
    """
    Custom Dataset for creating triplets from MNIST data.
    """

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return len(self.images)

    def __getitem__(self, index):
        """
        Get a triplet sample (anchor, positive, negative).

        Args:
            index (int): Index of the anchor sample.

        Returns:
            tuple: (anchor, positive, negative) tensors.
        """
        anchor_img, anchor_label = self.images[index], self.labels[index]
        positive_idx = np.where(self.labels == anchor_label)[0]
        negative_idx = np.where(self.labels != anchor_label)[0]

        positive_idx = positive_idx[np.random.randint(0, len(positive_idx))]
        negative_idx = negative_idx[np.random.randint(0, len(negative_idx))]

        positive_img = self.images[positive_idx]
        negative_img = self.images[negative_idx]

        return (
            torch.tensor(anchor_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(positive_img, dtype=torch.float32).unsqueeze(0),
            torch.tensor(negative_img, dtype=torch.float32).unsqueeze(0),
        )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SiameseNetwork().to(device)
criterion = TripletLoss(margin=0.2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train(model, criterion, optimizer, dataloader, epochs=10):
    """
    Train the Siamese Network.

    Args:
        model (nn.Module): The Siamese Network model.
        criterion (nn.Module): The loss function (Triplet Loss).
        optimizer (optim.Optimizer): Optimizer for training.
        dataloader (DataLoader): DataLoader for the training dataset.
        epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for anchor, positive, negative in dataloader:
            anchor, positive, negative = (
                anchor.to(device),
                positive.to(device),
                negative.to(device),
            )
            optimizer.zero_grad()
            anchor_out, positive_out = model(anchor, positive)
            _, negative_out = model(anchor, negative)
            loss = criterion(anchor_out, positive_out, negative_out)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")


class CFG:
    """
    Configuration class for file paths.
    """

    train_images_path = "../assets/MNIST/train-images-idx3-ubyte"
    train_labels_path = "../assets/MNIST/train-labels-idx1-ubyte"
    test_images_path = "../assets/MNIST/t10k-images-idx3-ubyte"
    test_labels_path = "../assets/MNIST/t10k-labels-idx1-ubyte"


def read_ubyte(file_name):
    """
    Read MNIST image or label file in the ubyte format.

    Args:
        file_name (str): Path to the ubyte file.

    Returns:
        np.ndarray: Array of images or labels.
    """
    with open(file_name, "rb") as file:
        _, _, dims = struct.unpack(">HBB", file.read(4))
        shape = tuple(struct.unpack(">I", file.read(4))[0] for _ in range(dims))
        return np.frombuffer(file.read(), dtype=np.uint8).reshape(shape)


train_images = read_ubyte(CFG.train_images_path)
train_images = train_images.reshape(-1, 28 * 28)
train_labels = read_ubyte(CFG.train_labels_path)

test_images = read_ubyte(CFG.test_images_path)
test_images = test_images.reshape(-1, 28 * 28)
test_labels = read_ubyte(CFG.test_labels_path)

train_triplet_dataset = TripletMNIST(train_images, train_labels)
train_loader = DataLoader(train_triplet_dataset, batch_size=32, shuffle=True)
train(model, criterion, optimizer, train_loader)
