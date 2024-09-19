import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class IndexedDataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        data = self.original_dataset[idx]
        return idx, data


# Define a transform to normalize the data
transform = transforms.Compose(
    [transforms.ToTensor()]  # , transforms.Normalize((0.1307,), (0.3081,))]
)

# Download and load the training data
train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
indexed_train_dataset = IndexedDataset(train_dataset)

# Create data loaders
train_loader = DataLoader(indexed_train_dataset, batch_size=128, shuffle=False)


class PosFourierEmbedding(torch.nn.Module):
    def __init__(self, in_dim, out_dim, std: float = 4.0):
        super().__init__()

        self.register_buffer("projection", torch.randn(out_dim // 2, in_dim) * std)

    def forward(self, x):
        x_shape = x.size()
        x = x.reshape(-1, 2)
        x_proj = (2.0 * torch.pi * x) @ self.projection.t()
        x_proj = x_proj.reshape(*x_shape[:-1], self.projection.size(0))
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Generator(torch.nn.Module):
    def __init__(
        self,
        vocab_size: int = 128,
        hidden_dim: int = 256,
        channels: int = 1,
        num_layers: int = 4,
        width: int = 28,
        height: int = 28,
    ):
        super().__init__()

        self.embeddings = torch.nn.Embedding(vocab_size, hidden_dim)
        self.linear_layers = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.out = torch.nn.Linear(hidden_dim, channels)

        pos_encoder = PosFourierEmbedding(2, hidden_dim)
        x = torch.linspace(0, 1, width)
        y = torch.linspace(0, 1, height)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        position_matrix = torch.stack((grid_x, grid_y), dim=-1)
        position_embedding_matrix = pos_encoder(position_matrix)

        self.register_buffer("position_embedding_matrix", position_embedding_matrix)

    def forward(self, x):
        x = self.embeddings(x)
        x = x.unsqueeze(-2).unsqueeze(-2) + self.position_embedding_matrix.unsqueeze(0)
        for layer in self.linear_layers:
            x = x + torch.relu(layer(x))
        return self.out(x)


def display_tensor_grid(tensor):
    # Ensure the tensor is on CPU and convert to numpy array
    images = tensor.cpu().squeeze().numpy()

    # Create a grid of subplots
    fig, axes = plt.subplots(4, 8, figsize=(20, 10))
    fig.suptitle("Grid of 32 Grayscale Images (28x28)")

    # Plot each image
    for i, ax in enumerate(axes.flat):
        if i < 32:
            ax.imshow(images[i], cmap="gray")
            ax.axis("off")
        else:
            ax.remove()  # Remove extra subplots if any

    plt.tight_layout()
    plt.show()


model = Generator(128)
optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01, lr=1e-3)

NUM_EPOCHS = 1000


for epoch in range(NUM_EPOCHS):
    for idx, (x, y) in train_loader:
        x = x.permute(0, 2, 3, 1)
        optimizer.zero_grad()
        preds = model(idx)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(preds, x)
        loss.backward()
        optimizer.step()

        print(loss.item(), end="\t\t\t\r")
        break
    if epoch % 100 == 0:
        display_tensor_grid(torch.sigmoid(preds.detach()))
        display_tensor_grid(x)
