import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


def load_embedding_model(pt_file, embedding_size):
    """Return an EmbeddingNet model with saved model weights, usable for inference only."""
    model = EmbeddingNet(embedding_size)
    # Explicitly map CUDA-trained models to CPU otherwise this will raise an error
    model.load_state_dict(torch.load(pt_file, map_location=torch.device('cpu')))
    model.eval()
    return model


def extract_embeddings(dataloader, model):
    """Return embeddings from a model with a get_embedding method (uses CPU)."""
    model = model.cpu()
    with torch.no_grad():
        model.eval()
        embedding_size = list(model.children())[-1].out_features
        embeddings = np.zeros((len(dataloader.dataset), embedding_size))
        labels = np.zeros(len(dataloader.dataset))
        count = 0
        for input_data, target in dataloader:
            embeddings[count:count+len(input_data), :] = model.get_embedding(input_data).data.cpu().numpy()
            labels[count:count+len(input_data)] = target.numpy()
            count += len(input_data)
    return embeddings, labels


class GaitDataset(torch.utils.data.Dataset):
    """Classification-based dataset which returns individual samples.
    Class signature is based on the PyTorch MNIST dataset."""
    def __init__(self, dfs, train=True):
        """dfs is a list of DataFrames corresponding to chunked data."""
        self._dfs = dfs
        self.train = train
        self.targets = torch.Tensor([df["user_id"].iloc[0] for df in dfs]).long()
        self.data = torch.Tensor([
            np.stack([
                chunk["linearaccelerometer_mag"].values, 
                chunk["gyroscope_mag"].values,
            ]) 
            for chunk in self._dfs
        ])
        self.transform = None
    @property
    def train_labels(self):
        return self.targets
    @property
    def test_labels(self):
        return self.targets
    @property
    def train_data(self):
        return self.data
    @property
    def test_data(self):
        return self.data
    def __getitem__(self, idx):
        return self.data[idx,:,:], self.targets[idx]
    def __len__(self):
        return len(self._dfs)


class EmbeddingNet(nn.Module):
    """Model definition for outputting a lower-dimensional embedding."""
    def __init__(self, embedding_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2, padding_mode="replicate"), nn.ReLU(),
            nn.Conv1d(16, 32, 5, padding=2, padding_mode="replicate"), nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(34, 64, 3, padding=1, padding_mode="replicate"), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1, padding_mode="replicate"), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Flatten()
        )
        self.fc = nn.Linear(in_features=32 * 64, out_features=embedding_size)
        
    def forward(self, x):
        conv1 = self.conv1(x)
        # Add FFT as intermediate channel, stack with conv1
        fft = self._rfft(x)
        encoder = self.conv2(torch.cat([conv1, fft], dim=1))
        embedding = self.fc(encoder)
        return embedding
    
    def get_embedding(self, x):
        return self.forward(x)

    def _rfft(self, signal, remove_mean=True):
        """Return FFT."""
        N = signal.shape[-1]
        if remove_mean:
            fft = torch.rfft(signal - signal.mean(), 1)
        else:
            fft = torch.rfft(signal, 1)
        # Clip last value so that size of output is N//2 (compatible with MaxPool)
        return (2/N * (fft[...,0].pow(2) + fft[...,1].pow(2)).sqrt())[...,:N//2]


class ClassificationNet(nn.Module):
    """Model definition for performing classification using embeddings."""
    def __init__(self, embedding_net, n_classes):
        super().__init__()
        self.embedding_net = embedding_net
        embedding_size = list(embedding_net.children())[-1].out_features
        self.n_classes = n_classes
        self.nonlinear = nn.ReLU()
        self.fc1 = nn.Linear(embedding_size, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))
    

def train_epoch(train_loader, model, loss_criterion, optimizer, device):
    """Run a single training epoch (update weights based on loss function).
    Arguments:
        train_loader: training DataLoader
        model: PyTorch model object
        loss_criterion: loss function
        optimizer: optimizer
        device: device to put inputs from dataset on (should match model)
    Returns:
        loss: the loss at the end of the epoch
    """
    model.train()
    total_loss = 0  # for computing accuracy 
    accuracy = 0
    total = 0
    for i, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criterion(out, target)
        # compute accumulated gradients
        loss.backward()
        # perform parameter update based on current gradients
        optimizer.step()
        total_loss += loss.item()
        accuracy += (out.argmax(dim=1) == target).sum().item()
        total += target.size(0)
    accuracy /= total
    total_loss /= len(train_loader)
    return loss, accuracy


def test_epoch(test_loader, model, loss_criterion, device):
    """Run a single validation epoch (run model in inference without updating weights).
    Arguments:
        test_loader: test DataLoader
        model: PyTorch model object
        loss_criterion: loss function
        device: device to put inputs from dataset on (should match model)
    Returns:
        loss: the loss at the end of the epoch
    """
    total_loss = 0  # for computing accuracy 
    accuracy = 0
    total = 0
    with torch.no_grad():
        model.eval()
        for i, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)
            out = model(data)
            loss = loss_criterion(out, target)
            total_loss += loss.item()
            accuracy += (out.argmax(dim=1) == target).sum().item()
            total += target.size(0)
        accuracy /= total
        total_loss /= len(test_loader)
        return loss, accuracy