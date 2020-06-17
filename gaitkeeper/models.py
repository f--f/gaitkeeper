import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

EMBEDDING_SIZE = 20  # Size of output embedding


def load_embedding_model(pt_file):
    """Return an EmbeddingNet model with saved model weights, usable for inference only."""
    model = EmbeddingNet()
    # Explicitly map CUDA-trained models to CPU otherwise this will raise an error
    model.load_state_dict(torch.load(pt_file, map_location=torch.device('cpu')))
    model.eval()
    return model


def extract_embeddings(dataloader, model):
    """Return embeddings from a model with a get_embedding method (uses CPU)."""
    model = model.cpu()
    with torch.no_grad():
        model.eval()
        embeddings = np.zeros((len(dataloader.dataset), EMBEDDING_SIZE))
        labels = np.zeros(len(dataloader.dataset))
        count = 0
        for input_data, target in dataloader:
            embeddings[count:count+len(input_data), :] = model.get_embedding(input_data).data.cpu().numpy()
            labels[count:count+len(input_data)] = target.numpy()
            count += len(input_data)
    return embeddings


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
    def __init__(self):
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
        self.fc = nn.Linear(in_features=32 * 64, out_features=EMBEDDING_SIZE)
        
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
        super(ClassificationNet, self).__init__()
        self.embedding_net = embedding_net
        self.n_classes = n_classes
        self.nonlinear = nn.ReLU()
        self.fc1 = nn.Linear(EMBEDDING_SIZE, n_classes)

    def forward(self, x):
        output = self.embedding_net(x)
        output = self.nonlinear(output)
        scores = F.log_softmax(self.fc1(output), dim=-1)
        return scores

    def get_embedding(self, x):
        return self.nonlinear(self.embedding_net(x))