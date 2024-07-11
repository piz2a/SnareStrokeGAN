import torch
from torch import nn
import os


# Generate initial embedding from snare drum sound resources
def generate_initial_embedding(resource_root: str):
    result = torch.LongTensor()
    return result


# Create embedding layer from trained data or initial data. If there is none, it creates one
def get_embedding_layer(resource_root: str):
    if os.path.exists('trained_embedding.pt'):
        weight = torch.load('trained_embedding.pt', weights_only=True)
    elif os.path.exists('initial_embedding.pt'):
        weight = torch.load('initial_embedding.pt', weights_only=True)
    else:
        weight = generate_initial_embedding(resource_root)
    return nn.Embedding.from_pretrained(weight)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resource_root = './resources'
    embedding_layer = get_embedding_layer(resource_root).to(device)