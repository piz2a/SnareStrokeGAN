import torch
import os


# Create embedding layer from trained data or initial data. If there is none, it creates one
def get_embedding_layer():
    if os.path.exists('./resources/embedding/trained_embedding.pt'):
        weight = torch.load('./resources/embedding/trained_embedding.pt', weights_only=True)
    elif os.path.exists('./resources/embedding/initial_embedding.pt'):
        weight = torch.load('./resources/embedding/initial_embedding.pt', weights_only=True)
    else:
        return None
    return torch.nn.Embedding.from_pretrained(weight)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_layer = get_embedding_layer().to(device)
    print(embedding_layer)
