import torch

from model.generator import Generator, get_embedding_layer
from model.discriminator import Discriminator


def train(n, batch_size, frame_count, device, optimizer, loss_fn, generator, discriminator):
    z = torch.randn((batch_size, 8, n-1))
    alpha = torch.LongTensor([[0, 3000, 6000, 9000, 12000, 15000, 18000, 21000], [0, 4500, 6000, 10500, 12000, 16500, 18000, 22500]])
    result = generator(z, alpha)


def main():
    n = 14676
    batch_size = 2
    frame_count = 24000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedding_layer = get_embedding_layer('./resources/embedding').to(device)
    generator = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    discriminator = Discriminator(device, frame_count * 2)
    optimizer = torch.optim.Adam(generator.parameters())
    loss_fn = torch.nn.BCELoss()
    train(n, batch_size, frame_count, device, optimizer, loss_fn, generator, discriminator)


if __name__ == '__main__':
    main()
