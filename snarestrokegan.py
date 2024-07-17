import torch
from torch.utils.data import DataLoader
from model.generator import Generator, get_embedding_layer
from model.discriminator import Discriminator
from preprocessing.stroke_annotation_dataset import StrokeAnnotationDataset


def train(n, batch_size, frame_count, device, optimizer, loss_fn, generator, discriminator):
    z = torch.randn((batch_size, 8, n-1))
    alpha = torch.LongTensor([[0, 3000, 6000, 9000, 12000, 15000, 18000, 21000], [0, 4500, 6000, 10500, 12000, 16500, 18000, 22500]])
    result = generator(z, alpha)


def main():
    n = 14676
    batch_size = 8
    frame_count = 24000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = StrokeAnnotationDataset('./resources', train=True, ratio=0.8, frame_count=48000, augmentation=20)
    test_dataset = StrokeAnnotationDataset('./resources', train=False, ratio=0.2, frame_count=48000, augmentation=20)

    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    embedding_layer = get_embedding_layer('./resources/embedding').to(device)
    generator = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    discriminator = Discriminator(device, frame_count * 2)
    optimizer = torch.optim.Adam(generator.parameters())
    loss_fn = torch.nn.BCELoss()

    train(n, batch_size, frame_count, device, optimizer, loss_fn, generator, discriminator)


if __name__ == '__main__':
    main()
