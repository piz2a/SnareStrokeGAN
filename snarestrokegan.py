import torch
from torch.utils.data import DataLoader
from model.generator import Generator, get_embedding_layer
from model.discriminator import Discriminator
from preprocessing.stroke_annotation_dataset import StrokeAnnotationDataset
from tqdm.notebook import tqdm


def train():
    n = 14676
    batch_size = 8
    frame_count = 48000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_layer = get_embedding_layer('./resources/embedding').to(device)
    generator = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    discriminator = Discriminator(device, frame_count * 2)
    optimizer_g = torch.optim.Adam(generator.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())
    criterion = torch.nn.BCELoss()

    epochs = 100

    min_annotation_count, max_annotation_count = 3, 12
    train_data_loaders, test_data_loaders = [], []
    for annotation_count in range(min_annotation_count, max_annotation_count + 1):
        train_dataset = StrokeAnnotationDataset('./resources', True, 0.8, frame_count, annotation_count)
        test_dataset = StrokeAnnotationDataset('./resources', False, 0.2, frame_count, annotation_count)
        train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_data_loader = DataLoader(test_dataset, batch_size, shuffle=True)
        train_data_loaders.append(train_data_loader)
        test_data_loaders.append(test_data_loader)

    p_real_trace = []
    p_fake_trace = []
    for epoch in tqdm(range(epochs)):
        # Train
        generator.train()
        discriminator.train()
        for annotation_count in range(max_annotation_count - min_annotation_count + 1):
            print('Annotation count:', annotation_count)
            train_data_loader = train_data_loaders[annotation_count]
            for annotations, sample in train_data_loader:
                annotations, sample = annotations.to(device), sample.to(device)
                # D: max V(D, G)
                p_real = discriminator(annotations, sample)
                z = torch.randn((batch_size, annotation_count, n-1))
                p_fake = discriminator(generator(z, annotations))
                loss_d = criterion(p_real, torch.ones_like(p_real).to(device)) + criterion(p_fake, torch.zeros_like(p_real).to(device))
                loss_d.backward()
                optimizer_d.step()

                # G: min V(D, G)
                optimizer_g.zero_grad()
                z = torch.randn((batch_size, annotation_count, n-1))
                p_fake = discriminator(generator(z, annotations))
                loss_g = criterion(p_fake, torch.ones_like(p_fake).to(device))
                loss_g.backward()
                optimizer_g.step()

        # Evaluate
        p_real, p_fake = 0., 0.
        generator.eval()
        discriminator.eval()
        for annotation_count in range(max_annotation_count - min_annotation_count + 1):
            print('Annotation count:', annotation_count)
            test_data_loader = test_data_loaders[annotation_count]
            for annotations, sample in test_data_loader:
                annotations, sample = annotations.to(device), sample.to(device)
                with torch.autograd.no_grad():
                    p_real += (torch.sum(discriminator(annotations, sample)).item()) / batch_size
                    z = torch.randn((batch_size, annotation_count, n-1))
                    p_fake += (torch.sum(discriminator(generator(z, annotations))).item()) / batch_size
        p_real_trace.append(p_real)
        p_fake_trace.append(p_fake)

        if (epoch+1) % 10 == 0:
            print(f'(epoch {epoch+1}/{epochs}) p_real: {p_real}, p_g: {p_fake}')


if __name__ == '__main__':
    train()
