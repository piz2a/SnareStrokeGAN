import torch
from torch.utils.data import DataLoader
from model.generator import Generator, get_embedding_layer
from model.discriminator import Discriminator
from preprocessing.stroke_annotation_dataset import StrokeAnnotationDataset
from tqdm.notebook import tqdm
import time


def train(device):
    n = 14676
    batch_size = 8
    frame_count = 48000

    embedding_layer = get_embedding_layer('./resources/embedding').to(device)
    generator = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    discriminator = Discriminator(device, frame_count)

    print('Generator parameters:', [p.numel() for p in generator.parameters()])
    print('Discriminator parameters:', [p.numel() for p in discriminator.parameters()])

    optimizer_g = torch.optim.Adam(generator.parameters())
    optimizer_d = torch.optim.Adam(discriminator.parameters())
    criterion = torch.nn.BCELoss()

    epochs = 10
    min_annotation_count, max_annotation_count = 3, 12

    train_data_loaders, test_data_loaders = {}, {}
    for annotation_count in range(min_annotation_count, max_annotation_count + 1):
        train_dataset = StrokeAnnotationDataset('./resources', True, 0.8, frame_count, annotation_count)
        test_dataset = StrokeAnnotationDataset('./resources', False, 0.2, frame_count, annotation_count)
        train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        test_data_loader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)
        train_data_loaders[annotation_count] = train_data_loader
        test_data_loaders[annotation_count] = test_data_loader
    train_annotations_count = sum(len(train_data_loader) for train_data_loader in train_data_loaders.values())
    test_annotations_count = sum(len(test_data_loader) for test_data_loader in test_data_loaders.values())

    p_real_trace = []
    p_fake_trace = []
    for epoch in tqdm(range(epochs)):
        # Train
        generator.train()
        discriminator.train()
        g_loss, d_loss = 0, 0
        for annotation_count in range(min_annotation_count, max_annotation_count + 1):
            print('Annotation count:', annotation_count)
            train_data_loader = train_data_loaders[annotation_count]
            print('Train DataLoader length:', len(train_data_loader))
            i = 0
            start_time = time.time()
            for annotations, sample in train_data_loader:
                annotations, sample = annotations.to(device), sample.to(device)
                # D: max V(D, G)
                p_real = discriminator(annotations, sample)
                z = torch.randn((batch_size, annotation_count, n-1)).to(device)
                p_fake = discriminator(annotations, generator(z, annotations))
                loss_d = criterion(p_real, torch.ones_like(p_real).to(device)) + criterion(p_fake, torch.zeros_like(p_real).to(device))
                d_loss += loss_d.item() / train_annotations_count
                loss_d.backward()
                optimizer_d.step()

                middle_time = time.time()

                # G: min V(D, G)
                optimizer_g.zero_grad()
                z = torch.randn((batch_size, annotation_count, n-1)).to(device)
                p_fake = discriminator(annotations, generator(z, annotations))
                loss_g = criterion(p_fake, torch.ones_like(p_fake).to(device))
                g_loss += loss_g.item() / train_annotations_count
                loss_g.backward()
                optimizer_g.step()

                i += 1
                print(i, end=' ')
                print(middle_time - start_time, time.time() - middle_time, time.time() - start_time)
                start_time = time.time()
            print()

        # Evaluate
        p_real, p_fake = 0., 0.
        generator.eval()
        discriminator.eval()
        for annotation_count in range(min_annotation_count, max_annotation_count + 1):
            print('Annotation count:', annotation_count)
            test_data_loader = test_data_loaders[annotation_count]
            print('Test DataLoader length:', len(test_data_loader))
            for annotations, sample in test_data_loader:
                annotations, sample = annotations.to(device), sample.to(device)
                with torch.autograd.no_grad():
                    p_real += (torch.sum(discriminator(annotations, sample)).item()) / test_annotations_count
                    z = torch.randn((batch_size, annotation_count, n-1)).to(device)
                    p_fake += (torch.sum(discriminator(generator(z, annotations))).item()) / test_annotations_count
        p_real_trace.append(p_real)
        p_fake_trace.append(p_fake)

        with open('record.pickle', 'a') as f:
            f.write(f'{g_loss} {d_loss} {p_real} {p_fake}\n')

        if (epoch + 1) % 3 == 0:
            print(f'(epoch {epoch + 1}/{epochs}) p_real: {p_real}, p_g: {p_fake}')
            torch.save(generator.state_dict(), 'generator.pth')
            torch.save(discriminator.state_dict(), 'discriminator.pth')


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.memory_summary())
    train(device)
