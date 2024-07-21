import torch
from torch.utils.data import DataLoader
from model.generator import Generator, get_embedding_layer
from preprocessing.stroke_annotation_dataset import StrokeAnnotationDataset
from pydub import AudioSegment
import matplotlib.pyplot as plt
import os
import time


def test(device):
    n = 14676
    batch_size = 2
    frame_count = 48000

    embedding_layer = get_embedding_layer('./resources/embedding').to(device)
    generator = Generator(device, batch_size, n, frame_count, embedding_layer).to(device)
    if os.path.exists('generator.pth'):
        generator.load_state_dict(torch.load('generator.pth'))

    min_annotation_count, max_annotation_count = 8, 12
    test_data_loaders = {}
    for annotation_count in range(min_annotation_count, max_annotation_count + 1):
        test_dataset = StrokeAnnotationDataset('./resources', False, 0.2, frame_count, annotation_count)
        test_data_loader = DataLoader(test_dataset, batch_size, shuffle=True, drop_last=True)
        test_data_loaders[annotation_count] = test_data_loader

    generator.eval()
    elapsed = []
    for annotation_count in range(min_annotation_count, max_annotation_count + 1):
        print('Annotation count:', annotation_count)
        test_data_loader = test_data_loaders[annotation_count]
        print('Test DataLoader length:', len(test_data_loader))
        for annotations, sample in test_data_loader:
            annotations, sample = annotations.to(device), sample.to(device)
            with torch.autograd.no_grad():
                z = torch.randn((batch_size, annotation_count, n-1)).to(device)
                print(annotations)
                start_time = time.time()
                sample_fake = generator(z, annotations)
                elapsed.append(time.time() - start_time)

                # continue
                plt.plot(list(range(frame_count)), sample[0])
                plt.show()

                plt.plot(list(range(frame_count)), sample_fake[0])
                plt.savefig('generated-sound-graph.jpg')

                sound0 = AudioSegment.from_file(f'./resources/original/multiple/0.m4a', 'm4a')
                sound_numpy = torch.round(sample_fake[0] * 32768).numpy()
                print(sound_numpy.shape, sound_numpy)
                new_sound = sound0._spawn(sound_numpy)
                new_sound.export(f'generated-sound.wav', format='wav')
                return
    print('Mean time elapsed:', sum(elapsed)/len(elapsed))


if __name__ == '__main__':
    device = torch.device("cpu")
    test(device)
