from .soundslicer import *
from torch.utils.data import Dataset, DataLoader
import pickle
import torch


class StrokeAnnotationDataset(Dataset):
    def __init__(self, resource_dir, train: bool, ratio: float, frame_count: int, annotation_count: int, save_filename='dataset.pickle'):
        self.resource_dir = resource_dir
        self.frame_count = frame_count
        self.annotation_count = annotation_count

        self.dataset_full = None

        dataset_path = f'{resource_dir}/{save_filename}'
        if os.path.exists(dataset_path):
            with open(dataset_path, 'rb') as f:
                self.dataset_full = pickle.load(f)  # Load from existing file
        else:
            self.dataset_full = self.create_dataset()  # Create new
            with open(dataset_path, 'wb') as f:
                pickle.dump(self.dataset_full, f)  # and save

        # Crop dataset
        # print(sum(len(self.dataset_full[i]) for i in self.dataset_full))  # 2209
        # print(self.dataset_full.keys())  # 1 ~ 12
        self.dataset = self.dataset_full[self.annotation_count]
        edge_index = round(len(self.dataset) * ratio)
        self.dataset = self.dataset[:edge_index] if train else self.dataset[-edge_index:]

    def create_dataset(self):  # with Augmentation / Certain count of annotation / sound sample length: always 1s
        directory = os.listdir(f'{self.resource_dir}/original/multiple')
        result = {}
        for filename in directory:
            file_num = int(filename.split(".")[0])
            lines = open(f'{self.resource_dir}/processed/annotation/{file_num}.txt').read().split('\n')
            file_frame_count, annotations = int(lines[0]), lines[1:]
            annotations = [list(map(int, annotation.split(' '))) for annotation in annotations]
            annotation_count = len(annotations)  # min 31 max 369
            for i, annotation in enumerate(annotations):
                annotations_fragment = []
                if file_frame_count - annotation[0] < self.frame_count * 1:
                    continue
                while i < annotation_count and annotations[i][0] - annotation[0] < self.frame_count * 1:  # within 1s
                    annotations_fragment.append(annotations[i][0])
                    i += 1
                annotations_fragment_count = len(annotations_fragment)
                if annotations_fragment_count not in result:
                    result[annotations_fragment_count] = []
                result[annotations_fragment_count].append((file_num, torch.LongTensor(annotations_fragment)))
        return result

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        file_num, annotations = self.dataset[index]
        sound = AudioSegment.from_file(f'{self.resource_dir}/original/multiple/{file_num}.m4a', 'm4a')
        samples = np.array(sound.get_array_of_samples())
        sample_cropped = samples[annotations[0]:annotations[0]+self.frame_count]
        # print(annotations, sample_cropped.shape)
        return annotations, sample_cropped


if __name__ == '__main__':
    train_dataset = StrokeAnnotationDataset('../resources', train=True, ratio=0.8, frame_count=48000, annotation_count=9)
    test_dataset = StrokeAnnotationDataset('../resources', train=False, ratio=0.2, frame_count=48000, annotation_count=6)

    batch_size = 4
    train_data_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size, shuffle=True)

    print(len(train_data_loader), len(test_data_loader))

    for x, y in train_data_loader:
        print('x:', x.size())
        print('y:', y.size())
        break
