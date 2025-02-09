import torch
import os
import numpy as np
from pydub import AudioSegment


# Generate initial embedding from snare drum sound resources
def create_initial_embedding(resource_root, leaves, length, max_volume):
    result = []
    # M, m = 0, 48000
    for leaf in leaves:
        directory = str(os.path.join(resource_root, leaf))
        for i in range(len(os.listdir(directory))):
            file_path = os.path.join(directory, f'{i}.wav')
            sound = AudioSegment.from_file(file_path, 'wav')
            # if i == 0: print(sound.frame_rate)  # 48000
            a = np.array(sound.get_array_of_samples())
            a = a / max_volume
            # M, m = max(M, len(a)), min(m, len(a))
            result.append(list(a) + [0] * (length - len(a)))
    # print(M, m)  # 14676 9601
    return torch.FloatTensor(result)


if __name__ == '__main__':
    embedding = create_initial_embedding('../resources/processed/single', ['strong', 'medium', 'tip'], 14676, 32768)
    print(embedding)  # max(abs(embedding)): 32767
    print(embedding.size())
    torch.save(embedding, '../resources/embedding/initial_embedding.pt')
