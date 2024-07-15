import array
import numpy as np
from pydub import AudioSegment


def annotate_sound(sound, min_volume, max_volume=1., patience=0.2, patience_volume=0.2):
    samples = sound.get_array_of_samples()
    max_samples = max(samples)
    annotation = []
    start_index = -1
    patience_length = 0
    for i, wave in enumerate(samples):
        wave = abs(wave) / max_samples
        if start_index == -1 and min_volume <= wave <= max_volume:
            start_index = i
        elif start_index != -1:
            end_index = i
            if wave < patience_volume:
                patience_length += 1
            elif patience_length != 0 and wave > patience_volume:
                patience_length = 0
            if patience_length > patience * sound.frame_rate:
                annotation.append([start_index, end_index])
                start_index = -1
                patience_length = 0
    return annotation


def slice_sound(samples, annotation):
    ...


if __name__ == '__main__':
    conditions = {
        'single': [
            ['strong-stroke', [0.5, 1.0]],
            ['medium-stroke', [0.5, 1.0]],
            ['strong-stroke', [0.1, 0.5]],
        ]
    }
    sound = AudioSegment.from_file('../resources/original/single/strong-stroke.m4a', 'm4a')
    samples = sound.get_array_of_samples()
    print(len(samples), max(samples))
    print(sound.frame_rate, sound.array_type)
    annotation = annotate_sound(sound, *conditions['single'][0][1])
    print(len(annotation))
    """
    shifted_samples = np.right_shift(samples, 1)  # volume /= 2
    shifted_samples_array = array.array(sound.array_type, shifted_samples)
    new_sound = sound._spawn(shifted_samples)
    print(max(new_sound.get_array_of_samples()))
    new_sound.export('../resources/original/single/strong-stroke-half.wav', format='wav')
    """