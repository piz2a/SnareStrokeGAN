import array
import os
import numpy as np
from pydub import AudioSegment


def annotate_sound1(sound, min_volume, max_volume=1., patience=0.2, patience_volume=0.2):
    samples = sound.get_array_of_samples()
    max_samples = max(samples)
    annotations = []
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
                annotations.append([start_index, end_index])
                start_index = -1
                patience_length = 0
    return annotations


def annotate_sound2(sound, min_volume, max_volume=1., patience=0.2):
    samples = sound.get_array_of_samples()
    max_samples = max(samples)
    annotations = []
    start_index = -1
    for i, wave in enumerate(samples):
        wave = abs(wave) / max_samples
        if start_index == -1 and min_volume <= wave <= max_volume:
            start_index = i
        elif start_index != -1:
            end_index = i
            if end_index - start_index > patience * sound.frame_rate:
                annotations.append([start_index, end_index])
                start_index = -1
                if min_volume <= wave <= max_volume:
                    start_index = i
    return annotations


def slice_sound(sound, annotations):
    samples = sound.get_array_of_samples()
    new_sounds = []
    for annotation in annotations:
        samples_numpy = np.array(samples)[annotation[0]:annotation[1]]
        new_samples_array = array.array(sound.array_type, samples_numpy)
        new_sound = sound._spawn(new_samples_array)
        new_sounds.append(new_sound)
    return new_sounds


def process_single(single_stroke_dir):
    for name, args in conditions_single:
        sound = AudioSegment.from_file(f'{single_stroke_dir}/{name}.m4a', 'm4a')
        samples = sound.get_array_of_samples()
        print(len(samples), max(samples))
        print(sound.frame_rate, sound.array_type)
        annotations = annotate_sound1(sound, *args)
        print(len(annotations))
        new_sounds = slice_sound(sound, annotations)
        directory = f'{single_stroke_dir}/{name}'
        if not os.path.exists(directory):
            os.mkdir(directory)
        for i, new_sound in enumerate(new_sounds):
            new_sound.export(f'{directory}/{i}.wav', format='wav')


def annotate_multiple_strokes(multiple_stroke_dir):
    for filename in os.listdir(multiple_stroke_dir):
        print(filename)
        sound = AudioSegment.from_file(f'{multiple_stroke_dir}/{filename}', 'm4a')
        annotations = annotate_sound2(sound, 0.3, 1.0, 1/12)
        with open(f'../resources/processed/annotation/{filename.split(".")[0]}.txt', 'w') as f:
            f.write(f'{len(sound.get_array_of_samples())}\n')
            f.write("\n".join(f'{annotation[0]} {annotation[1]}' for annotation in annotations))


if __name__ == '__main__':
    conditions_single = [
        ['strong-stroke', [0.5, 1.0]],
        ['medium-stroke', [0.5, 1.0]],
        ['tip-and-strong', [0.1, 0.5]],
    ]

    # Slice sounds
    process_single('../resources/original/single')

    # Annotate sounds
    annotate_multiple_strokes('../resources/original/multiple')
