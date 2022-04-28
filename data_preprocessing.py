import os
import librosa as lr
from librosa.display import waveplot
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import imageio
import warnings
from IPython.display import Audio

train = 'train'
test = 'test'

ar = 'arabic'
en = 'english'
de = 'german'

languages = [ar, en, de]
categories = [train, test]

dataset_root_path = 'test'

sample_rate = 8000
image_width = 500
image_height = 128


def load_audio_file(audio_file_path):
    warnings.simplefilter('ignore', UserWarning)
    try:
        audio_segment, _ = lr.load(audio_file_path, sr=sample_rate)
    except Exception:
        return None
    else:
        return audio_segment

    warnings.simplefilter('default', UserWarning)


def fix_audio_segment_to_10_seconds(audio_segment):
    target_len = 10 * sample_rate
    audio_segment = np.concatenate([audio_segment] * 3, axis=0)
    audio_segment = audio_segment[0:target_len]

    return audio_segment


def spectrogram(audio_segment):
    # Compute mel-scaled spectrogram image
    hl = audio_segment.shape[0] // image_width
    print(audio_segment.shape[0])
    print(hl)
    print(image_width)
    spec = lr.feature.melspectrogram(audio_segment, n_mels=image_height, hop_length=int(hl))

    # Logarithmic amplitudes
    image = lr.core.power_to_db(spec)

    # Convert to numpy matrix
    image_np = np.asmatrix(image)

    # Normalize and scale
    image_np_scaled_temp = (image_np - np.min(image_np))

    image_np_scaled = image_np_scaled_temp / np.max(image_np_scaled_temp)

    return image_np_scaled[:, 0:image_width]


def to_integer(image_float):
    # range (0,1) -> (0,255)
    image_float_255 = image_float * 255.

    # Convert to uint8 in range [0:255]
    image_int = image_float_255.astype(np.uint8)

    return image_int


# def audio_to_image_file(audio_file):
#     out_image_file = audio_file + '.png'
#     audio = load_audio_file(audio_file)
#     audio_fixed = fix_audio_segment_to_10_seconds(audio)
#     if np.count_nonzero(audio_fixed) != 0:
#         spectro = spectrogram(audio_fixed)
#         spectro_int = to_integer(spectro)
#         imageio.imwrite(out_image_file, spectro_int)
#     else:
#         print('WARNING! Detected an empty audio signal. Skipping...')


def loading_audio(folder_path):
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(folder_path)):
        for j, (filename) in enumerate(filenames):
            file_path = os.path.join(dirpath, filename)
            print(file_path)
            audio = load_audio_file(file_path)
            if audio is not None:
                audio_fixed = fix_audio_segment_to_10_seconds(audio)
                if np.count_nonzero(audio_fixed) != 0:
                    spectro = spectrogram(audio_fixed)
                    spectro_int = to_integer(spectro)
                    plt.imshow(spectro_int)
                    plt.show()
                    imageio.imwrite(f'test_spect/spectogram{j}.png', spectro_int)
            else:
                print('WARNING! Detected an empty audio signal. Skipping...')

loading_audio(dataset_root_path)