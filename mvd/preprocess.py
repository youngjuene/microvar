import numpy as np
import torch
import torchaudio as T
import torchaudio.transforms as TT
import librosa

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params


def preprocess_audio(y, sr, option, pad):
    # STFT
    D = librosa.stft(y, n_fft=params.n_fft, hop_length=params.hop_samples, win_length=params.hop_samples * 4)

    if option == 'hpss':
        # HPSS
        H, P = librosa.decompose.hpss(D)
        return H, P
    elif option == 'cen':
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y, sr=sr)[0]

        centroid_max = min(spectral_centroids.argmax() + pad, len(spectral_centroids) - 1)
        centroid_min = max(spectral_centroids.argmin() - pad, 0)

        mask = np.zeros(D.shape, dtype=bool)
        mask[centroid_min:centroid_max, :] = True

        target = D * mask
        residue = D * ~mask

        return target, residue
    else:
        raise ValueError('Option must be "hpss" or "cen".')

def transform(filename, option, pad):
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1.0, 1.0)

    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')

    with torch.no_grad():
        # Apply the preprocessing
        audio1, audio2 = preprocess_audio(audio.numpy(), sr, option, pad)
        audio1 = torch.from_numpy(audio1).float()
        audio2 = torch.from_numpy(audio2).float()

        mel_spec_transform = TT.MelSpectrogram(sample_rate=sr, win_length=params.hop_samples * 4, hop_length=params.hop_samples, n_fft=params.n_fft, f_min=20.0, f_max=sr / 2.0, n_mels=params.n_mels, power=1.0, normalized=True)
        spectrogram1 = mel_spec_transform(audio1)
        spectrogram2 = mel_spec_transform(audio2)

        spectrogram1 = 20 * torch.log10(torch.clamp(spectrogram1, min=1e-5)) - 20
        spectrogram1 = torch.clamp((spectrogram1 + 100) / 100, 0.0, 1.0)
        spectrogram2 = 20 * torch.log10(torch.clamp(spectrogram2, min=1e-5)) - 20
        spectrogram2 = torch.clamp((spectrogram2 + 100) / 100, 0.0, 1.0)

        np.save(f'{filename}_part1.spec.npy', spectrogram1.cpu().numpy())
        np.save(f'{filename}_part2.spec.npy', spectrogram2.cpu().numpy())


def main(args):
    filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(lambda filename: transform(filename, args.sep, args.pad), filenames), desc='Preprocessing', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='Prepares a dataset to train DiffWave')
    parser.add_argument('--dir',
                        help='Directory containing .wav files for training')
    parser.add_argument('--sep', type=str, required=True, choices=['hpss', 'cen'],
                        help='Separation method to use ("hpss" or "cen").')
    parser.add_argument('--pad', type=int, default=0,
                        help='Padding to add to the spectral centroid indices when using "cen" method.')
    main(parser.parse_args())
