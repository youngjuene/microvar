import torch
import torchaudio as T
import torchaudio.transforms as TT
import librosa
import soundfile as sf
import numpy as np
import os

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

from params import params

def create_directories(output_dir, option):
    if option == 'hpss':
        os.makedirs(os.path.join(output_dir, 'harm'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'perc'), exist_ok=True)
    else: # option == 'cen'
        os.makedirs(os.path.join(output_dir, 'target'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'residue'), exist_ok=True)

def separate_spectrogram(D, sr, option, pad):
    if option == 'hpss':
        # HPSS
        mel_target, mel_residue = librosa.decompose.hpss(D)

    elif option == 'cen':
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(S=np.abs(D), sr=sr)[0]

        centroid_max = min(spectral_centroids.argmax() + pad, D.shape[0] - 1)
        centroid_min = max(spectral_centroids.argmin() - pad, 0)

        mask = np.zeros(D.shape, dtype=bool)
        mask[centroid_min:centroid_max, :] = True

        mel_target = D * mask
        mel_residue = D * ~mask

    else:
        raise ValueError('Option must be "hpss" or "cen".')

    return mel_target, mel_residue

def transform(filename, option, pad):
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1.0, 1.0)

    if params.sample_rate != sr:
        raise ValueError(f'Invalid sample rate {sr}.')
    mel_args = {
        'sample_rate': sr,
        'win_length': params.hop_samples * 4,
        'hop_length': params.hop_samples,
        'n_fft': params.n_fft,
        'f_min': 20.0,
        'f_max': sr / 2.0,
        'n_mels': params.n_mels,
        'power': 1.0,
        'normalized': True,
    }
    mel_spec_transform = TT.MelSpectrogram(**mel_args)

    with torch.no_grad():
        # Convert the audio to a mel spectrogram
        mel_spectrogram = mel_spec_transform(audio)

        # Convert the mel spectrogram back to a linear-frequency spectrogram
        D = librosa.feature.inverse.mel_to_stft(mel_spectrogram.numpy(), sr=sr, n_fft=params.n_fft, power=1.0)

        # Apply the chosen source separation method
        D_target, D_residue = separate_spectrogram(D, sr, option, pad)

        # Convert the separated sources back to mel spectrograms
        mel_target = mel_spec_transform(torch.from_numpy(librosa.istft(D_target)).float())
        mel_residue = mel_spec_transform(torch.from_numpy(librosa.istft(D_residue)).float())

        # Apply the log-amplitude transformation and scale the spectrograms
        mel_target = 20 * torch.log10(torch.clamp(mel_target, min=1e-5)) - 20
        mel_target = torch.clamp((mel_target + 100) / 100, 0.0, 1.0)
        mel_residue = 20 * torch.log10(torch.clamp(mel_residue, min=1e-5)) - 20
        mel_residue = torch.clamp((mel_residue + 100) / 100, 0.0, 1.0)

        # Save the separated spectrograms
        output_dir = f'../data/{args.audio_dir}'
        create_directories(output_dir, option)

        if option == 'hpss':
            np.save(f"{output_dir}/harm/{filename.split('/')[-1].replace('.wav', '.npy')}", mel_target.cpu().numpy())
            np.save(f"{output_dir}/perc/{filename.split('/')[-1].replace('.wav', '.npy')}", mel_residue.cpu().numpy())
            sf.write(f"{output_dir}/harm/{filename.split('/')[-1]}", librosa.istft(D_target), sr)
            sf.write(f"{output_dir}/perc/{filename.split('/')[-1]}", librosa.istft(D_residue), sr)
        else: # option == 'cen'
            np.save(f"{output_dir}/target/{filename.split('/')[-1].replace('.wav', '.npy')}", mel_target.cpu().numpy())
            np.save(f"{output_dir}/residue/{filename.split('/')[-1].replace('.wav', '.npy')}", mel_residue.cpu().numpy())
            sf.write(f"{output_dir}/target/{filename.split('/')[-1]}", librosa.istft(D_target), sr)
            sf.write(f"{output_dir}/residue/{filename.split('/')[-1]}", librosa.istft(D_residue), sr)
            

def main(args):
    filenames = glob(f'../data/{args.audio_dir}/*.wav')
    for filename in tqdm(filenames, desc='Preprocessing', total=len(filenames)):
        transform(filename, args.sep, args.pad)


if __name__ == '__main__':
    parser = ArgumentParser(description='Separates a dataset of mel spectrograms using source separation')
    parser.add_argument('--audio_dir',
                        help='Directory containing .wav files for training')
    parser.add_argument('--sep', type=str, required=True, choices=['hpss', 'cen'],
                        help='Separation method to use ("hpss" or "cen").')
    parser.add_argument('--pad', type=int, default=0,
                        help='Padding to add to the spectral centroid indices when using "cen" method.')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Uniform duration for the audio files in seconds.')
    parser.add_argument('--hop', type=float, default=0.5,
                        help='Hop size in seconds for splitting audio files into shorter segments.')
    parser.add_argument('--sr', type=int, default=22050,
                        help='Target sample rate for the audio files.')
    args = parser.parse_args()
    
    args.audio_dir = f'{args.audio_dir}_d{args.duration}_h{args.hop}_sr{args.sr}'
    
    main(args)

