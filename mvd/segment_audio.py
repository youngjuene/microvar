import os
import torch
import torchaudio as T

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm


def segment_audio(filename, output_dir, duration, hop):
    audio, sr = T.load(filename)
    audio = torch.clamp(audio[0], -1.0, 1.0)

    duration_samples = int(duration * sr)
    hop_samples = int(hop * sr)

    # Split the audio into shorter segments
    for i in range(0, len(audio) - duration_samples, hop_samples):
        segment = audio[i:i + duration_samples]
        output_filename = os.path.join(output_dir, f'{os.path.splitext(os.path.basename(filename))[0]}_{i}.wav')
        T.save(output_filename, segment, sr)


def main(args):
    filenames = glob(f'{args.dir}/**/*.wav', recursive=True)
    os.makedirs(args.output_dir, exist_ok=True)
    with ProcessPoolExecutor() as executor:
        list(tqdm(executor.map(lambda filename: segment_audio(filename, args.output_dir, args.duration, args.hop), filenames), desc='Segmenting', total=len(filenames)))


if __name__ == '__main__':
    parser = ArgumentParser(description='Segments audio files into uniform duration')
    parser.add_argument('--dir',
                        help='Directory containing .wav files for training')
    parser.add_argument('--output_dir', type=str, default='uniform_duration',
                        help='Output directory for segmented audio files.')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Uniform duration for the audio files in seconds.')
    parser.add_argument('--hop', type=float, default=0.5,
                        help='Hop size in seconds for splitting audio files into shorter segments.')
    main(parser.parse_args())
