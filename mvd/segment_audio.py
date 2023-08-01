import os
import numpy as np
import soundfile as sf
import librosa

from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from tqdm import tqdm

def clean_filename(filename):
    return filename.replace(' ', '_').replace('(', '').replace(')', '')

def segment_audio(filename, output_dir, duration, hop, sr_target):
    audio, sr = sf.read(filename)

    # Handle mono and stereo audio separately
    if len(audio.shape) == 1:
        # Mono audio
        if sr != sr_target:
            audio = librosa.resample(audio, sr, sr_target)
            sr = sr_target
    else:
        # Stereo audio
        if sr != sr_target:
            audio = np.stack([librosa.resample(channel, sr, sr_target) for channel in audio.T])
            sr = sr_target

    duration_samples = int(duration * sr)
    hop_samples = int(hop * sr)

    # Calculate number of segments that will fit into the audio file
    num_segments = 1 + (len(audio) - duration_samples) // hop_samples

    # Split the audio into shorter segments
    for i in range(num_segments):
        start = i * hop_samples
        segment = audio[start:start + duration_samples]
        output_filename = os.path.join(output_dir, f'{clean_filename(os.path.splitext(os.path.basename(filename))[0])}_{i}.wav')
        sf.write(output_filename, segment, sr)


def main(args):
    output_dir = f'../data/{args.audio_dir}_d{args.duration}_h{args.hop}_sr{args.sample_rate}'
    os.makedirs(output_dir, exist_ok=True)

    filenames = glob(f'../data/{args.audio_dir}/*.wav')

    for filename in tqdm(filenames, desc='Segmenting', total=len(filenames)):
        segment_audio(filename, output_dir, args.duration, args.hop, args.sample_rate)


if __name__ == '__main__':
    parser = ArgumentParser(description='Segments audio files into uniform duration')
    parser.add_argument('--audio_dir',
                        help='Directory containing .wav files for training')
    parser.add_argument('--duration', type=float, default=1.0,
                        help='Uniform duration for the audio files in seconds.')
    parser.add_argument('--hop', type=float, default=0.5,
                        help='Hop size in seconds for splitting audio files into shorter segments.')
    parser.add_argument('--sample_rate', type=int, default=22050,
                        help='Target sample rate for audio files. Audio will be resampled if necessary.')
    main(parser.parse_args())
