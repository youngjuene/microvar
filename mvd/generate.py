import os
import re
import argparse
from glob import glob
import torch
import librosa
import soundfile as sf
import numpy as np
from pythonosc import udp_client
from inference import predict


def select_checkpoint(ckpt_dir, epoch):
    checkpoint_files = glob.glob(f'./{ckpt_dir}/weights-*.pt')

    if epoch == -1:
        return os.path.join(ckpt_dir, 'weights.pt')
    else:
        pattern = re.compile(r'weights-(\d+).pt')
        steps = [int(pattern.match(os.path.basename(f)).group(1)) for f in checkpoint_files]
        steps_per_epoch = steps[1]
        target_steps = steps_per_epoch * epoch
        ckpt_file = f'./{ckpt_dir}/weights-{target_steps}.pt'
        if ckpt_file in checkpoint_files:
            return ckpt_file
        else:
            raise ValueError(f'No checkpoint found for {epoch} epochs.')
        

def main(args, input_audio):
    spec = librosa.feature.melspectrogram(input_audio, sr=22050, n_fft=1024, hop_length=256, n_mels=80)
    spec = torch.from_numpy(np.load(spec))
    ckpt = select_checkpoint(args.ckpt_dir, args.epoch)
    recon_audio, sample_rate = predict(spec, ckpt, fast_sampling=True)
    y = recon_audio.detach().cpu().numpy()
    sf.write(args.save_file, y, sample_rate)
    return y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select a checkpoint epoch.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Path to the checkpoint directory.')
    parser.add_argument('-i', '--wav_input', type=str, default='input.wav')
    parser.add_argument('-o', '--wav_output', type=str, default='recon.wav')
    parser.add_argument('-e', '--epoch', type=int, default=-1,
                        help='Epoch of the checkpoint to select. Set to -1 to select the best checkpoint.')
    parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=5005, help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    
    main(args, args.wav_input)

    client.send_message("/audio", args.wav_output)

