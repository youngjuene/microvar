import pdb
import os
import glob
import re
import argparse
import torch
import torchaudio as T
import torchaudio.transforms as TT
import librosa
import soundfile as sf
import numpy as np
from params import params
from pythonosc import udp_client
from inference import predict
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server

#Python-osc client
def run(address, content):
  print(address)
  print(content)

#####

def audio_to_mel(audio, sr):
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
    mel_spectrogram = mel_spec_transform(audio)
    return mel_spectrogram


def select_checkpoint(args):
    if args.ckpt_path is not None:
        return args.ckpt_path
    
    checkpoint_files = glob.glob(f'./{args.ckpt_dir}/weights-*.pt')
    if args.epoch == -1:
        return os.path.join(args.ckpt_dir, 'weights.pt')
    else:
        pattern = re.compile(r'weights-(\d+).pt')
        steps = [int(pattern.match(os.path.basename(f)).group(1)) for f in checkpoint_files]
        steps_per_epoch = steps[1]
        target_steps = steps_per_epoch * args.epoch
        ckpt_file = f'./{args.ckpt_dir}/weights-{target_steps}.pt'
        if ckpt_file in checkpoint_files:
            return ckpt_file
        else:
            raise ValueError(f'No checkpoint found for {args.epoch} epochs.')
        

def main(args):
    audio, sr = T.load(args.wav_input)
    audio = torch.clamp(audio[0], -1.0, 1.0)

    with torch.no_grad():
        mel = audio_to_mel(audio, sr)
        
    ckpt = select_checkpoint(args)
    recon_audio, sample_rate = predict(mel, ckpt, fast_sampling=True)
    recon_audio = recon_audio.detach().cpu().numpy()
    recon_audio = recon_audio.flatten()
    audio = audio.detach().cpu().numpy()

    if recon_audio.size > audio.size:
        recon_audio = recon_audio[:audio.size]
    elif recon_audio.size < audio.size:
        audio = audio[:recon_audio.size]
    added_audio = audio + recon_audio
    sf.write(args.wav_output, added_audio, sample_rate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select a checkpoint epoch.')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='Path to the checkpoint file.')
    parser.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                        help='Path to the checkpoint directory.')
    parser.add_argument('-i', '--wav_input', type=str, default='input.wav')
    parser.add_argument('-o', '--wav_output', type=str, default='output.wav')
    parser.add_argument('-e', '--epoch', type=int, default=-1,
                        help='Epoch of the checkpoint to select. Set to -1 to select the best checkpoint.')
    parser.add_argument("--ip", default="127.0.0.1", help="The ip of the OSC server")
    parser.add_argument("--port", type=int, default=9857, help="The port the OSC server is listening on")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    
    main(args)

    client.send_message("/audio", args.wav_output)

    dispatcher = Dispatcher()
    dispatcher.map("/run", run)

    server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()