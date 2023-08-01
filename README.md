### Micro-variation of Sound Objects Using Component Separation and Diffusion Models
Official PyTorch implementation of **Micro-variation of Sound Objects Using Component Separation and Diffusion Models** (ICMC 2023).

### Requirements
1. Create conda environment
```
conda create -n microvar python=3.8 -y
conda activate microvar
conda env update -f environment.yaml
```

2. Place the desired audio dataset in `data` directory and preprocess it as follows
```
cd mvd
python segment_audio.py --audio_dir {directory of original dataset}
python preprocess.py --audio_dir {directory with segmented audio files} --sep {separation options}
```

3. Train the model on custom datasets
Run the train.py script to train the model. Replace {model name} with the desired name for your model, and {directory with preprocessed audio files} with the path to the preprocessed audio files.
```
python train.py --model_dir {model name} --data_dirs {directory with preprocessed audio files}
```

4. Generate samples using the model checkpoints
Run the generate.py script to generate samples using the trained model. Replace {model name} with the name of directory where your trained model is located, {input audio} with the name of the input audio file and {output audio filename} with the desired name for the output audio file.
```
python generate.py --ckpt_dir {ckpt dir} -i {input audio} -o {output audio filename}
```

### Demo Using Pretrained Models
1. Download model checkpoints (WIP)
```
wget https://zenodo.org/record/00000/files/mvd.tar.gz
tar -zxvf mvd.tar.gz 
```
2. Run generate.py by specifying the pretrained checkpoint
```
python generate.py --ckpt_path {pretrained ckpt} -i {input audio} -o {output audio filename}
```

3. Try out other examples on Max/MSP and Unreal Engine
<br> Download the project files [Link](https://github.com/youngjune/microvar)


### License
This project is under the CC-BY-NC 4.0 license. See LICENSE for details.

### Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
```
@article{micro2023liu,
  title={Micro-variation of Sound Objects Using Component Separation and Diffusion Models},
  author={a, b, c},
  proceedings={International Computer Music Conference},
  year={2023}
}
```


### Reference
[DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://github.com/lmnt-com/diffwave)
