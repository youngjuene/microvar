### Micro-variation of Sound Objects Using Component Separation and Diffusion Models
Official PyTorch implementation of **Micro-variation of Sound Objects Using Component Separation and Diffusion Models** (ICMC 2023).

### Requirements
1. Create conda environment
```
conda create -n microvar python=3.8 -y
conda activate microvar
conda env update -f environment.yaml
```

### Demo Using Pretrained Model
1. Sample generation
```
wget https://zenodo.org/record/00000/files/mvd.tar.gz
tar -zxvf mvd.tar.gz 
```

Please refer to [notebook/demo.ipynb](https://github.com/youngjuene/microvar/notebook/demo.ipynb) for [FSD50k](https://github.com/youngjuene/microvar/notebook/samples) subsets. Below is the code instruction.

2. Application with Max/MSP and Unreal Engine
Download the files [Link](https://github.com/youngjune/microvar)


### License
This project is under the CC-BY-NC 4.0 license. See LICENSE for details.

### Citation
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follow.
```
@article{micro2023liu,
  title={Micro-variation of Sound Objects Using Component Separation and Diffusion Models},
  author={Philip Liu, Youngjun Choi, Jinjoon Lee},
  journal={International Computer Music Conference},
  year={2023}
}
```


### Reference
[DiffWave: A Versatile Diffusion Model for Audio Synthesis](https://github.com/lmnt-com/diffwave)
