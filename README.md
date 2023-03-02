# TriAAN-VC: Triple Adaptive Attention Normalization for any-to-any Voice Conversion (ICASSP 2023)

This is a Pytorch implementation of [TriAAN-VC: Triple Adaptive Attention Normalization for any-to-any Voice Conversion](https://arxiv.org/abs/2203.02181). TriAAN-VC is a deep learning model for any-to-any voice conversion. TriAAN-VC can maintain the linguistic contents of source speech and represent target characteristics, unlike previous methods. Experimental results on the VCTK dataset suggest that TriAAN-VC achieves state-of-the-art performance. 
We strongly recommend you to visit our [demo site](https://winddori2002.github.io/vc-demo.github.io/).

The overall architecture of TriAAN-VC is as below:

<p align="center">
	<img src="./img/triaan_vc.png" alt="TriAAN-VC" width="90%" height="90%"/>
</p>

# Installation & Enviornment

The OS, Python, and PyTorch version are as below (You can also use other versions.):
- Windows
- Linux
- python == 3.8
- pytorch == 1.9.1
- torchaudio == 0.9.1

You can install requirements through git and requirements.txt except for pytorch and torchaudio.
```C
git clone https://github.com/winddori2002/TriAAN-VC.git
cd TriAAN-VC
pip install -r requirements.txt
```

# Prepare for usage

## 1. Prepare dataset

We use the VCTK dataset consisting of 110 speakers with 400 utterances per speaker.

- The dataset can be downloaded [here](https://datashare.ed.ac.uk/handle/10283/3443).
- We divide the dataset depending on seen-to-seen and unseen-to-unseen scenarios for evaluation.

## 2. Prepare pre-trained vocoder and feature extractor

We use the pre-trained ParallelWaveGAN as vocoder and CPC extractor as feature extractor. 
You can use the pre-trained weights in this repository. 
The vocoder is trained on the VCTK dataset and CPC extractor is trained on the LibriSpeech dataset.

- This repository provides pre-trained ParallelWaveGAN provided by [here](https://github.com/Wendison/VQMIVC) and CPC extractor provided by [here](https://github.com/facebookresearch/CPC_audio).
- Or you can train [ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) and [CPC](https://github.com/facebookresearch/CPC_audio).

## 3. Preprocess

The preprocess stages contain dataset split, feature extraction, making data paths, and eval pairs.

The steps are for VCTK dataset and if you want to use other dataset, you need to modify the details.

- To split dataset and get mel-spectrograms, lf0, and metadata, run the following code and modify the directories in the ```./config/preprocess.yaml```.

```
python preprocess.py
```

- To get CPC features and represent the paths on the metadata, run the following code. (You need pre-trained weights for CPC)
```
python preprocess_cpc.py
```

- To get evaluation pairs for conversion, run the following code.
```
python generate_eval_pair.py
```

The dataset split and eval pairs are for evaluation and investigation, they are not actually necessary to train models.


# How to use

## 1. Train

### Training with default settings

You can train MANNER with the default setting by running the following code.

```
python main.py train --aug True --aug_type tempo
```

### Training with other arguments
If you want to edit model settings, you can run the following code with other arguments. 

In ```config.py```, you can find other arguments, such as batch size, epoch, and so on.

```
python main.py train --hidden 60 --depth 4 --growth 2 --kernel_size 8 --stride 4 --segment_len 64 --aug True --aug_type tempo

MANNER arguments:
  --in_channels : initial in channel size (default:1)
  --out_channels: initial out channel size (default:1)
  --hidden      : channel size to expand (default:60)
  --depth       : number of layers for encoder and decoder (default:4)
  --kernel_size : kernel size for UP/DOWN conv (default:8)
  --stride      : stride for UP/DOWN conv (default:4)
  --growth      : channel expansion ration (default:2)
  --head        : number of head for global attention (default:1)
  --segment_len : chunk size for overlapped chunking in a dual-path processing (default:64)
  
Setting arguments:
  --sample_rate: sample_rate (default:16000)
  --segment    : segment the audio signal with seconds (default:4)
  --set_stride : Overlapped seconds when segment the signal (default:1)
  
Augmentation arguments:
  --aug     : True/False 
  --aug_type: augmentation type (tempo, speed, shift available. only shift available on Windows.)
```

### Training with logging

The logs are uploaded on [neptune.ai](https://neptune.ai/)
```
python main.py train --logging True

Logging arguments:
  --logging    : True/False
```

## 2. evaluation

After training, you can evaluate the model in terms of PESQ and STOI by running the code below.
You need to keep the model arguments in the training phase.
```
python main.py test --save_enhanced True --enhanced_path []

evaluation arguments:
  --save_enhanced: saving enhanced audio file
  --enhanced_path: enhanced file directory
```

If you want to evaluate with all measures (PESQ, STOI, CSIG, CBAK, COVL), run the following code.
```
python eval_measure.py

clean_path    = 'test clean path'
enhanced_path = 'enhanced path'
```


## 3. Pretrained weights

The pretrained weights of MANNER is uploaded on the github release [here](https://github.com/winddori2002/MANNER/releases/tag/v1.0)


## 4. MANNER small

For MANNER (small), it is necessary to change the code in ```./model.py``` since it has a different structure from MANNER (base). The code for MANNER (small) is commented below the line for MANNER in  ```./model.py```.

## 5. Custom enhancement

For custom enhancement, you can estimate enhanced speech by running code with ```custom_enhance.py```. 
The codes include input data processing (downsample from 48 kHz to 16 kHz).
```
python custom_enhance.py

enhance arguments:
  --device: Cuda device or CPU
  --noisy_path: Path (folder) which contains noisy wav files
  --model_name: Model version (small, base, large) - you can use one of them
```


## 6. Enhanced samples

We provide randomly selected audio sample pairs in ```./samples```.  Each pair is composed of enhanced speech and noisy speech. We tried to choose some pairs containing more noises.

# Experimental Results

The experimental results are from the provided pre-trained weights, and the results can be slightly different from the paper.
We divide versions into "VCTK Split" and "VCTK All". "VCTK Split" indicates the pre-trained weight with dataset split as in the paper.
"VCTK All" means the pre-trained weight using all VCTK dataset. \# uttr indicates the number of target utterances used for conversion.


Below, the results are summarized with the "TriAAN-VC-Split". Each score is the average of seen-to-seen and unseen-to-unseen scenarios.

|Model|Pre-trained Ver.|\# uttr|WER AVG (\%)|CER AVG (\%)|SV AVG (\%)|
|:---|:---:|:---:|:---:|:---:|:---:|
|TriAAN-VC-Mel|VCTK Split|1|27.61|14.78|89.42|
|TriAAN-VC-Mel|VCTK Split|3|22.86|12.15|95.92|
|TriAAN-VC-CPC|VCTK Split|1|21.50|11.24|92.33|
|TriAAN-VC-CPC|VCTK Split|3|17.42|8.86|97.75|

Below, the results are summarized with the "TriAAN-VC-All". Since the model was trained on the whole VCTK dataset regardless of split setting, the score is only about seen-to-seen scenarios.

|Model|Pre-trained Ver.|\# uttr|WER AVG (\%)|CER AVG (\%)|SV AVG (\%)|
|:---|:---:|:---:|:---:|:---:|:---:|
|TriAAN-VC-Mel|VCTK All|1||||
|TriAAN-VC-Mel|VCTK All|3||||
|TriAAN-VC-CPC|VCTK All|1||||
|TriAAN-VC-CPC|VCTK All|3||||

## Citation

```
@inproceedings{park2022manner,
  title={MANNER: Multi-View Attention Network For Noise Erasure},
  author={Park, Hyun Joon and Kang, Byung Ha and Shin, Wooseok and Kim, Jin Sob and Han, Sung Won},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={7842--7846},
  year={2022},
  organization={IEEE}
}
```

## License

This repository is released under the MIT license.

The file ```src/dataset.py``` was adapted from the [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser), released under the CC-BY-NC 4.0. license. We modified the speaker selection in the dataset. The ```src/metrics.py``` was adapted from the [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) and [santi-pdp/segan_pytorch](https://github.com/santi-pdp/segan_pytorch), where latter was released under the MIT license. The ```src/stft_loss.py``` was adapted from the [facebookresearch/denoiser](https://github.com/facebookresearch/denoiser) and [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN), where latter was released under the MIT license.
