# lll-tts
Code for our ICASSP 2022 paper: [TOWARDS LIFELONG LEARNING OF MULTILINGUAL TEXT-TO-SPEECH SYNTHESIS](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746968). Audio demo is available [here](https://mu-y.github.io/speech_samples/llltts/).

## Install dependencies
Install a Pytorch version that is compatible to your CUDA setups. Then
```
pip install -r requirements.txt 
```
The baselines alrogithms require other dependencies. Please look at the corresponding repos ([EWC](https://github.com/moskomule/ewc.pytorch), [GEM](https://github.com/facebookresearch/GradientEpisodicMemory)) to install them.

## Data preparation
1. Download CSS10 corpus [here](https://github.com/Kyubyong/css10). In this work, we only use four languages: German, Dutch, Chinese, Japanese.
2. Move the downloaded archives to `data/css10`, and unzip the archives.
3. Run the following command to prepare the spectrograms:
  ```
  cd data/
  python prepare_css_spectrograms.py
  ```
  
## Training
We form a language sequence of German -- Dutch -- Chinese -- Japanese.
The following commands assume that we have a model checkpoint trained on German, and we want to continually and sequentially train the model on the following three languages. We provide a pre-trained German checkpoint [here](https://drive.google.com/drive/folders/1t9FBu3L4KzQ901J_0hKUZ5mFDFfRmGQ1?usp=sharing).

Train the proposed data replay scheme with the dual-sampling strategy.
```
python train_continual.py --hyper_parameters dual_samp.json --save_dir exp_dual --checkpoint checkpoints_ge/checkpoints/SHARED-TRAINING_loss-99-0.147
```
The other sampling strategies in the paper, i.e. Random Sampling and Weighted sampling are also available. Simply change the `hyper_parameters` argument to the corresponding json config file under `params/`

Train baseline, [Elastic Weight Consolidation (EWC)](https://arxiv.org/abs/1612.00796).
```
python train_continual.py --hyper_parameters ewc.json --save_dir exp_ewc --checkpoint checkpoints_ge/checkpoints/SHARED-TRAINING_loss-99-0.147-fisher
```

Train baseline, [Gradient Episodic Memory (GEM)](https://arxiv.org/abs/1706.08840).
```
python train_continual.py --hyper_parameters gem.json --save_dir exp_gem --checkpoint checkpoints_ge/checkpoints/SHARED-TRAINING_loss-99-0.147
```

Train the lower-bound approach (fine-tune).
```
python train_continual.py --hyper_parameters finetune.json --save_dir exp_finetune --checkpoint checkpoints_ge/checkpoints/SHARED-TRAINING_loss-99-0.147
```

The upper-bound approach is to train a normal TTS model on all 4 languages jointly, not sequentially.
```
python train_joint.py --hyper_parameters joint.json --save_dir exp_joint
```

## Inference
We use WaveRNN for vocoder. To generate waveforms, first download and setup the WaveRNN vocoder:
```
git clone https://github.com/Tomiinek/WaveRNN
# download the WaveRNN checkpoint trained on the entire CSS10 corpus
cd WaveRNN/
mkdir -p checkpoints && cd checkpoints
curl -O -L "https://github.com/Tomiinek/Multilingual_Text_to_Speech/releases/download/v1.0/wavernn_weight.pyt" 
```
Then append the path to the `WaveRNN` folder to the `PYTHONPATH` variable:
```
export PYTHONPATH=$PYTHONPATH:<absolute path to the WaveRNN dir>
```
Finally, use the following command to generate a waveform:
```
# at the root of this repo
python synthesize.py --tts_ckpt_path release_checkpoints/dual_samp/SHARED-TRAINING_loss-299-0.108 \
                     --wavernn_path <absolute path to the WaveRNN dir> \
                     --wavernn_ckpt_path <absolute path to the pre-trained WaveRNN checkpoint> \
                     --text "|fēi tè xìucái yīnwèi shàngchéng qù bàoguān， bèi bùhǎo de gémìngdǎng jiǎn le biànzǐ， érqiě yòu pòfèi le èrshí qiān de shǎngqián， suǒyǐ quánjiā yě hàotáo le。|chinese|chinese" \
                     --saved_wav generated.wav
```
The above command shows an example of synthesizing Chinese speech, using the model trained by our proposed dual-sampler method after the entire training sequence finished. We release a few trained models by different approaches. Models can be downloaded [here](https://drive.google.com/drive/folders/17a1d8Dgy2aGO4wNhRkzcGi_8cmWUclf8?usp=sharing).


## Acknowledgements
- The TTS implemenation is adapted from https://github.com/Tomiinek/Multilingual_Text_to_Speech.
- The lifelong learning baselines are adapted from the following repos:
  - EWC: https://github.com/moskomule/ewc.pytorch
  - GEM: https://github.com/facebookresearch/GradientEpisodicMemory

## Citation
Cite our paper!
```
@INPROCEEDINGS{9746968,
  author={Yang, Mu and Ding, Shaojin and Chen, Tianlong and Wang, Tong and Wang, Zhangyang},
  booktitle={ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Towards Lifelong Learning of Multilingual Text-to-Speech Synthesis}, 
  year={2022},
  volume={},
  number={},
  pages={8022-8026},
  doi={10.1109/ICASSP43922.2022.9746968}}
```
