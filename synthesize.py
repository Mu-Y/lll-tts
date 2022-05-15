import sys
import os
from datetime import datetime
from tqdm import tqdm
import pickle
import numpy as np
import scipy
import torch
# import TTS model
from utils import audio, text
from utils import build_model
from params.params import Params as hp
# import wavernn vocoder
from wavernn.models.fatchord_version import WaveRNN
from wavernn.utils import hparams as hp_wavernn
from scripts.gen_wavernn import generate

def synthesize(model, input_data, force_cpu=False, mel_mean=None, mel_var=None):

    item = input_data.split('|')
    clean_text = item[1]

    if not hp.use_punctuation:
        clean_text = text.remove_punctuation(clean_text)
    if not hp.case_sensitive:
        clean_text = text.to_lower(clean_text)
    if hp.remove_multiple_wspaces:
        clean_text = text.remove_odd_whitespaces(clean_text)

    t = torch.LongTensor(text.to_sequence(clean_text, use_phonemes=hp.use_phonemes))
    text_length = t.size(0)

    # prepare language one-hot embedding
    language = item[2]
    language = torch.LongTensor([hp.languages.index(language)])
    l = language.unsqueeze(1).expand((-1, text_length)).unsqueeze(2)
    one_hots = torch.zeros(l.size(0), l.size(1), hp.language_number).zero_()
    l = one_hots.scatter_(2, l.data, 1)

    s = torch.LongTensor([hp.unique_speakers.index(item[2])]) if hp.multi_speaker else None

    if torch.cuda.is_available() and not force_cpu:
        t = t.cuda(non_blocking=True)
        if l is not None: l = l.cuda(non_blocking=True)
        if s is not None: s = s.cuda(non_blocking=True)

    s = model.inference(t, speaker=s, language=l).cpu().detach().numpy()
    if mel_mean is not None and mel_var is not None:
        hp.mel_normalize_mean = mel_mean
        hp.mel_normalize_variance = mel_var
    s = audio.denormalize_spectrogram(s, not hp.predict_linear)

    return s

if __name__ == '__main__':
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--tts_ckpt_path", type=str, required=True, help="TTS Model checkpoint.")
    parser.add_argument("--wavernn_path", type=str, required=True, help="Absolute Path to WaveRNN")
    parser.add_argument("--wavernn_ckpt_path", type=str, required=True, help="Absolute Path to WaveRNN checkpoint")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--saved_wav", type=str, default="generated.wav", help="Absolute Path to the saved wav file")
    parser.add_argument("--cpu", action='store_true', help="Force to run on CPU.")
    args = parser.parse_args()


    print("Building model ...")
    model, hp = build_model(args.tts_ckpt_path)
    model = model.eval()
    ## always use the mean and std from all10 languages
    with open("stats_per_lang_w-all10.pkl", "rb") as f:
        stats = pickle.load(f)
        hp.mel_normalize_mean = stats["all10"]["mel_normalize_mean"]
        hp.mel_normalize_variance = stats["all10"]["mel_normalize_variance"]
    pred_spec = synthesize(
               model,
               args.text,
               force_cpu=args.cpu,
               mel_mean=hp.mel_normalize_mean,
               mel_var=hp.mel_normalize_variance
           )
    # generate waveform using WaveRNN
    hp_wavernn.configure(os.path.join(args.wavernn_path, "hparams.py"))
    wavernn_model = WaveRNN(rnn_dims=hp_wavernn.voc_rnn_dims, fc_dims=hp_wavernn.voc_fc_dims, bits=hp_wavernn.bits, pad=hp_wavernn.voc_pad, upsample_factors=hp_wavernn.voc_upsample_factors,
                    feat_dims=hp_wavernn.num_mels, compute_dims=hp_wavernn.voc_compute_dims, res_out_dims=hp_wavernn.voc_res_out_dims, res_blocks=hp_wavernn.voc_res_blocks,
                    hop_length=hp_wavernn.hop_length, sample_rate=hp_wavernn.sample_rate, mode=hp_wavernn.voc_mode).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    wavernn_model.load(args.wavernn_ckpt_path)

    waveform = generate(wavernn_model, pred_spec, hp_wavernn.voc_gen_batched, hp_wavernn.voc_target, hp_wavernn.voc_overlap)
    scipy.io.wavfile.write(args.saved_wav, 22050, waveform.astype(np.float32))

