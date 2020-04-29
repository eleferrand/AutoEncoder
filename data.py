import os
import scipy.io.wavfile as wav
import json
import numpy as np
from pydub import AudioSegment
import _pickle as pickle

import torch
from torch import nn
from fairseq.models.wav2vec import Wav2VecModel
import soundfile as sf

from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.features.processor.plp import PlpProcessor
from shennong.features.postprocessor.cmvn import CmvnPostProcessor
from shennong.audio import Audio

class PretrainedWav2VecModel(nn.Module):

    def __init__(self, fname):
        super().__init__()

        checkpoint = torch.load(fname)
        self.args = checkpoint["args"]
        model = Wav2VecModel.build_model(self.args, None)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        self.model = model

    def forward(self, x):
        with torch.no_grad():
            z = self.model.feature_extractor(x)
            if isinstance(z, tuple):
                z = z[0]
            c = self.model.feature_aggregator(z)
        return z, c

class Prediction():
    """ Lightweight wrapper around a fairspeech embedding model """

    def __init__(self, fname, gpu=0):
        self.gpu = gpu
        self.model = PretrainedWav2VecModel(fname).cuda(gpu)

    def __call__(self, x):
        x = torch.from_numpy(x).float().cuda(self.gpu)
        with torch.no_grad():
            z, c = self.model(x.unsqueeze(0))

        return z.squeeze(0).cpu().numpy(), c.squeeze(0).cpu().numpy()
def read_audio(fname):
    """ Load an audio file and return PCM along with the sample rate """

    wav, sr = sf.read(fname)
    assert sr == 16e3

    return wav, 16e3

def get_mfcc_dd(wav_fn, norm="cmvn"):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    audio = Audio.load(wav_fn)
    processor = MfccProcessor(sample_rate=audio.sample_rate, window_type="hamming",frame_length=0.025, frame_shift=0.01,
                              cepstral_lifter=26.0,low_freq=0, vtln_low=60, vtln_high=7200, high_freq=audio.sample_rate/2)
    d_processor = DeltaPostProcessor(order=2)
    mfcc_static = processor.process(audio, vtln_warp=1.0)
    mfcc_deltas = d_processor.process(mfcc_static)
    features = np.float64(mfcc_deltas._to_dict()["data"])

    if norm=="cmvn":
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    return features

def get_plp_dd(wav_fn, norm):
    """Return the MFCCs with deltas and delta-deltas for a audio file."""
    audio = Audio.load(wav_fn)
    processor = PlpProcessor(sample_rate=audio.sample_rate, window_type="hamming",frame_length=0.025, frame_shift=0.01,
                              low_freq=0, vtln_low=60, vtln_high=7200, high_freq=audio.sample_rate/2)
    plp_static = processor.process(audio, vtln_warp=1.0)
    d_processor = DeltaPostProcessor(order=2)
    plp_deltas = d_processor.process(plp_static)
    features = np.float64(plp_deltas._to_dict()["data"])
    if norm == "cmvn":
        features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features

def get_wtv(wav_fn, norm, model):
    signal, sr = read_audio(wav_fn)
    
    z, c = model(signal)
    if norm =="cmvn":
        p = (c - np.mean(c, axis=0)) / np.std(c, axis=0)
        sortie = (np.float64(p)).transpose()
        sortie = sortie.copy(order='C')
    else:
        sortie = (np.float64(c)).transpose()
        sortie = sortie.copy(order='C')
    return sortie
def load_utt(part, rep, norm):
    if part == 'valid' or part == 'val':
        part='dev'
    root = '/home/getalp/leferrae/thesis/corpora/mboshi-french-parallel-corpus/full_corpus_newsplit/{}/'.format(part)
    tout = '/home/getalp/leferrae/thesis/corpora/mboshi-french-parallel-corpus/full_corpus_newsplit/all/'
    label=0
    x = []
    keys = []
    labels = []
    train_lengths = []
    if rep == "wtv":
        model_wtv = Prediction("/home/getalp/leferrae/thesis/mb_model/checkpoint_best.pt", 0)
    for wav_fn in os.listdir(root):
        if '.wav' in wav_fn:
            print("loading {}".format(wav_fn))
            if rep == "mfcc":
                feat = get_mfcc_dd(tout+wav_fn, norm=norm)
            elif rep == "plp":
                feat = get_plp_dd(tout+wav_fn, norm=norm)
            elif rep == "wtv":
                feat = get_wtv(tout+wav_fn, norm=norm, model=model_wtv)
            x.append(feat)
            keys.append(label)
            labels.append(label)
            train_lengths.append(feat.shape[0])
            label+=1
    with open("{}_{}.pkl".format(part, rep), mode="wb") as pfile:
        pickle.dump([x, labels, train_lengths, keys], pfile)
    return x, labels, train_lengths, keys