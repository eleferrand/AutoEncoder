import tensorflow as tf

import torch
from torch import nn
from fairseq.models.wav2vec import Wav2VecModel
import soundfile as sf

import numpy as np
from shennong.features.processor.mfcc import MfccProcessor
from shennong.features.postprocessor.delta import DeltaPostProcessor
from shennong.features.processor.plp import PlpProcessor
from shennong.features.postprocessor.cmvn import CmvnPostProcessor
from shennong.audio import Audio
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import os
import warnings

tf.enable_eager_execution()
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

class AE_layer(tf.keras.Model):
    def __init__(self):
        super(AE_layer, self).__init__()
        self.Encoder = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        self.bottleneck = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        self.Decoder = tf.keras.layers.Dense(100, activation=tf.nn.tanh)

    def call(self, inp):
        x = self.Encoder(inp)
        bn = self.bottleneck(x)
        x = self.Decoder(bn)
        return x, bn

class AutoEncoder(tf.keras.Model):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.ae1 = AE_layer()
        self.ae2 = AE_layer()
        self.ae3 = AE_layer()
        self.ae4 = AE_layer()
        self.ae5 = AE_layer()
        self.ae6 = AE_layer()
        self.ae7 = AE_layer()

        self.LastEnc = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        
        self.LastBN = tf.keras.layers.Dense(39, activation=tf.nn.tanh)
    
        self.LastDec = tf.keras.layers.Dense(100, activation=tf.nn.tanh)
        
        self.dense_final = tf.keras.layers.Dense(512, activation=tf.nn.relu)
        
    
    def call(self, inp):
        x, bn = self.ae1(inp)
        x, bn = self.ae2(bn)
        x, bn = self.ae3(bn)
        x, bn = self.ae4(bn)
        x, bn = self.ae5(bn)
        x, bn = self.ae6(bn)
        x, bn = self.ae7(bn)
        x = self.LastEnc(bn)
        bn = self.LastBN(x)
        x = self.LastDec(bn)

        x = self.dense_final(x)
        return x, bn

def trunc_and_limit_dim(x, lengths, d_frame, max_length):
    for i, seq in enumerate(x):
        x[i] = x[i][:max_length, :d_frame]

        lengths[i] = min(lengths[i], max_length)

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.logging.set_verbosity(tf.logging.ERROR)
if type(tf.contrib) != type(tf):
    tf.contrib._warning = None

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


model_wtv = Prediction("/home/getalp/leferrae/thesis/mb_model/checkpoint_best.pt", 0)
model_path = "./models/AE_wtv_ckpt/ae_best_model.ckpt"
print("loading weights")
autoencoder = AutoEncoder()
autoencoder.load_weights(model_path)



print("testing model")
file1 = "/home/getalp/leferrae/thesis/corpora/audios/abiayi_2015-09-08-12-50-23_samsung-SM-T530_mdw_elicit_Dico17_177.wav"
inp_x= [get_wtv(file1, norm="cmvn", model=model_wtv)]

trunc_and_limit_dim(inp_x, [len(inp_x)], 512, len(inp_x[0]))
inp_x = np.asarray(inp_x)
x, bn = autoencoder(inp_x)

print(bn.shape, inp_x[0].shape)
d = cdist(inp_x[0], x[0], metric = "cosine")
bs = cdist(inp_x[0], inp_x[0], metric = "cosine")
plt.imshow(d, interpolation='nearest')
plt.show()
plt.imshow(bs, interpolation='nearest')
plt.show()
