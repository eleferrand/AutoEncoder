# AutoEncoder
AutoEncoder features extractor made with Tensorflow2

## Dependencies
  Tensorflow 2
  pickle
  numpy
  
## For data extraction:
    scipy
    shennong
## If using Wav2Vec representation:
    torch
    fairseq
    soundfile
## else, 
    remove in data.py class PretrainedWav2VecModel and Prediction function read_audio
    and erase import torch, nn, fairseq and soundfile
