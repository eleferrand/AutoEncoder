from train_ae import AutoEncoder
from progress.bar import Bar
import numpy as np
import _pickle as pickle
from data import get_mfcc_dd
import os
import argparse
import scipy.io.wavfile as wav
import json
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--data", type=str, default="lex")



args = parser.parse_args()
choix = args.data

def trunc_and_limit_dim(x, d_frame):
    for i, seq in enumerate(x):
        x[i] = x[i][:x[i].shape[0], :d_frame]
def parse_corpus(root):
    x=[]
    refs = []
    repo = os.listdir(root)
    for i in range(0,len(repo)):
        if ".wav" in repo[i]:
            f = get_mfcc_dd(root+repo[i], norm="cmvn")
            x.append(f)
            refs.append(repo[i])
    return [x, refs]

root = '/home/getalp/leferrae/thesis/corpora/audios/'

# trunc_and_limit_dim(data[0], 39)


model_path = "./models/AE_wtv_ckpt/ae_best_model.ckpt"
print("loading weights")
autoencoder = AutoEncoder()
autoencoder.load_weights(model_path)
if choix=="corpus":
    print("extracting corpus")
    data = parse_corpus(root)
    output = []
    bar = Bar("feature extraction", max=5130)
    for elt in data[0]:
        inp_x = np.asarray([elt])
        out_x, out_bn = autoencoder(inp_x)
        bnf = np.asarray(out_bn)
        output.append(bnf[0])
        bar.next()
    bar.finish()
    final = {}
    sortie = []
    for i in range(0, len(output)):
        final[i] = output[i]
    for i in range(0, len(output)):
        sortie.append({"file" : data[1][i], "data" : final[i]})


    with open("corp_cae_mfcc_cmvn.pkl", mode="wb") as jfile:
        pickle.dump(sortie, jfile)
elif choix == "lex":
    print("extracting lexicon")
    queries_list = []
    f_lex = "../corpora/crops/lex100/spoken_lex100.json"
    with open(f_lex, mode='r', encoding='utf8') as jfile:
        lex = json.load(jfile)
    wavs = []
    durs = []
    for ind in lex:
        wavs.append(ind["crop"])
        rate, signal = wav.read(ind["crop"])
        durs.append(len(signal) / rate * 1000)
    vects = []
    for i in wavs:
        vects.append(get_mfcc_dd(i, norm="cmvn"))


    output = []

    for elt in vects:
        inp_x = np.asarray([elt])
        out_x, out_bn = autoencoder(inp_x)
        bnf = np.asarray(out_bn)
        output.append(bnf[0])

    rep_dict = {}
    for ind in range(0, len(output)):
        rep_dict[ind] = output[ind]
    for ind in range(0, len(rep_dict)):
        query= {}
        query["duree"] = durs[ind]
        query["data"] = [rep_dict[ind]]
        query["word"] = lex[ind]["mboshi"]
        query["ref"] = [lex[ind]["ref"]]
        queries_list.append(query)
    with open("lex_cae_mfcc_cmvn.pkl", mode="wb") as jfile:
        pickle.dump(queries_list, jfile)

    for i in range(0, len(vects)):
        print(vects[i].shape, queries_list[i]["data"][0].shape)
