import tensorflow as tf
import data
import batching
from tensorflow.keras.datasets import mnist
import numpy as np
import os
import _pickle as pickle
tf.enable_eager_execution()


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


def loss(x, x_bar):
    return tf.losses.mean_squared_error(x, x_bar)
def grad(model, inputs):
    with tf.GradientTape() as tape:
        x, bn = model(inputs)
        loss_value = loss(inputs, x)
    return loss_value, tape.gradient(loss_value, model.trainable_variables), inputs, x

def main():
    batch_size = 300
    input_x = tf.keras.Input(shape=(100,39))
    autoencoder = AutoEncoder()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    global_step = tf.Variable(0)
    n_epochs = 150

    if os.path.isfile("train_wtv.pkl"):
        with open("train_wtv.pkl", mode="rb") as pfile:
            f = pickle.load(pfile)
        train_x = f[0]
        train_labels = f[1]
        train_lengths = f[2]
        train_keys = f[3]
    else:
        train_x, train_labels, train_lengths, train_keys = data.load_utt("train", "wtv", "smp")
    if os.path.isfile("dev_wtv.pkl"):
        with open("dev_wtv.pkl", mode="rb") as pfile:
            f = pickle.load(pfile)
        val_x = f[0]
        val_labels = f[1]
        val_lengths = f[2]
        val_keys = f[3]
    else:
        val_x, val_labels, val_lengths, val_keys = data.load_utt("val", "wtv", "smp")

    trunc_and_limit_dim(train_x, train_lengths, 512, 100)
    trunc_and_limit_dim(val_x, val_lengths, 512, 100)

    train_x=np.asarray(train_x)
    val_x= np.asarray(val_x)
    best_val = np.inf
    losses = []
    for epoch in range(n_epochs):
        print("Epoch: ", epoch)
        for x in range(0, len(train_x), batch_size):
            x_inp = train_x[x : x + batch_size]

            loss_value, grads, inputs, reconstruction = grad(autoencoder, x_inp)
            optimizer.apply_gradients(zip(grads, autoencoder.trainable_variables),
                                global_step)
            print("batch nb. {}      loss: {}".format(int(x/300), loss_value))
        v, bn = autoencoder(val_x)
        val_loss = loss(val_x, v)
        losses.append(loss_value)
        print("validation:          Loss: {} train loss : {}".format(loss(val_x, v),np.mean(losses)))
        if val_loss<best_val:
            best_val=val_loss
            if not os.path.isdir("./models"):
                os.mkdir("models")
            if not os.path.isdir("./models/AE_ckpt"):
                os.mkdir("models/AE_ckpt")
            autoencoder.save_weights("./models/AE_wtv_ckpt/ae_best_model.ckpt")
            print("new best model")
        if epoch == 0:
            autoencoder.save_weights("./models/AE_wtv_ckpt/ae_rand_model.ckpt")
        else:
            autoencoder.save_weights("./models/AE_wtv_ckpt/ae_last_model.ckpt")


if __name__ == "__main__":
    main()
