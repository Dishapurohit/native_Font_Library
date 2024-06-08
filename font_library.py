# Monkey patch for collections.abc import issue
import collections.abc   # I found error like in 'setup language' so that I use this library to
# iterate 'collection' file
collections.Iterable = collections.abc.Iterable

from inltk.inltk import setup     # use inltk library to perform various NLP tasks
# like indian native language classification

import pandas as pd     # for data collection
import numpy as np      # for data analysis
import matplotlib.pyplot as plt   # for data visualization
import tensorflow as tf   # for machine learning operations
from tensorflow.keras import layers

# Setting up inltk for Hindi, Malayalam, and Tamil
setup('hi')  # Hindi
setup('ml')  # Malayalam
setup('ta')  # Tamil

# [1] Data Collection
load_data = pd.read_csv('characters_native_languages.csv')      # Load data from CSV
print(load_data)

# GAN models


def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(256, input_shape=(100,)))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(1024))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Dense(np.prod((28, 28, 1)), activation='tanh'))
    model.add(layers.Reshape((28, 28, 1)))
    return model


def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(512))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model


# Compile models
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

# Combined model
z = layers.Input(shape=(100,))
img = generator(z)
discriminator.trainable = False
valid = discriminator(img)

combined = tf.keras.Model(z, valid)
combined.compile(optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), loss='binary_crossentropy')


# Training function
def train(epochs, batch_size=128, save_interval=50):
    X_train, _ = load_data
    X_train = (X_train / 127.5) - 1.
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
        g_loss = combined.train_on_batch(noise, valid)

        if epoch % save_interval == 0:
            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100 * d_loss[1]}] [G loss: {g_loss}]")
            save_imgs(epoch, generator)


def save_imgs(epoch, generator, sample_size=5):
    noise = np.random.normal(0, 1, (sample_size * sample_size, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5
    fig, axs = plt.subplots(sample_size, sample_size)
    cnt = 0
    for i in range(sample_size):
        for j in range(sample_size):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"generated_images_{epoch}.png")
    plt.close()
