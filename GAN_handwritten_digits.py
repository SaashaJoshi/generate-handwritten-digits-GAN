import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
(train_imgs, train_labels), (test_imgs, test_labels)=mnist.load_data()

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Reshape
from keras.layers import LeakyReLU
from keras.utils.vis_utils import plot_model

# DISCRIMINATOR MODEL
def define_discriminator(in_shape=(28,28,1)):
  model=Sequential()
  model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same', input_shape=in_shape))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Conv2D(64, (3,3), strides=(2, 2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Dropout(0.4))
  model.add(Flatten())
  model.add(Dense(1, activation='sigmoid'))
  
  opt = Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
  return model

# GENERATOR MODEL
def define_generator(latent_dim):
  model=Sequential()
  n_nodes=128*7*7
  model.add(Dense(n_nodes, input_dim=latent_dim))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Reshape((7, 7, 128)))
  model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
  model.add(LeakyReLU(alpha=0.2))
  model.add(Conv2DTranspose(1, (7, 7), activation='sigmoid', padding='same'))
  return model

# GAN
def define_gan(g_model, d_model):
  d_model.trainable=False
  model=Sequential()
  model.add(g_model)
  model.add(d_model)
  
  opt=Adam(lr=0.0002, beta_1=0.5)
  model.compile(loss='binary_crossentropy', optimizer=opt)
  return model

# LOAD MNIST TRAINING IMAGES
def load_real_samples():
  (train_imgs, _), (_, _)=mnist.load_data()
  train_img_samples=np.expand_dims(train_imgs, axis=-1)
  train_img_samples=train_img_samples.astype('float32')/255
  return train_img_samples

# SELECT REAL SAMPLES
def generate_real_samples(dataset, n_samples):
  x=np.random.randint(0, dataset.shape[0], n_samples)
  train_img_samples=dataset[x]
  train_label_samples=np.ones((n_samples, 1))
  return train_img_samples, train_label_samples

# GENERATE POINTS IN LATENT SPACE
def generate_latent_points(latent_dim, n_samples):
  latent_points=np.random.randn(latent_dim*n_samples)
  latent_points=latent_points.reshape(n_samples, latent_dim)
  return latent_points

# GENERATE FAKE SAMPLES USING GENERATOR
def generate_fake_samples(g_model, latent_dim, n_samples):
  latent_points=generate_latent_points(latent_dim, n_samples)
  img_fake_samples=g_model.predict(latent_points)
  label_fake_samples=np.zeros((n_samples, 1))
  return img_fake_samples, label_fake_samples

# PLOT GENERATED IMAGES

# EVALUATE PERFORMANCE

# TRAIN GENERATOR AND DISCRIMINATOR

# VARIABLES AND FUNCTION CALL