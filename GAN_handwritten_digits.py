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
def save_plot(examples, epoch, n=10):
  for i in range(n*n):
    plt.subplot(n, n, i+1)
    plt.axis('off')
    plt.imshow(examples[1, :, :, 0], cmap='gray_r')
  filename='generated_plot_e{}.png'.format(epoch+1)
  plt.savefig(filename)
  plt.close()

# EVALUATE PERFORMANCE
def sum_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
  img_real, label_real=generate_real_samples(dataset, n_samples)
  _, acc_real=d_model.evaluate(img_real, label_real, verbose=1)
  img_fake, label_fake=generate_fake_samples(g_model, latent_dim, n_samples)
  _, acc_fake=d_model.evaluate(img_fake, label_fake, verbose=1)
  print('Accuracy real: {}, Accuracy fake: {}'.format(acc_real*100, acc_fake*100))
  save_plot(img_fake, epoch)
  filename='generated_plot_{}.h5'.format(epoch+1)
  g_model.save(filename)

# TRAIN GENERATOR AND DISCRIMINATOR
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epoch=100, n_batch=256):
  batch_per_epoch=int(dataset.shape[0]/n_batch)
  half_batch=int(n_batch/2)
  for i in range(n_epoch):
    for j in range(batch_per_epoch):
      img_real, label_real=generate_real_samples(dataset, half_batch)
      img_fake, label_fake=generate_fake_samples(g_model, latent_dim, half_batch)
      img, label=np.vstack((img_real, img_fake)), np.vstack((label_real, label_fake))
      d_loss, _=d_model.train_on_batch(img, label)
      img_gan=generate_latent_points(latent_dim, n_batch)
      label_gan=np.ones((n_batch, 1))
      g_loss=gan_model.train_on_batch(img_gan, label_gan)
      print('{}, {}/{}, d_loss: {}, g_loss: {}'.format(i+1, j+1, batch_per_epoch, d_loss, g_loss))
      
    if (i+1)%10==0:
      sum_performance(i, g_model, d_model, dataset, latent_dim)

# VARIABLES AND FUNCTION CALL
latent_dim=100
d_model=define_discriminator()
g_model=define_generator(latent_dim)
gan_model=define_gan(g_model, d_model)
dataset=load_real_samples()
train(g_model, d_model, gan_model, dataset, latent_dim)

# GENERATED IMAGES 
from keras.models import load_model
model=load_model('generated_plot_100.h5')
latent_points=generate_latent_points(100, 25)
x=model.predict(latent_points)
save_plot(x, 5)

# GENERATED IMAGES FOR A SPECIFIC POINT IN LATENT SPACE
model=load_model('generated_plot_100.h5')
vector=np.asarray([[0.0 for _ in range (100)]])
x=model.predict(vector)
plt.imshow(x[0, :, :, 0], cmap='gray_r')
plt.show()