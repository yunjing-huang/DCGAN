import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
from IPython import display
from PIL import Image
BATCH_SIZE=32
train_images = tf.keras.preprocessing.image_dataset_from_directory("D:/cb", label_mode=None, batch_size=BATCH_SIZE, image_size=(128, 128))
normalization_layer = layers.experimental.preprocessing.Rescaling(scale=1./127.5, offset=-1)
normalized_images = train_images.map(lambda x: normalization_layer(x))
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(8*8*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((8, 8, 256)))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 8, 8, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 16, 16, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 64, 64, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 3)
    return model
generator = make_generator_model()
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 3]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model
discriminator = make_discriminator_model()
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
EPOCHS = 5000
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
def saveimage(image,epoch):
    image_array1=np.concatenate([image[0],image[1],image[2],image[3]], axis=1)
    image_array2 = np.concatenate([image[4], image[5], image[6], image[7]], axis=1)
    image_array3 = np.concatenate([image[8], image[9], image[10], image[11]], axis=1)
    image_array4 = np.concatenate([image[12], image[13], image[14], image[15]], axis=1)
    image_array = np.concatenate([image_array1, image_array2, image_array3, image_array4], axis=0)
    image_array=(image_array+1)*127.5
    image=Image.fromarray(image_array.astype('uint8'))
    image.save('C:/Users/Yunji/PycharmProjects/python1/deeplearning/image/image_at_epoch_{:05d}.jpg'.format(epoch))
def train(dataset, epochs):
    for epoch in range(1, epochs+1):
        start = time.time()
        for image_batch in dataset:
            train_step(image_batch)
        predictions = generator(seed, training=False)
        saveimage(predictions, epoch)
        if epoch % 20 == 0:
            generator.save('C:/Users/Yunji/PycharmProjects/python1/deeplearning/model/generator_at_epoch_{:05d}'.format(epoch))
            discriminator.save('C:/Users/Yunji/PycharmProjects/python1/deeplearning/model/discriminator_at_epoch_{:05d}'.format(epoch))
        print ('Time for epoch {} is {} sec'.format(epoch, time.time()-start))
train(normalized_images, EPOCHS)





