import tensorflow as tf
import os, cv2
import numpy as np

IMG_SIZE = (512, 512)
CROP_SIZE = [256, 256, 3]
BATCH_SIZE = 8


def img_reader(img_path):    
    def resize(img):
        return cv2.resize(img, IMG_SIZE, cv2.INTER_LINEAR)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    # tf.image.resize(img, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)    
    img = tf.numpy_function(resize, [img], tf.float32)
    img = tf.image.random_crop(img, CROP_SIZE) 
    return img


def mscoco_dataset(path, shuffle = True, batchsize = BATCH_SIZE):
    ds = tf.data.Dataset.list_files(os.path.join(path, '*'))
    ds = ds.map(img_reader, tf.data.experimental.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batchsize).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    # ds = ds.cache().shuffle(1000).repeat().batch(8).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return ds