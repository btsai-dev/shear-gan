import os
from pathlib import Path
import sys
import tensorflow as tf
import imageio
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import PIL
from PIL import Image
from tensorflow.keras import layers
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from IPython import display


ROOT_DIR = str(Path(__file__).resolve().parents[1])
sys.path.append(ROOT_DIR)

DATA_DIR = os.path.join(ROOT_DIR, 'data')
AUG_DIR = os.path.join(ROOT_DIR, 'augmented')
AUG_OUT_DIR = os.path.join(AUG_DIR, 'image')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoints')
MODEL_DIR = os.path.join(ROOT_DIR, 'saved_models')

MODEL_LOC = os.path.join(MODEL_DIR, 'RecentModel', 'gen_model.h5')


generator = tf.keras.models.load_model(MODEL_LOC)
generator.summary()

noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
gen_numpy = generated_image.numpy()

gen_numpy_disp = (generated_image[0, :, :, 0] * 127.5 + 127.5).numpy()
print(gen_numpy)
print(gen_numpy.shape)
im_disp = Image.fromarray(gen_numpy_disp)
if im_disp.mode != 'RGB':
    im_disp = im_disp.convert('RGB')
im_disp.save("FAKE.png")

