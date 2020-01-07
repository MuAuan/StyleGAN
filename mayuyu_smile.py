import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import sys
from PIL import Image, ImageDraw
from encoder.generator_model import Generator
import matplotlib.pyplot as plt

tflib.init_tf()
fpath = './weight_files/tensorflow/karras2019stylegan-ffhq-1024x1024.pkl'
with open(fpath, mode='rb') as f:
    generator_network, discriminator_network, Gs_network  = pickle.load(f)

generator = Generator(Gs_network, batch_size=1, randomize_noise=False)

def generate_image(latent_vector):
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img.resize((256, 256))

def move_and_show(latent_vector, direction, coeffs):
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8]
        plt.imshow(generate_image(new_latent_vector))
        plt.pause(1)
        plt.savefig("./results/example{}.png".format(i))
        plt.close()
    

mayuyu = np.load('./latent/anime.npy') 
smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')  #age.npy'  #gender.npy'

move_and_show(mayuyu, smile_direction, [-1,-0.8,-0.6,-0.4,-0.2, 0,0.2,0.4,0.6,0.8, 1])
s=22
images = []
for i in range(0,11,1):
    im = Image.open(config.result_dir+'/example'+str(i)+'.png') 
    im =im.resize(size=(640,480), resample=Image.NEAREST)
    images.append(im)
    
for i in range(10,0,-1):
    im = Image.open(config.result_dir+'/example'+str(i)+'.png') 
    im =im.resize(size=(640, 480), resample=Image.NEAREST)
    images.append(im)     

images[0].save(config.result_dir+'/anime_smile{}.gif'.format(11), save_all=True, append_images=images[1:s], duration=100*2, loop=0)        