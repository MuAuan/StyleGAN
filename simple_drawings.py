# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def main():
    # Initialize TensorFlow.
    tflib.init_tf()
    fpath = './weight_files/tensorflow/karras2019stylegan-ffhq-1024x1024.pkl'
    with open(fpath, mode='rb') as f:
        _G, _D, Gs = pickle.load(f)


    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    rnd = np.random.RandomState(5) #5
    latents1 = rnd.randn(1, Gs.input_shape[1])
    print(latents1.shape)
    
    # Generate image.
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents1, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    plt.imshow(images.reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/simple1_.png")
    plt.close()
    """
    latents = rnd.randn(1, Gs.input_shape[1])
    images = Gs.get_output_for(latents, None, is_validation=True, randomize_noise=True)
    #images = tflib.convert_images_to_uint8(images)
    print(images.shape)
   
    images = tf.reshape(images, [1024, 1024, 3])
    print(images)
    im = np.array(images, dtype=float)
    #im = np.squeeze(images)
    plt.imshow(im)
    plt.pause(1)
    plt.savefig("./results/simple2_.png")
    plt.close()
    """
    
    src_seeds=[5,6,7,8,9]
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    plt.imshow(src_images[1].reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/simple3_.png")
    plt.close()
    
    for i in range(1,101,4):
        latents = i/100*latents1+(1-i/100)*src_latents[1].reshape(1,512)
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
    
    #latents1 = rnd.randn(1, Gs.input_shape[1])
    for i in range(1,101,4):
        #latents = i/100*latents1+(1-i/100)*src_latents[3].reshape(1,512)
        # Generate image.
        #src_dlatents1 = Gs.components.mapping.run(latents1, None) # [seed, layer, component]
        src_dlatents1 = np.load('./latent/mayuyu256.npy') #face128.npy #donald_trump_01.npy #mayuyu256.npy
        src_dlatents2 = Gs.components.mapping.run(src_latents[1].reshape(1,512), None) # [seed, layer, component]
        src_dlatents = i/100*src_dlatents1+(1-i/100)*src_dlatents2
        
        src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'src_images{}.png'.format(i))
        PIL.Image.fromarray(src_images[0], 'RGB').save(png_filename)
        
    s=25
    images = []
    for i in range(1,101,4):
        im = Image.open(config.result_dir+'/src_images'+str(i)+'.png') 
        im =im.resize(size=(1024, 1024), resample=Image.NEAREST)
        images.append(im)
    
    images[0].save(config.result_dir+'/example{}_{}.gif'.format(0,1), save_all=True, append_images=images[1:s], duration=100*2, loop=0)           
    
if __name__ == "__main__":
    main()
