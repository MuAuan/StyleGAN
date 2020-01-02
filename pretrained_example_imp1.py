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
    latents2_ = np.load('./latent/donald_trump_01.npy') #face128.npy #donald_trump_01.npy
    latents2 = np.zeros((18,512))
    latents2 = latents2_
    print(latents1.shape, latents2[1].shape)
    fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    images = Gs.run(latents2[2].reshape(1,512), None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
    #images = Gs.components.synthesis.run(latents2, randomize_noise=False, **synthesis_kwargs)
    #latents2_images = Gs.components.synthesis.run(latents2, randomize_noise=False, **synthesis_kwargs)
    plt.imshow(images.reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/trump_.png")
    plt.close()
    
    for i in range(1,101,4):
        latents = i/100*latents1+(1-i/100)*latents2[2].reshape(1,512)
        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7, randomize_noise=True, output_transform=fmt)
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)

    s=25
    images = []
    for i in range(1,101,4):
        im = Image.open(config.result_dir+'/example'+str(i)+'.png') 
        im =im.resize(size=(256, 256), resample=Image.NEAREST)
        images.append(im)
    
    images[0].save(config.result_dir+'/example{}_{}.gif'.format(0,1), save_all=True, append_images=images[1:s], duration=100*5, loop=0)    
    
if __name__ == "__main__":
    main()
