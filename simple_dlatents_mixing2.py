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
    rnd = np.random.RandomState(6) #5
    latents1 = rnd.randn(1, Gs.input_shape[1])
    print(latents1.shape)
    
    # Generate image.
    dlatents1 = Gs.components.mapping.run(latents1, None) # [seed, layer, component]
    images = Gs.components.synthesis.run(dlatents1, randomize_noise=False, **synthesis_kwargs)
    plt.imshow(images.reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/simple1_.png")
    plt.close()
    print("1")
    
    #src_seeds=[6,7]
    #src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    #src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    src_dlatents = np.load('./latent/mayuyu250px_01.npy')
    src_dlatents = src_dlatents.reshape(1,18,512)
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    plt.imshow(src_images[0].reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/simple3_.png")
    plt.close()
    
    for i in range(1,101,4):
        dlatents = i/100*dlatents1+(1-i/100)*src_dlatents
        # Generate image.
        images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
  
        
    s=50
    images = []
    for i in range(101,1,-4):
        im = Image.open(config.result_dir+'/example'+str(i)+'.png') 
        im =im.resize(size=(1024, 1024), resample=Image.NEAREST)
        images.append(im)
    for i in range(1,101,4):
        im = Image.open(config.result_dir+'/example'+str(i)+'.png') 
        im =im.resize(size=(1024, 1024), resample=Image.NEAREST)
        images.append(im)     
    
    images[0].save(config.result_dir+'/simple_method1{}_{}_dr1024.gif'.format(0,1), save_all=True, append_images=images[1:s], duration=100*2, loop=0)           
    
if __name__ == "__main__":
    main()
