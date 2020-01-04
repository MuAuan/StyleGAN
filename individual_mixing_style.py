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
    dlatents1 = Gs.components.mapping.run(latents1, None) # [seed, layer, component]
    images = Gs.components.synthesis.run(dlatents1, randomize_noise=False, **synthesis_kwargs)
    plt.imshow(images.reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/simple1_.png")
    plt.close()
    print("dlatents1.shape=",dlatents1.shape)
    
    src_seeds=[6]
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    plt.imshow(src_images[0].reshape(1024,1024,3))
    plt.pause(1)
    plt.savefig("./results/simple3_.png")
    plt.close()
    print("src_dlatents.shape=",src_dlatents.shape)
    
    for i in range(1,26,1):
        dlatents=src_dlatents
        dlatents[0][0] = i/100*dlatents1[0][0]+(1-i/100)*src_dlatents[0][0]
        # Generate image.
        images = Gs.components.synthesis.run(dlatents, randomize_noise=False, **synthesis_kwargs)
        # Save image.
        os.makedirs(config.result_dir, exist_ok=True)
        png_filename = os.path.join(config.result_dir, 'example{}.png'.format(i))
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)
  
        
    s=25
    images = []
    for i in range(1,26,1):
        im = Image.open(config.result_dir+'/example'+str(i)+'.png') 
        im =im.resize(size=(512, 512), resample=Image.NEAREST)
        images.append(im)
    
    images[0].save(config.result_dir+'/simple_method1{}_{}.gif'.format(0,1), save_all=True, append_images=images[1:s], duration=100*2, loop=0)           
    
if __name__ == "__main__":
    main()
