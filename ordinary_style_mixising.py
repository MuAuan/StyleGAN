import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import matplotlib.pyplot as plt

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

def load_Gs():
    fpath = './weight_files/tensorflow/karras2019stylegan-ffhq-1024x1024.pkl'
    with open(fpath, mode='rb') as f:
        _G, _D, Gs = pickle.load(f)
    return Gs

def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    print(png)
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
#    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
#    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]

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
    dst_dlatents = np.zeros((6,18,512))
    for j in range(6):
        dst_dlatents[j] = dlatents1

    print(dst_dlatents.shape)

    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)
    print(dst_images.shape)

    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    for col, src_image in enumerate(list(src_images)):
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    for row, dst_image in enumerate(list(dst_images)):
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        for col, image in enumerate(list(row_images)):
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

def main():
    tflib.init_tf()
    os.makedirs(config.result_dir, exist_ok=True)
    draw_style_mixing_figure(os.path.join(config.result_dir, 'style-mixing256.png'), 
                             load_Gs(), w=1024, h=1024, src_seeds=[6,701,687,615,2268], dst_seeds=[0,0,0,0,0,0],
                             style_ranges=[range(0,8)]+[range(1,8)]+[range(2,8)]+[range(1,18)]+[range(4,18)]+[range(5,18)])
#case1;style_ranges=[range(0,12)]+[range(0,10)]+[range(0,8)]+[range(0,6)]+[range(0,4)]+[range(0,2)]
#case2;style_ranges=[range(0,12)]+[range(1,12)]+[range(2,12)]+[range(3,12)]+[range(4,12)]+[range(8,12)]
#case3;style_ranges=[range(8,18)]+[range(10,18)]+[range(12,18)]+[range(14,18)]+[range(16,18)]+[range(18,18)]

if __name__ == "__main__":
    main()