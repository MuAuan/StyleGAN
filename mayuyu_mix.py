import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

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

    dlatents = np.load('./latent/mayuyu256.npy') #face128.npy #donald_trump_01.npy
    dst_dlatents = np.zeros((6,18,512))
    dst_dlatents[0] = dlatents
    dst_dlatents[1] = dlatents
    dst_dlatents[2] = dlatents
    dst_dlatents[3] = dlatents
    dst_dlatents[4] = dlatents
    dst_dlatents[5] = dlatents

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
    draw_style_mixing_figure(os.path.join(config.result_dir, 'mayuyu256-style-mixing_case2.png'), 
                             load_Gs(), w=1024, h=1024, src_seeds=[639,701,687,615,2268], dst_seeds=[0,0,0,0,0,0],
                             style_ranges=[range(0,12)]+[range(1,12)]+[range(2,12)]+[range(3,12)]+[range(4,12)]+[range(8,12)])
#case1;style_ranges=[range(0,12)]+[range(0,10)]+[range(0,8)]+[range(0,6)]+[range(0,4)]+[range(0,2)]
#case2;style_ranges=[range(0,12)]+[range(1,12)]+[range(2,12)]+[range(3,12)]+[range(4,12)]+[range(8,12)]
    
if __name__ == "__main__":
    main()