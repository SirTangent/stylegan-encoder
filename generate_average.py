# Load Configuration
import configparser

config = configparser.ConfigParser()
config.read("settings.ini")
glob = config["DEFAULT"]

import os
import argparse
import numpy as np
import config
import dnnlib
import dnnlib.tflib as tflib
import pickle
import PIL.Image

# load the StyleGAN model into Colab
URL_FFHQ = glob['ResnetModelURL']
tflib.init_tf()
with dnnlib.util.open_url(URL_FFHQ, cache_dir=config.cache_dir) as f:
    generator_network, discriminator_network, Gs_network = pickle.load(f)


# Load latents and return average
def load_latents(dir):
    s = []
    for filen in os.listdir(dir):
        if filen.endswith(".npy"):
            s.append(np.expand_dims(np.load(os.path.join(dir, filen)), axis=0))
    return (1 / len(s)) * sum(s)


def main():
    parser = argparse.ArgumentParser(
        description='Generate the average face from a directory of latents.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Directory with latents')
    parser.add_argument('--dst_dir', default='images-average/', help='Directory to output average')
    parser.add_argument('--name', default='output', help='PNG file name to output')

    args, other_args = parser.parse_known_args()

    savg = load_latents(args.src_dir)

    save_dir = args.dst_dir

    # run the generator network to render the latents:
    synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),
                            minibatch_size=8)
    images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)
    # display(PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS))

    out_name = os.path.join(save_dir, args.name + ".png")
    img = PIL.Image.fromarray(images.transpose((0, 2, 3, 1))[0], 'RGB').resize((512, 512), PIL.Image.LANCZOS)
    img.save(out_name)

if __name__ == "__main__":
    main()