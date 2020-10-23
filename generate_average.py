# Load Configuration
import configparser

config = configparser.ConfigParser()
config.read("settings.ini")
glob = config["DEFAULT"]

import os
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


print("RUN_NEW")
savg = load_latents('images-latent/test_1')

save_dir = "images-average/"

# run the generator network to render the latents:
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)
# display(PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS))

out_name = save_dir + "test_1" + ".png"
img = PIL.Image.fromarray(images.transpose((0, 2, 3, 1))[0], 'RGB').resize((512, 512), PIL.Image.LANCZOS)
img.save(out_name)
