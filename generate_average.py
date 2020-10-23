# Load Configuration
import configparser
config = configparser.ConfigParser()
config.read("settings.ini")
glob = config["DEFAULT"]

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

# load the latents
s1 = np.load('images-latent/Sub-directory-name/image1-name_01.npy')
s2 = np.load('images-latent/Sub-directory-name/image2-name_01.npy')
s3 = np.load('images-latent/Sub-directory-name/image3-name_01.npy')
s4 = np.load('images-latent/Sub-directory-name/image4-name_01.npy')
s5 = np.load('images-latent/Sub-directory-name/image5-name_01.npy')

s1 = np.expand_dims(s1,axis=0)
s2 = np.expand_dims(s2,axis=0)
s3 = np.expand_dims(s3,axis=0)
s4 = np.expand_dims(s4,axis=0)
s5 = np.expand_dims(s5,axis=0)

save_dir = "images-average/"

# average
savg = (1/5)*(s1+s2+s3+s4+s5)

# run the generator network to render the latents:
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
images = Gs_network.components.synthesis.run(savg, randomize_noise=False, **synthesis_kwargs)
display(PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS))

out_name = save_dir + "Sub-directory-name" + ".png"
img = PIL.Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').resize((512,512),PIL.Image.LANCZOS)
img.save(out_name)
