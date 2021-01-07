import os

class runtime:
    def __init__(self, img_dir="images/"):
        self.dir = img_dir

    def run_aligner(self, src_dir="", target_dir=""):
        if src_dir == "":
            src_dir = self.dir
        if target_dir == "":
            target_dir = os.path.join(src_dir, "images-aligned")

        # Run asynchronously
        return os.system(f"python align_images.py {src_dir} {target_dir}")

    def run_encoder(self, aligned_dir="", gen_dir="", latent_dir=""):
        if aligned_dir == "":
            aligned_dir = os.path.join(self.dir, "images-aligned")
        if gen_dir == "":
            gen_dir = os.path.join(self.dir, "images-generated")
        if latent_dir == "":
            latent_dir = os.path.join(self.dir, "images-latent")

        # Run asynchronously
        return os.system(f"python encode_images.py --batch_size=1 --output_video=False {aligned_dir} {gen_dir} {latent_dir}")