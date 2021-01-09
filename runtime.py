import os


class Runtime:
    def __init__(self, img_dir="images/"):
        self.dir = img_dir

    def run_aligner(self, src_dir="", target_dir=""):
        if src_dir == "":
            src_dir = self.dir
        if target_dir == "":
            target_dir = os.path.join(src_dir, "images-aligned")

        # Run asynchronously
        return os.system(f"python align_images.py {src_dir} {target_dir}")

    def run_encoder(self, aligned_dir="", gen_dir="", latent_dir="", output_video=True):
        if aligned_dir == "":
            aligned_dir = os.path.join(self.dir, "images-aligned")
        if gen_dir == "":
            gen_dir = os.path.join(self.dir, "images-generated")
        if latent_dir == "":
            latent_dir = os.path.join(self.dir, "images-latent")

        # Run asynchronously
        return os.system(
            f"python encode_images.py --batch_size=1 --output_video={'True' if output_video else 'False'} {aligned_dir} {gen_dir} {latent_dir}")

    def run_averager(self, latent_dir="", output_file=""):
        if latent_dir == "":
            latent_dir = os.path.join(self.dir, "images-latent")
        if output_file == "":
            output_file = os.path.join(self.dir, "images-average/out")

        # Split path
        head, tail = os.path.split(output_file)

        # TODO: Validate appropriate path name

        # Run asynchronously
        return os.system(f"python generate_average.py {latent_dir} {f'--dst_dir {head}' if head else ''} --name={tail}")

    def run_pipeline(self, src_dir="", aligned_dir="", gen_dir="", latent_dir="", output_file=""):
        return [
            self.run_aligner(src_dir, aligned_dir),
            self.run_encoder(aligned_dir, gen_dir, latent_dir),
            self.run_averager(latent_dir, output_file)
        ]