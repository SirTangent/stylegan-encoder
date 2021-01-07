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
        os.system(f"python align_images.py {src_dir} {target_dir}")
        return "Operation successful!", True