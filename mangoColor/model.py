import os
import pathlib
import subprocess
import kaggle 
from scipy import misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image

class MangoColor:
    def __init__(self, model_path="", dataset_path="", model_stored=False):
        cur_dir = str(pathlib.Path(__file__).parent.resolve())

        self.model_path = os.path.join(cur_dir, "model.ckpt")
        self.dataset_path = os.path.join(cur_dir, "dataset") 
        self.model_stored = model_stored

        self.image_transform_done = False

        if(self.model_stored == False):
            ## Dataset is downloaded, stored. dir for storing model checkpoint is created. 
            os.makedirs(self.dataset_path)
            os.chdir(self.dataset_path)
            kaggle_cmd = "kaggle datasets download -d ktaebum/anime-sketch-colorization-pair"
            unzip_cmd = "unzip anime-sketch-colorization-pair.zip"
            delete_zip = "rm -f ktaebum/anime-sketch-colorization-pair"
            subprocess.run(kaggle_cmd, shell=True)
            subprocess.run(unzip_cmd, shell=True)
            subprocess.run(delete_zip, shell=True)
            os.chdir(cur_dir)
            print(f"Dataset downloaded and stored at {self.dataset_path}")

        else:
            print("environment varibales set")
        
    def load_images(self, force=False):
        if(self.image_transform_done == True and not force):
            print("this step already done")
            return
        self.train_path = os.path.join(self.cur_dir, "dataset/data/data/train")



def main():
    # mc = MangoColor()
    img = mpimg.imread('dataset/data/data/train/1020.png')
    height, width, _ = img.shape

    # Split the image into two parts
    half_width = width // 2
    left_half = img[:, :half_width, :]
    right_half = img[:, half_width:, :]

    # Convert the numpy arrays back to PIL images
    # led

    plt.subplot(1, 2, 1)
    plt.imshow(left_half)
    plt.title('Left Half')

    plt.subplot(1, 2, 2)
    plt.imshow(right_half)
    plt.title('Right Half')

    plt.show()

    return


if __name__=="__main__":
    main()    
