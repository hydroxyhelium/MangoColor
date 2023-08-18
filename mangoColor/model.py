import os
import pathlib
import subprocess
import kaggle 
from scipy import misc
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision

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
        
    def load(self, image_path: str):
        img = mpimg.imread(image_path)
        height, width, _ = img.shape
        tensor_image = torch.from_numpy(img)

        half_width = width // 2
        left_half = tensor_image[:, :half_width, :]
        right_half = tensor_image[:, half_width:, :] 

        left_half = left_half.permute(2, 0, 1)
        right_half = right_half.permute(2, 0, 1)
            
        return right_half, right_half
    
    def resize(self, input_image, real_image, height, width):
        input_image = torchvision.transforms.Resize((height,width),torchvision.transforms.InterpolationMode.NEAREST)(input_image)
        real_image = torchvision.transforms.Resize((height,width),torchvision.transforms.InterpolationMode.NEAREST)(real_image)

        return input_image, real_image
    
    def random_crop(self, input_image, real_image):
        stacked_image = torch.stack((input_image, real_image), 0)
        cropped_image = torchvision.transforms.RandomCrop((2, 512, 512, 3))(stacked_image)

        return cropped_image[0], cropped_image[1]
    
    def normalize(self, input_image, real_image):
        input_image = (input_image*2)-1
        real_image = (real_image*2)-1 

        return input_image, real_image
    
    def random_jitter(self, input_image, real_image):
        input_image, real_image = self.resize(input_image, real_image, 542, 542)
        input_image, real_image = self.random_crop(input_image, real_image)
        input_image, real_image = self.normalize(input_image, real_image)

        ## we could in the future implement random flip, for now I'm skipping it

        return input_image, real_image
    
    def plot(image_tensor1, image_tensor2):
        image_array1 = image_tensor1.numpy()
        image_array2 = image_tensor2.numpy()

        # Scale the pixel values to [0, 1] range (assuming images are in the range [-1, 1])
        image_array1 = (image_array1 + 1) / 2
        image_array2 = (image_array2 + 1) / 2

        # Create a figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display the images in the subplots
        axes[0].imshow(np.transpose(image_array1, (1, 2, 0)))  # Transpose to (height, width, channels)
        axes[0].set_title('Image 1')
        axes[0].axis('off')

        axes[1].imshow(np.transpose(image_array2, (1, 2, 0)))
        axes[1].set_title('Image 2')
        axes[1].axis('off')

        # Show the plot
        plt.tight_layout()
        plt.show()

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
