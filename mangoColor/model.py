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
import numpy as np

from torch.utils.data import Dataset, DataLoader

from datasetClass import MangoColorDataset

class MangoColor:
    def __init__(self, model_path="", dataset_path="", model_stored=False):
        cur_dir = str(pathlib.Path(__file__).parent.resolve())

        self.model_path = os.path.join(cur_dir, "model.ckpt")
        self.dataset_path = os.path.join(cur_dir, "dataset") 
        self.model_stored = model_stored
        self.BATCH_SIZE = 1

        self.image_transform_done = False

        self.image_path = os.path.join(self.dataset_path, "data/data/train")
        self.loss_object = torch.nn.BCEWithLogitsLoss() ## we define it here, to make sure gradient accumulates
        
        ## I already downloaded data so skipping this step for now

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

        self.dataset_object = MangoColorDataset(self.image_path, self)

        self.data_loader = DataLoader(self.dataset_object, self.BATCH_SIZE, shuffle=True)
        self.iterator = iter(self.data_loader)
        
    def load(self, image_path: str):
        img = mpimg.imread(image_path)
        height, width, _ = img.shape
        tensor_image = torch.from_numpy(img)

        half_width = width // 2
        left_half = tensor_image[:, :half_width, :]
        right_half = tensor_image[:, half_width:, :] 

        left_half = left_half.permute(2, 0, 1)
        right_half = right_half.permute(2, 0, 1)
            
        return right_half, left_half
    
    def resize(self, input_image, real_image, height, width):
        input_image = torchvision.transforms.Resize((height,width),torchvision.transforms.InterpolationMode.NEAREST)(input_image)
        real_image = torchvision.transforms.Resize((height,width),torchvision.transforms.InterpolationMode.NEAREST)(real_image)

        return input_image, real_image
    
    def random_crop(self, input_image, real_image):
        stacked_image = torch.stack((input_image, real_image), 0)
        cropped_image = torchvision.transforms.RandomCrop((512, 512))(stacked_image)

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
    
    def plot(self, image_tensor1, image_tensor2):
        """Note the dim needs to be (3, 512, 512) in order to use and not 
        (1, 3, 512, 512)
        """
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

    def generator_loss(self, disc_generatored_output, gen_output, target):
        
        mean_loss_layer = torch.nn.MSELoss()
        gan_loss = self.loss_object(torch.ones_like(disc_generatored_output), disc_generatored_output)

        l1_loss = mean_loss_layer(gen_output, target)
        total_loss = l1_loss + gan_loss

        return total_loss, l1_loss, gan_loss
    
    def discriminator_loss(self, disc_real_output, disc_generated_output):
        real_loss = self.loss_object(torch.ones_like(disc_real_output), disc_real_output)
        gen_loss = self.loss_object(torch.zeros_like(disc_generated_output),disc_generated_output)

        total_loss = real_loss+gen_loss

        return total_loss
    
    def train_per_epoch(self, epoch_index, gen_tb_writer, disc_tb_writer):
        ## we add tb_writer to visulize running loss

        for i, data in enumerate(self.data_loader):
            input_image, real_image = data
            self.generator_optimizer.zero_grad()
            self.discriminator_optimizer.zero_grad()

            gen_output = self.generator(input_image)

            disc_real_output = self.discriminator(torch.concat((input_image, real_image), 1))
            disc_fake_output = self.discriminator(torch.concat((input_image, gen_output), 1))

            gen_total_loss, gen_l1_loss, gen_gan_loss = self.generator_loss(disc_fake_output, gen_output, real_image)
            total_disc_loss = self.discriminator_loss(disc_real_output, disc_fake_output)

            gen_total_loss.backward()
            total_disc_loss.backward() 

            self.generator_optimizer.step()
            self.discriminator_optimizer.step()

            if(i%1000 == 0):
                print(f"epoch {epoch_index}: image iteration {i}, generator loss: {gen_total_loss}, disc loss :{gen_total_loss}")
                tb_x = epoch_index*len(self.data_loader)+i+1
                gen_tb_writer.add_scalar('Loss/train', gen_total_loss, tb_x)
                disc_tb_writer.add_scalar('Loss/train', total_disc_loss, tb_x)
            






def main():
    mc = MangoColor()

    return


if __name__=="__main__":
    main()    
