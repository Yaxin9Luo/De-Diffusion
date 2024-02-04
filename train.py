from utils.logging import *
from utils.ckpt import *
from utils.distributed import *
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import time
from utils import config
from utils.utils import *
from importlib import import_module
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.encoder import MyModel  
from diffusers import StableDiffusionPipeline  
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from torchvision import datasets
import time
import yaml

class CfgNode:
    def __init__(self, init_dict=None):
        if init_dict is None:
            init_dict = {}
        for k, v in init_dict.items():
            if isinstance(v, dict):
                setattr(self, k, CfgNode(v))
            else:
                setattr(self, k, v)
    
    def __getattr__(self, name):
        return None

def load_cfg_from_cfg_file(file_path):
    with open(file_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    return CfgNode(cfg_dict)

        
def generate_image(text,__C):
    # Load the pre-trained Stable Diffusion model
    # Make sure to adjust the model path or handle as per your setup
    pipe = StableDiffusionPipeline.from_pretrained(__C, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt = text
    image = pipe(prompt).images[0]  
    return image

def tokens_to_sentence(tokens):
    # Join the tokens into a sentence. Adjust the separator if needed
    sentence = ' '.join(tokens)
    return sentence


def train(model,__C):

    #Initialize the model
    model.vit.requires_grad_(True)
    model.clip_model.requires_grad_(True)
    model.to(device)
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=__C.training.learning_rate)
    # Data loaders
    transform = transforms.Compose([
        transforms.Resize((__C.training.resize_dataloader,__C.training.resize_dataloader)),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='/root/autodl-tmp/my_dediffusion/datasets', train=True, download=True, transform=transform)
    # Load the dataset
    indices = list(range(10))  # indices for the first 10 samples
    subset_train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset_train_dataset, batch_size=__C.training.batch_size, shuffle=True) # for 100 images

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) # for whole images

    # Training loop
    for epoch in range(__C.training.num_epochs):
        model.train()
        epoch_start_time = time.time()  # Record the start time of the epoch
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()
            images, _ = data  # Assuming you don't need labels for this task
            # Convert original images to tensor
            images = images.to(device).requires_grad_()  # Original images as tensor

            # Generate text prompts and then images
            generated_texts = model(images)
            sentence = tokens_to_sentence(generated_texts)
            generated_images = generate_image(sentence, __C.training.pre_trained_diffusion_model)

            # Convert generated images to tensor
            generated_image_tensor = transform(generated_images).unsqueeze(0).to(device).requires_grad_() # Add batch dimension
            
            # Compute loss  
            loss = criterion(generated_image_tensor, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{__C.training.num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                print(f"Sample Generated Text: {sentence}")  # Print part of the generated text

    epoch_duration = time.time() - epoch_start_time  # Calculate the epoch duration
    print(f"Epoch [{epoch+1}/{__C.training.num_epochs} completed in {epoch_duration:.2f} seconds.")
    # Save the trained model
    model_save_path = os.path.join(__C.model_checkpoint.path, __C.model_checkpoint.filename)
    torch.save(model.state_dict(), model_save_path)

# Assuming the load_cfg_from_cfg_file function is defined somewhere in your script
def main():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()
    __C = load_cfg_from_cfg_file(args.config)
    # Ensure the 'device' variable is defined globally or passed appropriately
    global device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MyModel(__C.model)  
    train(model, __C)

if __name__ == "__main__":
    main()
