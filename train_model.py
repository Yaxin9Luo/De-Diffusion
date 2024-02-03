import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from models.encoder import MyModel  
from diffusers import StableDiffusionPipeline  
from torch.utils.data import Subset
from PIL import Image
import requests
import hydra
from omegaconf import DictConfig
import logging 
import matplotlib.pyplot as plt
from torchvision import datasets
import time
def generate_image(text,model_id):
    # Load the pre-trained Stable Diffusion model
    # Make sure to adjust the model path or handle as per your setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt = text
    image = pipe(prompt).images[0]  
    return image

def tokens_to_sentence(tokens):
    # Join the tokens into a sentence. Adjust the separator if needed
    sentence = ' '.join(tokens)
    return sentence

@hydra.main(config_path="../configs", config_name="main")
def _main(cfg: DictConfig):
    enc = hydra.utils.instantiate(cfg)
    model = MyModel(enc.model)
    train(enc.epoch, model)


def train(cfg, model):
    #Configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = cfg['learning_rate']
    batch_size = cfg['batch_size']
    num_epochs = cfg['num_epochs']

    #Initialize the model
    model.vit.requires_grad_(True)
    model.clip_model.requires_grad_(True)
    model.to(device)

    #Initialize the text-to-image model
    diffusion_model = StableDiffusionPipeline.from_pretrained(cfg['pre_trained_diffusion_model']).to(device)
    # Loss function
    criterion = nn.MSELoss()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # Data loaders
    transform = transforms.Compose([
        transforms.Resize((cfg['resize_dataloader'],cfg['resize_dataloader'])),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.CIFAR10(root='/root/autodl-tmp/MLops_project_Group18/data/raw', train=True, download=True, transform=transform)
    # Load the dataset
    # url = cfg['image_url'] 
    # image = Image.open(requests.get(url, stream=True).raw) # 640x480
    indices = list(range(1000))  # indices for the first 100 samples
    subset_train_dataset = Subset(train_dataset, indices)
    train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(cfg['num_epochs']):
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
            generated_images = generate_image(sentence, cfg['generate_image_model'])

            # Convert generated images to tensor
            generated_image_tensor = transform(generated_images).unsqueeze(0).to(device).requires_grad_() # Add batch dimension
            
            # Compute loss  
            loss = criterion(generated_image_tensor, images)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:  # Print every 10 batches
                print(f"Epoch [{epoch+1}/{cfg['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                print(f"Sample Generated Text: {sentence}")  # Print part of the generated text

    epoch_duration = time.time() - epoch_start_time  # Calculate the epoch duration
    print(f"Epoch [{epoch+1}/{cfg['num_epochs']} completed in {epoch_duration:.2f} seconds.")
    # Save the trained model
    torch.save(model.state_dict(), "/root/autodl-tmp/MLops_project_Group18/logs/model_checkpoints_medium.pth")

if __name__ == "__main__":
    _main()