import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
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
from logs.logger import logger
import wandb

wandb.init(
    # set the wandb project where this run will be logged
    project="MLOps",
    entity="stone-sitong-chen", 
    group="diffusion", # optional
    name="first", # optional
    # track hyperparameters and run metadata
    config={
    "architecture": "Vit",
    "dataset": "one picture",
    "epochs": 10,
    "learning_rate": 0.0001,
    "pre_trained_diffusion_model": "CompVis/stable-diffusion-v1-4",
    "resize_dataloader": 224,
    "image_url": 'http://images.cocodataset.org/val2017/000000039769.jpg',
    "generate_image_model": "CompVis/stable-diffusion-v1-4"
    }
)

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
    # train(enc.epoch, model)


def train(cfg, model):
    #Configurations
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    learning_rate = cfg['learning_rate']
    batch_size = cfg['batch_size']
    num_epochs = cfg['num_epochs']

    #Initialize the model
    model.vit.requires_grad_(False)
    model.clip_model.requires_grad_(False)
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
    # train_dataset = datasets.CIFAR10(root='/root/autodl-tmp/MLops_project_Group18/data/raw', train=True, download=True, transform=transform)
    # Load the dataset
    url = cfg['image_url'] 
    image = Image.open(requests.get(url, stream=True).raw) # 640x480
    # indices = list(range(100))  # indices for the first 100 samples
    # subset_train_dataset = Subset(train_dataset, indices)
    # train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(cfg['num_epochs']):
        model.train()
            # for images, _ in train_loader:
        images = image
        optimizer.zero_grad()    
        # Generate text prompts using your model
        generated_texts = model(images)
        sentence = tokens_to_sentence(generated_texts)
        # Generate images from text prompts
        generated_images = generate_image(sentence,cfg['generate_image_model'])
        # Apply the transform to both PIL images to convert them to tensors and resize
        generated_image_tensor = transform(generated_images).unsqueeze(0)  # Add batch dimension
        original_image_tensor = transform(images).unsqueeze(0)  # Add batch dimension
         # Compute loss between generated images and original images
        loss = criterion(generated_image_tensor, original_image_tensor)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
        wandb.log({ "loss": loss})
    # Save the trained model
    torch.save(model.state_dict(), "../checkpionts/trained_model.pth")
    wandb.finish()
    logger.info("Trained model saved")

if __name__ == "__main__":
    from logs.logger import logger
    _main()