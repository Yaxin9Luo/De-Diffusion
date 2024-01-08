import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from models.encoder import MyModel  
from diffusers import StableDiffusionPipeline  
from torch.utils.data import Subset
import numpy as np
from PIL import Image
import requests

def generate_image(text):
    # Load the pre-trained Stable Diffusion model
    # Make sure to adjust the model path or handle as per your setup
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt = text
    image = pipe(prompt).images[0]  
    return image
def tokens_to_sentence(tokens):
    # Join the tokens into a sentence. Adjust the separator if needed
    sentence = ' '.join(tokens)
    return sentence
#Configurations
device = torch.device('cuda')
learning_rate = 1e-4
batch_size = 6
num_epochs = 10

#Initialize the model
model = MyModel()
model.vit.requires_grad_(False)
model.clip_model.requires_grad_(False)
model.to(device)

#Initialize the text-to-image model
diffusion_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
# Loss function
criterion = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Data loaders
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
# train_dataset = datasets.CIFAR10(root='/root/autodl-tmp/MLops_project_Group18/data/raw', train=True, download=True, transform=transform)
# Load the dataset
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw) # 640x480
# indices = list(range(100))  # indices for the first 100 samples
# subset_train_dataset = Subset(train_dataset, indices)
# train_loader = DataLoader(subset_train_dataset, batch_size=batch_size, shuffle=True)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
# for epoch in range(num_epochs):
model.train()
    # for images, _ in train_loader:
images = image
optimizer.zero_grad()    
        # Generate text prompts using your model
generated_texts = model(images)
sentence = tokens_to_sentence(generated_texts)
        # Generate images from text prompts
generated_images = generate_image(sentence)
# Apply the transform to both PIL images to convert them to tensors and resize
generated_image_tensor = transform(generated_images).unsqueeze(0)  # Add batch dimension
original_image_tensor = transform(images).unsqueeze(0)  # Add batch dimension
        # Compute loss between generated images and original images
loss = criterion(generated_image_tensor, original_image_tensor)

        # Backward pass and optimize
loss.backward()
optimizer.step()

print(f"Epoch [{1+1}/{num_epochs}], Loss: {loss.item():.4f}")
exit()
# Save the trained model
torch.save(model.state_dict(), "/root/autodl-tmp/MLops_project_Group18/logs/trained_model.pth")