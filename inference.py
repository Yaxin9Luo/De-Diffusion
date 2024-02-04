import sys
import os
import torch
from PIL import Image
import requests
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import argparse
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

# Define the prediction function
def predict(text_model, image):
    text_model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient tracking
        predictions = text_model(image)
    return predictions
def tokens_to_sentence(tokens):
    # Join the tokens into a sentence. Adjust the separator if needed
    sentence = ' '.join(tokens)
    return sentence
def generate_image(text,__C):
    # Load the pre-trained Stable Diffusion model
    # Make sure to adjust the model path or handle as per your setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(__C.training.pre_trained_diffusion_model, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt = text
    image = pipe(prompt).images[0]  
    image.save(__C.inference.output_path)
    return image
def inference(__C):
    # Load the dataset
    url = __C.inference.img_url
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB') # 640x480
    # image_path ="/root/autodl-tmp/MLops_project_Group18/data/raw/111.png"
    # image = Image.open(image_path).convert('RGB')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(__C.inference.pretrained_blip_model)
    model = BlipForConditionalGeneration.from_pretrained(__C.inference.pretrained_blip_model).to(device)
    ## Load the model
    inputs = processor(image, return_tensors="pt").to(device)
    max_length = __C.inference.max_length
    out = model.generate(**inputs,max_length=max_length, min_length=__C.inference.min_length, num_beams=__C.inference.num_beams)
    text = processor.decode(out[0], skip_special_tokens=True)
    print(text)
    generate_image(text,__C)

def load_cfg_from_cfg_file(file_path):
    with open(file_path, 'r') as file:
        cfg_dict = yaml.safe_load(file)
    return CfgNode(cfg_dict)

def main():
    parser = argparse.ArgumentParser(description="Run predictions.")
    parser.add_argument('--config', type=str, required=True, help="Path to the configuration YAML file.")
    args = parser.parse_args()

    # Load configuration
    __C = load_cfg_from_cfg_file(args.config)

    # Run main function with loaded configuration
    inference(__C)

if __name__ == "__main__":
    main()