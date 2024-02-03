import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize,transforms
from models.encoder import MyModel
from transformers import ViTFeatureExtractor
from datasets import load_dataset
import requests
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
import hydra
from omegaconf import DictConfig
from logs.logger import logger

@hydra.main(config_path="../configs", config_name="main")
def _main(cfg: DictConfig):
    enc = hydra.utils.instantiate(cfg)
    main(enc.predict)

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
def generate_image(text,cfg):
    # Load the pre-trained Stable Diffusion model
    # Make sure to adjust the model path or handle as per your setup
    model_id =cfg['model_id'] 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt = text
    image = pipe(prompt).images[0]  
    image.save(cfg['output_path'])
    return image
def main(cfg: DictConfig):
    # Load the dataset
    url = cfg['url']
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB') # 640x480
    # image_path ="/root/autodl-tmp/MLops_project_Group18/data/raw/111.png"
    # image = Image.open(image_path).convert('RGB')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    processor = BlipProcessor.from_pretrained(cfg["pretrained_model"])
    model = BlipForConditionalGeneration.from_pretrained(cfg["pretrained_model"]).to(device)
    ## Load the model
    # text_model = MyModel().to("cuda")
    # Predict
    prompt_text = cfg["prompt_text"]
    inputs = processor(image,prompt_text, return_tensors="pt").to(device)
    max_length = cfg["max_length"]
    out = model.generate(**inputs,max_length=max_length, min_length=cfg["min_length"], num_beams=cfg["num_beams"])
    text = processor.decode(out[0], skip_special_tokens=True)
    logger.info(processor.decode(out[0], skip_special_tokens=True))
    # predictions = tokens_to_sentence(predict(text_model, image))
    generate_image(text,cfg)
    # Print predictions

if __name__ == "__main__":
    _main()