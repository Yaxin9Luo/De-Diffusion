import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize,transforms
from models.encoder import MyModel
from transformers import ViTFeatureExtractor
from datasets import load_dataset
import requests
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration

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
def generate_image(text):
    # Load the pre-trained Stable Diffusion model
    # Make sure to adjust the model path or handle as per your setup
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda"
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    prompt = text
    image = pipe(prompt).images[0]  
    image.save("/root/autodl-tmp/MLops_project_Group18/reports/figures/test1.png")
    return image
def main():
    # Load the dataset
    url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
    image = Image.open(requests.get(url, stream=True).raw).convert('RGB') # 640x480
    # image_path ="/root/autodl-tmp/MLops_project_Group18/data/raw/111.png"
    # image = Image.open(image_path).convert('RGB')

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
    ## Load the model
    # text_model = MyModel().to("cuda")
    # Predict
    prompt_text = "a photography of"
    inputs = processor(image,prompt_text, return_tensors="pt").to("cuda")
    max_length = 75
    out = model.generate(**inputs,max_length=max_length, min_length=40, num_beams=5)
    text = processor.decode(out[0], skip_special_tokens=True)
    print(processor.decode(out[0], skip_special_tokens=True))
    # predictions = tokens_to_sentence(predict(text_model, image))
    generate_image(text)
    # Print predictions

if __name__ == "__main__":
    main()