import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models.encoder import MyModel
from transformers import ViTFeatureExtractor
from datasets import load_dataset
import requests
from diffusers import StableDiffusionPipeline


# Define a function to preprocess the images
def preprocess_images(images):
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-large-patch16-224-in21k')
    return feature_extractor(images=images, return_tensors="pt")

# Define a function to load and preprocess the dataset
def get_preprocessed_dataset(dataset_name, split='test'):
    dataset = load_dataset(dataset_name, split=split)
    # Assuming the dataset has an "image" column
    images = [Image.open(image_path).convert("RGB") for image_path in dataset['image']]
    return preprocess_images(images)

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
    image = Image.open(requests.get(url, stream=True).raw) # 640x480
    # Load the model
    text_model = MyModel()
    # Predict
    predictions = tokens_to_sentence(predict(text_model, image))
    generate_image(predictions)
    # Print predictions
    print(predictions)

if __name__ == "__main__":
    main()