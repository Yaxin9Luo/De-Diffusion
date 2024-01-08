import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from models.encoder import MyModel
def preprocess_image(image_path):
    image = Image.open(image_path)
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MyModel().to(device)
    model.eval()  # Set to evaluation mode if not training

    # Preprocess the image
    image_path = '/root/autodl-tmp/MLops_project_Group18/data/raw/train2014/COCO_train2014_000000000064.jpg' 
    print("Image loaded successfully.")

    preprocessed_image = preprocess_image(image_path).to(device)

    # Forward pass through the model
    with torch.no_grad():  # No need to track gradients for inference
        human_readable_text = model(preprocessed_image)

    # Example: Print the output
    for text in human_readable_text:
        print(text)
if __name__ == "__main__":
    main()