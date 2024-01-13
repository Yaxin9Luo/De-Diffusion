import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(project_root)
from PIL import Image
import hydra
from omegaconf import DictConfig
import requests
from DeDiffuser.models.encoder import MyModel

def test_model():
    @hydra.main(config_path="../configs", config_name="main")
    def _main(cfg: DictConfig):
        enc = hydra.utils.instantiate(cfg)
        print(enc)
        model = MyModel(enc.model)
        url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
        image = Image.open(requests.get(url, stream=True).raw)
        output = model(image)
        assert isinstance(output, list) and len(output) > 0

if __name__ == "__main__":
    test_model()



