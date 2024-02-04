# my_dediffusion

This is the my unofficial implementaion of the model from paper "De-Diffusion Makes Text a Strong Cross-Modal Interface" by Chen Wei, Chenxi Liu, Siyuan Qiao, Zhishuai Zhang, Alan Yuille and Jiahui Yu

## Project structure

The directory structure of the project looks like this:

```txt
├── README.md            <- The top-level README for developers using this project.
├── datasets
│   ├──dataloader.py         <- load the datasets and preprocessing.
│
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
│
├── models  <- Source code for use in this project.
│   ├── __init__.py
│   ├── decoder.py
|   ├── encoder.py
│   │
│── train_model.py   <- script for training the model
│── predict_model.py <- script for predicting from a model
│
├── utils   <- helper files for genric project.
│   ├── __init__.py
│   ├── ckpt.py
│   ├── config.py
│   ├── distributed.py
│   ├── logging.py
│   ├── utils.py
└── LICENSE              <- Open-source license if one is chosen
```
## Installation 
Instructions on how to clone and set up your repository:

* Clone this repo :
```bash
git clone https://github.com/Yaxin9Luo/my_dediffusion.git
cd my_dediffusion

* Create a conda virtual environment and activate it:
```bash
conda create -n dediffusion python=3.11 -y
conda activate dediffusion

* Install the required dependencies:
```bash
pip install -r requirements.txt
