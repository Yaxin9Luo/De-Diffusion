# diffuser

This is the my unofficial implementaion of the model from paper "De-Diffusion Makes Text a Strong Cross-Modal Interface" by Chen Wei, Chenxi Liu, Siyuan Qiao, Zhishuai Zhang, Alan Yuille and Jiahui Yu

## Project structure

The directory structure of the project looks like this:

```txt
├── README.md            <- The top-level README for developers using this project.
├── data
│   ├── processed        <- The final, canonical data sets for modeling.
│   └── raw              <- The original, immutable data dump.
│
│
├── logs               <- Trained and serialized models, model predictions, or model summaries
│
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
││
├── DeDiffusion  <- Source code for use in this project.
│   │
│   ├── dataloader             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- model implementations, training script and prediction script
│   │   ├── __init__.py
│   │   ├── decoder.py
|   |   ├── encoder.py
│   │
│── train_model.py   <- script for training the model
│── predict_model.py <- script for predicting from a model
│
└── LICENSE              <- Open-source license if one is chosen
```
