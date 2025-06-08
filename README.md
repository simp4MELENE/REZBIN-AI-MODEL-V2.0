# Rezbin AI Model v2.0

## Description

An image multi-class classification task developed using PyTorch and MobileNetV2, trained on the [TrashNet](https://www.kaggle.com/datasets/feyzazkefe/trashnet) dataset. It is deployed as a containerized microservice, ready to be plugged into any backend for real-time predictions.

The model references the [WasteNet](https://arxiv.org/pdf/2006.05873) paper as a guide to ensure cutting-edge performance and adhere with the best practices.

## Setup (as a microservice)
In this setup, it requires Docker desktop to properly run.

> This setup has only been tried in Windows OS (Windows 10). For other OS's, you might need to approach this setup cautiously.

```bash
# 1. Clone the repository
git clone https://github.com/Rezbin/ai-model-v2.0.git
cd ai-model-v2.0

# 2. Build the Docker image
$env:DOCKER_BUILDKIT=1 # Optional but recommended for fast builds
docker build -t rezbin-ai-v2 .

# 3. Run the Docker image
docker run -p 8000:8000 rezbin-ai-v2
```

## Model Performance
The model used transfer learning, specifically MobileNetV2 which performs best in low resource hardware such as mobile phones. In our model training, we used the WasteNet research paper, however, we did not follow some steps due to resource constraints.

We applied selective layer freezing and experimented with various hyperparameters to optimize model performance on constrained hardware.

![Model Results](./models/MobileNetV2/output.jpg)

Above is the accuracy and loss curves during training of MobileNetV2 on the TrashNet dataset. For the first 50 epochs, we only trained the classifier head. The next 20 epochs trained layers 10 to 18 exclusively, diverging from the gradual unfreezing strategy outlined in the WasteNet paper.

## The Streamlit App
The included Streamlit application serves demonstration purposes only and is not part of the Dockerized microservice. To try out the demo, simply run `streamlit run streamlit_app.py`.

> Install the dependencies in the `requirements.txt` file in order to run properly.

## Developers
Rezbin AI Model version 2.0 is developed by the AI interns team of Rezbin (April - June 2025).