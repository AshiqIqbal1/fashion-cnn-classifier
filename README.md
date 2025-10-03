# FashionMNIST CNN Classifier

A convolutional neural network (CNN) built with **PyTorch** to classify clothing items from the [FashionMNIST dataset](https://github.com/zalandoresearch/fashion-mnist).  
Supports training, evaluation, and prediction on custom images.

---

## Features
- CNN built from scratch in PyTorch
- Training and evaluation pipeline
- Data augmentation support
- Model saving/loading
- Prediction script for custom images

---

## Setup

Clone the repo and install dependencies:

```bash
git clone https://github.com/yourusername/fashion-mnist-cnn.git
cd fashion-mnist-cnn
pip install -r requirements.txt
```

---

## Training
To train the model:
```bash
python main.py
```
- Downloads FashionMNIST dataset automatically
- Trains for 5 epochs (adjustable in main.py)
- Saves model weights to models/fashion_cnn.pth

---

## Predicting
You can predict any grayscale image (preferably 28×28 px).

Run the prediction script with an image path:
```bash
python predict.py path/to/image.png
```

## Output
