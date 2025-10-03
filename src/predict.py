# predict.py
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model import FashionCNN
from training import load_model
import sys

###############################################################################
################################## SETUP ######################################
###############################################################################

device = torch.device("cpu")
classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
           'Sandal','Shirt','Sneaker','Bag','Ankle Boot']

model = FashionCNN()
load_model(model, '../models/fashion_cnn.pth', device)

###############################################################################
############################ DATA PREPROCESSING ###############################
###############################################################################
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')
    print("Original size:", img.size, "Mode:", img.mode)

    transform = transforms.Compose([
        transforms.Resize((28,28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    img_tensor = transform(img)

    return img_tensor.unsqueeze(0)

def predict(image_path):
    model.eval()
    image = preprocess_image(image_path).to(device)
    
    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        top3_prob, top3_idx = torch.topk(probs, 3) 

    print(f"Top 3 predictions for {image_path}:")
    for i in range(3):
        print(f"{classes[top3_idx[0][i]]}: {top3_prob[0][i]:.2f}")

    img = Image.open(image_path)
    plt.imshow(img, cmap='gray')
    plt.title(f"Top prediction: {classes[top3_idx[0][0]]} with {top3_prob[0][0]*100:.2f}% Confidence")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    predict(image_path)
