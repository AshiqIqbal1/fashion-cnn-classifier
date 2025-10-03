# main.py
import torch
from torchvision import datasets, transforms, utils
import matplotlib.pyplot as plt
import numpy as np

from model import FashionCNN
from training import train_model, test_model, save_model, load_model

###############################################################################
#################################### SETUP ####################################
###############################################################################
batch_size = 64
epochs = 10
model_path = '../models/fashion_cnn.pth'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

###############################################################################
############################# DATASET LOADING #################################
###############################################################################

transform = transforms.Compose([
    # Commented Lines were used for dataset augmentation
    # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(10),
    # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ToTensor(),
    # transforms.RandomApply([transforms.Lambda(lambda x: 1.0 - x)], p=0.5),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.FashionMNIST(root='../data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='../data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###############################################################################
################################ NEW MODEL ####################################
###############################################################################

model = FashionCNN().to(device)

###############################################################################
################################# TRAINING ####################################
###############################################################################

import os
if os.path.exists(model_path):
    load_model(model, model_path, device)
    # Train loaded model
    print("Continuing training on existing model...")
    train_model(model, train_loader, device, epochs=15)
    test_model(model, test_loader, device)
    save_model(model, model_path)
else:
    # Train from scratch
    train_model(model, train_loader, device, epochs=15)
    test_model(model, test_loader, device)
    save_model(model, model_path)

###############################################################################
######################## Visualize some predictions ###########################
###############################################################################

classes = ['T-shirt/top','Trouser','Pullover','Dress','Coat',
           'Sandal','Shirt','Sneaker','Bag','Ankle Boot']

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

dataiter = iter(test_loader)
images, labels = next(dataiter)

model.eval()
outputs = model(images.to(device))
_, predicted = torch.max(outputs, 1)

imshow(utils.make_grid(images))
print("Predicted: ", ' '.join(f"{classes[p]}" for p in predicted[:8]))
print("Actual:    ", ' '.join(f"{classes[l]}" for l in labels[:8]))
