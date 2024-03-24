# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# define the image size
img_size = 48 # 48x48 pixels to match the FER2013 dataset

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# define the path to the dataset
train_dir = 'CK+'

# load the dataset
dataset = datasets.ImageFolder(root = train_dir, transform = transform)

train_size = int(0.7 * len(dataset))
validation_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - validation_size
train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])

# create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# labels mapping
emotion_labels = {0: "anger", 
                  1: "contempt", 
                  2: "disgust", 
                  3: "fear", 
                  4: "happy", 
                  5: "sadness", 
                  6: "surprise"}

# function to display images
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), cmap = 'gray')
    
detaiter = iter(train_loader)
images, labels = next(detaiter)

# display images with their labels
fig = plt.figure(figsize = (10, 4))

# display first 8 images and their labels
for idx in range(8):
    ax = fig.add_subplot(1, 8, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(emotion_labels[labels[idx].item()])
    
plt.show()

# select device for training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# configure the CNN model
class CNN(nn.Module):
    def __init__(self, num_classes = 7):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # Use padding=1 to maintain the spatial dimensions
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        # compute the flattened size after the convolutions and pooling
        self.flattened_size = 128 * 6 * 6 # varies according to input size and architecture
        
        # fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # apply the first two conv layers followed by pooling
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # flatten the output for the dense layer
        x = x.view(x.size(0), -1)
        
        # apply the dense layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
model = CNN(num_classes = 7).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)

# number of epochs
num_epochs = 30

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        # forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")
    
    # validation
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        print(f'Accuracy of the model on the validation images: {100 * correct / total} %')

        # set our model to evaluation mode
model.eval()

# store predictions and true labels
test_predictions = []
test_labels = []

with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        output = model(data)
        _, predicted = torch.max(output, 1)
        
        # store predictions and labels
        test_predictions.extend(predicted.cpu().numpy())
        test_labels.extend(target.cpu().numpy())

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, average='weighted')
recall = recall_score(test_labels, test_predictions, average='weighted')
f1 = f1_score(test_labels, test_predictions, average='weighted')

print(f'Test Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

# ----------------- // ---------------------

# Predict on actual photos

from PIL import Image

# load the image
image_path = 'photos/img4.png'
image = Image.open(image_path).convert('L')

# preprocess the image
transform = transforms.Compose([
    transforms.Resize((48,48)),
    transforms.ToTensor(),
])

# apply the transformation to the image
image = transform(image).unsqueeze(0)
image = image.to(device)

# display the transformed image
np_img = image.squeeze().numpy()
plt.imshow(np_img, cmap = 'gray')
plt.title('Transformed Image')
plt.axis('off')
plt.show()

# predict the mood
with torch.no_grad():
    model.eval()
    outputs = model(image)
    _, predicted = torch.max(outputs, 1)
    predicted_mood = emotion_labels[predicted.item()]
    
print(f"The predicted mood is: {predicted_mood}")