
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torch import nn, optim
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import time

class LandmarkNet(nn.Module):
    """
    Neural Network model for landmark prediction.
    """
    def __init__(self, constant_weight=None):
        super(LandmarkNet, self).__init__()
        
        # Define CNN layers
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 224 -> 112
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112 -> 56
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 56 -> 28
        )
        
        # Define fully connected layers
        self.linear = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 50)
        )

        # Initialize weights if required
        if constant_weight:
            self._initialize_weights(constant_weight)
    
    def _initialize_weights(self, constant_weight):
        """
        Initialize weights of the model with a constant value.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, constant_weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Define the forward pass through the network.
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.linear(x)
        return x


class Trainer:
    """
    Trainer class to handle model training and testing.
    """
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, epochs=10):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs

    def train(self):
        """
        Train the model for a given number of epochs.
        """
        self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for images, labels in self.train_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
            
            print(f"Epoch [{epoch+1}/{self.epochs}], Loss: {running_loss/len(self.train_loader):.4f}")

    def test(self):
        """
        Test the model on the validation set.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in self.valid_loader:
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy


class LandmarkPredictor:
    """
    Predictor class to load models and make predictions.
    """
    def __init__(self, model_path, classes):
        self.model = LandmarkNet()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.classes = classes
        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def predict_landmarks(self, img_path, k=3):
        """
        Predict top K landmarks for a given image.
        """
        try:
            img = Image.open(img_path)
            transformed = self.test_transforms(img).unsqueeze(0)
            output = self.model(transformed)
            topk = torch.topk(output, k=k)
            return [self.classes[i] for i in topk[1][0]]
        except Exception as e:
            print(f"Error in predicting landmarks: {e}")
            return []

    def suggest_locations(self, img_path):
        """
        Display image and suggest landmark predictions.
        """
        predicted_landmarks = self.predict_landmarks(img_path, 3)
        if predicted_landmarks:
            img = Image.open(img_path)
            plt.imshow(img)
            plt.show()
            print(f"Is this a picture of {', '.join(predicted_landmarks)}?")


def main():
    # Example usage
    model_path = 'model_transfer.pt'
    classes = ['Landmark1', 'Landmark2', 'Landmark3']  # Replace with actual classes

    predictor = LandmarkPredictor(model_path, classes)
    predictor.suggest_locations('sample_image.jpg')


if __name__ == "__main__":
    main()
