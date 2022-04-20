# Imports
import torch
from torchvision import transforms, datasets, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import torch.nn as nn
import time
# import torch.nn.functional as F


# Model built from Scratch
class Net(nn.Module):

    def __init__(self, constant_weight=None):
        super(Net, self).__init__()

        # Define layers of a CNN
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # MaxPool 224 -> 112
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Maxpool 112 -> 56
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        # Maxpool 56 -> 28
        self.linear = nn.Sequential(
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(128, 50)
        )

        # Custom weights code

        if constant_weight:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.constant_(m.weight, constant_weight)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Define forward behavior
        # Block 1
        x = self.block1(x)
        # Block 2
        x = self.block2(x)
        # Block 3
        x = self.block3(x)

        # Reshape output to feed into the linear layers
        x = x.view(x.shape[0], -1)

        # Classifier
        x = self.linear(x)

        return x


class LandmarkClassifier:

    def __init__(self):

        self.use_cuda = torch.cuda.is_available()

        self.classes = None
        self.testing_images = None
        self.training_images = None
        self.loaders = None
        self.test_data = None
        self.train_data = None
        self.test_transforms = None
        self.train_transforms = None
        self.valid_size = None
        self.batch_size = None
        self.num_workers = None

        self.model_scratch = None
        self.model_transfer = None

        self.criterion = {
            'CEL': nn.CrossEntropyLoss()
        }
        return

    def data_loading(self, data_images='landmark_images/'):
        self.training_images = data_images + 'train'
        self.testing_images = data_images + 'test'

        self.num_workers = 0
        self.batch_size = 10
        self.valid_size = 0.2

        self.train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.train_data = datasets.ImageFolder(self.training_images, transform=self.train_transforms)
        self.test_data = datasets.ImageFolder(self.testing_images, transform=self.test_transforms)

        num_train = len(self.train_data)
        indices = list(range(num_train))
        np.random.shuffle(indices)
        split = int(np.floor(self.valid_size * num_train))
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        self.loaders = {'train': DataLoader(self.train_data,
                                            batch_size=self.batch_size,
                                            sampler=train_sampler,
                                            num_workers=self.num_workers),
                        'valid': DataLoader(self.train_data,
                                            batch_size=self.batch_size,
                                            sampler=valid_sampler,
                                            num_workers=self.num_workers),
                        'test': DataLoader(self.test_data,
                                           batch_size=self.batch_size,
                                           num_workers=self.num_workers)}

        self.classes = self.train_data.classes

        return

    def imshow(self, img):
        img = img / 2 + 0.5  # unnormalize
        plt.imshow(np.transpose(img, (1, 2, 0)))  # convert from Tensor image
        return

    def visualize(self):
        dataiter = iter(self.loaders['train'])
        images, labels = dataiter.next()
        images = images.numpy()  # convert images to numpy for display

        # plot the images in the batch, along with the corresponding labels
        fig = plt.figure(figsize=(25, 4))
        # display batch_size number of images
        for idx in np.arange(self.batch_size):
            ax = fig.add_subplot(2, self.batch_size // 2, idx + 1, xticks=[], yticks=[])
            self.imshow(images[idx])
            ax.set_title(self.classes[labels[idx]].split('.')[1].replace('_', ' '))

    def get_optimizer(self, model, option='SGD', learn=0.001, moment=0.9):
        optimizers = {
            'SGD': optim.SGD(model.parameters(), lr=learn, momentum=moment),
            'Adam': optim.Adam(model.parameters(), lr=learn)
        }
        return optimizers[option]

    def custom_weight_init(self, m):

        classname = m.__class__.__name__

        if classname.find('Linear') != -1:
            n = m.in_features
            y = 1.0 / np.sqrt(n)
            m.weight.data.normal_(0, y)
            m.bias.data.fill_(0)

    def default_weight_init(self, m):
        reset_parameters = getattr(m, 'reset_parameters', None)
        if callable(reset_parameters):
            m.reset_parameters()

    def reset_weights(self, model):
        model.apply(self.default_weight_init)
        return

    def train(self, n_epochs, model, save_path, optimizer='SGD'):
        """returns trained model"""
        # initialize tracker for minimum validation loss
        valid_loss_min = np.Inf

        for epoch in range(1, n_epochs + 1):

            start = time.time()

            print(f'Epoch {epoch} started.')
            # initialize variables to monitor training and validation loss
            train_loss = 0.0
            valid_loss = 0.0

            ###################
            # train the model #
            ###################
            # set the module to training mode
            print('Training Begun')
            model.train()
            for batch_idx, (data, target) in enumerate(self.loaders['train']):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                self.get_optimizer(model, optimizer).zero_grad()

                output = model(data)

                loss = self.criterion['CEL'](output, target)

                loss.backward()

                self.get_optimizer(model, optimizer).step()

                train_loss = train_loss + (1 / (batch_idx + 1)) * (loss.data.item() - train_loss)
                # train_loss += loss.item()*data.size(0)

            print(f'Epoch {epoch} training time: {time.time() - start:.3f}')

            ######################
            # validate the model #
            ######################
            val_start = time.time()
            print('Validation Begun')

            # set the model to evaluation mode
            model.eval()
            for batch_idx, (data, target) in enumerate(self.loaders['valid']):
                # move to GPU
                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                output = model(data)

                loss = self.criterion['CEL'](output, target)

                valid_loss = valid_loss + (1 / (batch_idx + 1)) * (loss.data.item() - valid_loss)
                # valid_loss += loss.item()*data.size(0)

            # train_loss = train_loss/len(loaders['train'].dataset)
            # valid_loss = valid_loss/len(loaders['valid'].dataset)

            print(f'Epoch {epoch} validation time: {time.time() - val_start:.3f}')
            print(f'Total Epoch time: {time.time() - start:.3f}')
            # print training/validation statistics
            print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}\n\n'.format(
                epoch,
                train_loss,
                valid_loss
            ))

            if valid_loss <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...\n\n'.format(
                    valid_loss_min,
                    valid_loss))
                torch.save(model.state_dict(), save_path)
                valid_loss_min = valid_loss

        return model

    def train_scratch(self, num_epochs, optim='SGD', custom_weight=False):

        # instantiate the CNN
        self.model_scratch = Net()

        # move tensors to GPU if CUDA is available
        if self.use_cuda:
            self.model_scratch.cuda()

        if custom_weight:
            self.model_scratch.apply(self.custom_weight_init)

        self.model_scratch = self.train(n_epochs=num_epochs,
                                        model=self.model_scratch,
                                        save_path='model_scratch.pt',
                                        optimizer=optim)

        return self.model_scratch

    def train_transfer(self, num_epochs, optim='SGD'):

        self.model_transfer = models.vgg16(pretrained=True)

        for param in self.model_transfer.features.parameters():
            param.requires_grad = False

        n_inputs = self.model_transfer.classifier[6].in_features

        self.model_transfer.classifier[6] = nn.Linear(n_inputs, len(self.classes)*4)
        self.model_transfer.classifier.add_module(nn.ReLU(inplace=True))
        self.model_transfer.classifier.add_module(nn.Dropout(p=0.5))
        self.model_transfer.classifier.add_module(nn.Linear(len(self.classes)*4, len(self.classes)))

        if self.use_cuda:
            self.model_transfer = self.model_transfer.cuda()

        self.model_transfer = self.train(n_epochs=num_epochs,
                                         model=self.model_transfer,
                                         save_path='model_transfer.pt',
                                         optimizer=optim)

        return self.model_transfer

    def test(self, model):

        # monitor test loss and accuracy
        test_loss = 0.
        correct = 0.
        total = 0.

        # set the module to evaluation mode
        model.eval()

        for batch_idx, (data, target) in enumerate(self.loaders['test']):
            # move to GPU
            if self.use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = self.criterion['CEL'](output, target)
            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data.item() - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to true label
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred)).cpu().numpy()))
            total += data.size(0)

        print('Test Loss: {:.6f}\n'.format(test_loss))

        print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
            100. * correct / total, correct, total))

    def test_scratch(self):

        self.model_scratch.load_state_dict(torch.load('model_scratch.pt'))

        return self.test(model=self.model_scratch)

    def test_transfer(self):

        self.model_transfer.load_state_dict(torch.load('model_transfer.pt'))

        return self.test(model=self.model_transfer)

    def predict_landmarks(self, img_path, k):
        img = Image.open(img_path)

        transformed = self.test_transforms(img).float().unsqueeze(0).cuda()

        self.model_transfer.load_state_dict(torch.load('model_transfer.pt'))

        output = self.model_transfer(transformed)

        klargest = torch.topk(output, k=k)

        klargest_names = []

        for i in klargest[1][0]:
            name = self.classes[i].split('.')[1].replace('_', ' ')
            klargest_names.append(name)

        return klargest_names

    def suggest_locations(self, img_path):
        # get landmark predictions
        predicted_landmarks = self.predict_landmarks(img_path, 3)

        plt.imshow(Image.open(img_path))
        plt.show()

        print(
            f'Is this a picture of the \n{predicted_landmarks[0]}, {predicted_landmarks[1]}, or {predicted_landmarks[2]}?')

        return

