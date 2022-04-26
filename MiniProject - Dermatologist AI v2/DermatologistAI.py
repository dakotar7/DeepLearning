import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class DAI:
    def __init__(self, data_path='/data/'):
        self.std_test = None
        self.mean_test = None
        self.std_valid = None
        self.mean_valid = None
        self.std_train = None
        self.mean_train = None
        self.use_cuda = torch.cuda.is_available()

        self.train_path = data_path + 'train'
        self.valid_path = data_path + 'valid'
        self.test_path = data_path + 'test'

        self.pixel_averaging()

        self.train_transform = transforms.Compose([transforms.Resize((256, 256)),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.RandomRotation(10),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(self.mean_train, self.std_train)])

        self.valid_transform = transforms.Compose([transforms.Resize((256, 256)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(self.mean_valid, self.std_valid)])

        self.test_transform = transforms.Compose([transforms.Resize((256, 256)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(self.mean_test, self.std_test)])

        self.train_data = datasets.ImageFolder(root=self.train_path, transform=self.train_transform)
        self.valid_data = datasets.ImageFolder(root=self.valid_path, transform=self.valid_transform)
        self.test_data = datasets.ImageFolder(root=self.test_path, transform=self.test_transform)

        self.trainloader = torch.utils.data.DataLoader(self.train_data,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       num_workers=0)

        self.validloader = torch.utils.data.DataLoader(self.valid_data,
                                                       batch_size=32,
                                                       shuffle=True,
                                                       num_workers=0)

        self.testloader = torch.utils.data.DataLoader(self.test_data,
                                                      batch_size=32,
                                                      shuffle=False,
                                                      num_workers=0)

        self.model = models.efficientnet_b7(pretrained=True)

        self.num_features = self.model.classifier[-1].in_features

        self.model.classifier[-1] = nn.Linear(self.num_features, 2)

        if self.use_cuda:
            self.model.cuda()

        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optim.Adam(self.model.classifier.parameters, lr=0.001)

        # self.step_lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=7, gamma=0.1)

        return

    def pixel_averaging(self):
        train_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

        train_set = datasets.ImageFolder(root=self.train_path, transform=train_transform)
        valid_set = datasets.ImageFolder(root=self.valid_path, transform=train_transform)
        test_set = datasets.ImageFolder(root=self.test_path, transform=train_transform)

        train_loader = DataLoader(
            train_set,
            batch_size=20,
            shuffle=True,
            num_workers=0
        )

        valid_loader = DataLoader(
            valid_set,
            batch_size=20,
            shuffle=True,
            num_workers=0
        )

        test_loader = DataLoader(
            test_set,
            batch_size=20,
            shuffle=True,
            num_workers=0
        )

        def calc_mean_std(loader):
            channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

            for data, _ in tqdm(loader):
                channels_sum += torch.mean(data, dim=[0, 2, 3])
                channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
                num_batches += 1

            mean = channels_sum / num_batches
            std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

            return mean, std

        self.mean_train, self.std_train = calc_mean_std(train_loader)

        self.mean_valid, self.std_valid = calc_mean_std(valid_loader)

        self.mean_test, self.std_test = calc_mean_std(test_loader)

        return

    def train_model(self, num_epochs=5, SK=True):
        valid_loss_min = -np.Inf
        valid_loss = []
        valid_accuracy = []
        train_loss = []
        train_accuracy = []
        total_steps = len(self.trainloader)

        for e in range(1, num_epochs+1):
            running_loss, num_correct, total = 0.0, 0, 0

            # Training Pass
            for batch_i, (data, target) in enumerate(self.trainloader):
                if SK:
                    target = np.array([1 if i == 2 else 0 for i in target.numpy()])
                    target = torch.tensor(target.astype(np.longlong))
                else:
                    target = np.array([1 if i == 0 else 0 for i in target.numpy()])
                    target = torch.tensor(target.astype(np.longlong))

                if self.use_cuda:
                    data, target = data.cuda(), target.cuda()

                self.optimizer.zero_grad()

                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                _, pred = torch.max(outputs, dim=1)

                num_correct += torch.sum(pred == target).item()
                total += target.size(0)

                if batch_i % 20 == 0:
                    print(f'Epoch [{e}/{num_epochs}], Step [{batch_i}/{total_steps}], Loss: {loss.item():.4f}')

            train_accuracy.append(num_correct/total * 100)
            train_loss.append(running_loss/total_steps)

            print(f'Train Loss: {np.mean(train_loss):.4f}, Train Accuracy: {train_accuracy[-1]:.4f}')

            # Validation Pass
            batch_loss, total_val, num_correct_val = 0, 0, 0

            with torch.no_grad():
                self.model.eval()

                for data_v, target_v in self.validloader:
                    if SK:
                        target_v = np.array([1 if i == 2 else 0 for i in target_v.numpy()])
                        target_v = torch.tensor(target_v.astype(np.longlong))
                    else:
                        target_v = np.array([1 if i == 0 else 0 for i in target_v.numpy()])
                        target_v = torch.tensor(target_v.astype(np.longlong))

                    if self.use_cuda:
                        data_v, target_v = data_v.cuda(), target_v.cuda()

                    outputs_v = self.model(data_v)
                    loss_v = self.criterion(outputs_v, target_v)
                    batch_loss += loss_v.item()
                    _, pred_v = torch.max(outputs_v, dim=1)
                    num_correct_val += torch.sum(pred_v == target_v).item()
                    total_val += target_v.size(0)

                valid_accuracy.append(num_correct_val/total_val * 100)
                valid_loss.append(batch_loss/len(self.validloader))

                print(f'validation loss: {np.mean(valid_loss):.4f}, validation accuracy: {valid_accuracy[-1]:.4f}\n')

                if batch_loss < valid_loss_min:
                    valid_loss_min = batch_loss
                    if SK:
                        torch.save(self.model.state_dict(), 'model_SK.pt')
                    else:
                        torch.save(self.model.state_dict(), 'model_MM.pt')
                    print('Validation loss decreased, saving model.')

            self.model.train()

        return train_loss, valid_loss

    def train_SK(self):
        train_loss, valid_loss = self.train_model(num_epochs=5, SK=True)

        return train_loss, valid_loss

    def train_MM(self):
        train_loss, valid_loss = self.train_model(num_epochs=5, SK=False)

        return train_loss, valid_loss

    def plot_results(self, train_loss, valid_loss):
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.set_title("Train-Validation Loss")
        ax.plot(train_loss, label='train')
        ax.plot(valid_loss, label='validation')
        ax.set_xlabel('num_epochs', fontsize=12)
        ax.set_ylabel('loss', fontsize=12)
        ax.legend(loc='best')

        plt.show()

        return

