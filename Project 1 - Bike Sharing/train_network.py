from NeuralNetwork import NeuralNetwork
from data_prep import DataPrep
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd


class TrainNetwork:
    def __init__(self, iterations=5000, learning_rate=0.7, hidden_nodes=10, output_nodes=1):
        self.network = None
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.loader = DataPrep()

        self.test_features, self.test_targets = self.loader.test_loader()
        self.train_features, self.train_targets = self.loader.train_loader()
        self.val_features, self.val_targets = self.loader.val_loader()

    def train(self):
        def MSE(y, Y):
            return np.mean((y - Y) ** 2)

        N_i = self.train_features.shape[1]
        self.network = NeuralNetwork(N_i, self.hidden_nodes, self.output_nodes, self.learning_rate)

        losses = {'train': [], 'validation': []}
        for ii in range(self.iterations):
            # Go through a random batch of 128 records from the training data set
            batch = np.random.choice(self.train_features.index, size=128)
            X, y = self.train_features.loc[batch].values, self.train_targets.loc[batch]['cnt']

            self.network.train(X, y)

            # Printing out the training progress
            train_loss = MSE(self.network.run(self.train_features).T, self.train_targets['cnt'].values)
            val_loss = MSE(self.network.run(self.val_features).T, self.val_targets['cnt'].values)
            sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(self.iterations)) \
                             + "% ... Training loss: " + str(train_loss)[:5] \
                             + " ... Validation loss: " + str(val_loss)[:5])
            sys.stdout.flush()

            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)

        return losses

    def visualize_loss(self, losses):
        plt.plot(losses['train'], label='Training loss')
        plt.plot(losses['validation'], label='Validation loss')
        plt.legend()
        _ = plt.ylim()

        return

    def visualize_predictions(self):
        fig, ax = plt.subplots(figsize=(8, 4))

        mean, std = self.loader.scaled_features['cnt']
        predictions = self.network.run(self.test_features).T * std + mean
        ax.plot(predictions[0], label='Prediction')
        ax.plot(self.test_targets['cnt'].values, label='Data')
        ax.set_xlim(right=len(predictions))
        ax.legend()

        dates = pd.to_datetime(self.loader.rides.loc[self.loader.test_data.index]['dteday'])
        dates = dates.apply(lambda d: d.strftime('%b %d'))
        ax.set_xticks(np.arange(len(dates))[12::24])
        _ = ax.set_xticklabels(dates[12::24], rotation=45)

        return

    def train_network(self):
        losses = self.train()
        self.visualize_loss(losses)
        self.visualize_predictions()

        return

