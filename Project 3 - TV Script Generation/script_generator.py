import helper
import utils
import torch
import numpy as np
from model import RNN


class ScriptGenerator:
    def __init__(self, data_dir='./data/Seinfeld_Scripts.txt', sequence_length=10, batch_size=256):
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        self.text = helper.load_data(data_dir)

        helper.preprocess_and_save_data(data_dir, utils.token_lookup, utils.create_lookup_tables)

        self.int_text, self.vocab_to_int, self.int_to_vocab, self.token_dict = helper.load_preprocess()

        self.train_on_gpu = torch.cuda.is_available()

        self.train_loader = utils.batch_data(self.int_text, self.sequence_length, self.batch_size)

    def forward_back_prop(self, rnn, optimizer, criterion, inp, target, hidden):
        """
        Forward and backward propagation on the neural network
        :param rnn: The PyTorch Module that holds the neural network
        :param optimizer: The PyTorch optimizer for the neural network
        :param criterion: The PyTorch loss function
        :param inp: A batch of input to the neural network
        :param target: The target output for the batch of input
        :return: The loss and the latest hidden state Tensor
        """

        # TODO: Implement Function

        # move data to GPU, if available
        if self.train_on_gpu:
            inp, target = inp.cuda(), target.cuda()

        # perform backpropagation and optimization
        h = tuple([each.data for each in hidden])

        # zero accumulated gradients
        rnn.zero_grad()

        # get the output from the model
        output, h = rnn(inp, h)

        # calculate the loss and perform backprop
        loss = criterion(output, target.long())
        loss.backward()
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        # nn.utils.clip_grad_norm_(rnn.parameters(), clip)
        optimizer.step()

        # return the loss over a batch and the hidden state produced by our model
        return loss.item(), h

    def train_rnn(self, rnn, batch_size, optimizer, criterion, n_epochs, show_every_n_batches=100):
        batch_losses = []

        rnn.train()

        print("Training for %d epoch(s)..." % n_epochs)
        for epoch_i in range(1, n_epochs + 1):

            # initialize hidden state
            hidden = rnn.init_hidden(batch_size)

            for batch_i, (inputs, labels) in enumerate(self.train_loader, 1):

                # make sure you iterate over completely full batches, only
                n_batches = len(self.train_loader.dataset) // batch_size
                if batch_i > n_batches:
                    break

                # forward, back prop
                loss, hidden = self.forward_back_prop(rnn, optimizer, criterion, inputs, labels, hidden)
                # record loss
                batch_losses.append(loss)

                # printing loss stats
                if batch_i % show_every_n_batches == 0:
                    print('Epoch: {:>4}/{:<4}  Loss: {}\n'.format(
                        epoch_i, n_epochs, np.average(batch_losses)))
                    batch_losses = []

        # returns a trained rnn
        return rnn