import random
import helper
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tkinter as tk
from model import RNN


class ScriptGenerator:
    def __init__(self):
        self.trained_rnn = None
        self.hyperparameters = None
        self.optimizer = None
        self.criterion = None
        self.model = None
        self.train_loader = None
        self.token_dict = None
        self.int_to_vocab = None
        self.vocab_to_int = None
        self.int_text = None
        self.text = None
        self.batch_size = None
        self.sequence_length = None
        self.output_size = None
        self.input_size = None
        self.show_every_n_batches = None
        self.n_layers = None
        self.hidden_dim = None
        self.embedding_dim = None
        self.learning_rate = None
        self.num_epochs = None
        self.vocab_size = None

        self.train_on_gpu = torch.cuda.is_available()

    def forward_back_prop(self, rnn, optimizer, criterion, inp, target, hidden):
        """
        Forward and backward propagation on the neural network
        :param hidden: hidden state(s) to be passed back into the NN
        :param rnn: The PyTorch Module that holds the neural network
        :param optimizer: The PyTorch optimizer for the neural network
        :param criterion: The PyTorch loss function
        :param inp: A batch of input to the neural network
        :param target: The target output for the batch of input
        :return: The loss and the latest hidden state Tensor
        """

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

    def initialize_data_loader(self, data_dir='./data/Seinfeld_Scripts.txt'):

        self.text = helper.load_data(data_dir)

        helper.preprocess_and_save_data(data_dir, utils.token_lookup, utils.create_lookup_tables)

        self.int_text, self.vocab_to_int, self.int_to_vocab, self.token_dict = helper.load_preprocess()

        self.train_loader = utils.batch_data(self.int_text, self.sequence_length, self.batch_size)

        return

    def initialize_hyperparameters(self):

        # List of field labels
        parameters = {
            "Sequence Length:": 10,
            "Batch Size:": 256,
            "Number of Epochs:": 20,
            "Learning Rate:": 0.002,
            "Embedding Dimension:": 128,
            "Hidden Dimension:": 128,
            "Number of LSTM Layers:": 2,
            "Input (Vocab) Size:": len(self.int_to_vocab),
            "Output Size:": len(self.int_to_vocab),
            "Print Every:": 500,
        }

        def get_all_entry_widgets_text_content(parent_widget, labels_dict):

            children_widgets = parent_widget.winfo_children()
            label_text = None

            for child_widget in children_widgets:

                if child_widget.winfo_class() == 'Label':
                    label_text = child_widget.cget('text')

                if child_widget.winfo_class() == 'Entry':
                    labels_dict[label_text] = float(child_widget.get())

            return labels_dict

        # Create a new window with the title "Set Hyperparameters"
        window = tk.Tk()
        window.title("Set Hyperparameters")
        # window.geometry('500x500')

        # Create a new frame `frm_form` to contain the Label
        # and Entry widgets for entering parameter information
        frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
        # Pack the frame into the window
        frm_form.pack()

        # Loop over the list of field labels
        for idx, text in enumerate(parameters):
            # Create a Label widget with the text from the labels list
            label = tk.Label(master=frm_form, text=text)
            # Create an Entry widget
            entry = tk.Entry(master=frm_form, width=50)
            entry.insert(0, parameters[text])
            # Use the grid geometry manager to place the Label and
            # Entry widgets in the row whose index is idx
            label.grid(row=idx, column=0, sticky="e")
            entry.grid(row=idx, column=1)

        # Create a new frame `frm_buttons` to contain the
        # Submit and Close buttons. This frame fills the
        # whole window in the horizontal direction and has
        # 5 pixels of horizontal and vertical padding.
        frm_buttons = tk.Frame()
        frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)

        # Define the function to close the window upon pressing the "Close" button.
        def Close():
            return window.destroy

        # Create the "Submit" button and pack it to the
        # right side of `frm_buttons`
        btn_submit = tk.Button(master=frm_buttons, text="Submit",
                               command=lambda w=frm_form: get_all_entry_widgets_text_content(w, parameters))
        btn_submit.pack(side=tk.LEFT, padx=10, ipadx=10)

        # Create the "Clear" button and pack it to the
        # right side of `frm_buttons`
        btn_clear = tk.Button(master=frm_buttons, text="Close", command=Close())
        btn_clear.pack(side=tk.RIGHT, ipadx=10, padx=10)

        # Start the application
        window.mainloop()

        self.num_epochs = parameters['Number of Epochs:']
        self.learning_rate = parameters['Learning Rate:']
        self.embedding_dim = parameters['Embedding Dimension:']
        self.hidden_dim = parameters['Hidden Dimension:']
        self.n_layers = parameters['Number of LSTM Layers:']
        self.show_every_n_batches = parameters['Print Every:']
        self.input_size = parameters['Input (Vocab) Size:']
        self.output_size = parameters['Output Size:']
        self.sequence_length = parameters['Sequence Length:']
        self.batch_size = parameters['Batch Size:']

        self.hyperparameters = parameters

        return

    def select_optimizer(self):

        optimizers = {
            'Adam': torch.optim.Adam(self.model.parameters(), lr=self.learning_rate),
            'SGD': torch.optim.SGD(self.model.parameters(), lr=self.learning_rate),
            'RMSprop': torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate),
            'NAdam': torch.optim.NAdam(self.model.parameters(), lr=self.learning_rate)
        }

        # Create a new window with the title "Select Optimizer"
        window = tk.Tk()
        window.title("Select Optimizer")
        # window.geometry('500x500')

        # Create a new frame `frm_form` to contain the Label
        # and Entry widgets for entering parameter information
        frm_form = tk.Frame(relief=tk.SUNKEN, borderwidth=3)
        # Pack the frame into the window
        frm_form.pack()

        opt_list = list(optimizers.keys())

        # control variable
        var = tk.IntVar(frm_form, 0)

        for idx, text in enumerate(opt_list):
            # Create a Radiobutton widget with the text from the optimizers list
            tk.Radiobutton(master=frm_form, text=text, value=idx, variable=var).pack()

        frm_buttons = tk.Frame()
        frm_buttons.pack(fill=tk.X, ipadx=5, ipady=5)

        # Define the function to close the window upon pressing the "Close" button.
        def Close():
            return window.destroy

        # Create the "Submit" button and pack it to the
        # right side of `frm_buttons`
        # btn_submit = tk.Button(master=frm_buttons, text="Submit",
        #                        command=lambda w=frm_form: get_all_entry_widgets_text_content(w, labels))
        # btn_submit.pack(side=tk.LEFT, padx=10, ipadx=10)

        # Create the "Close" button and pack it to the
        # right side of `frm_buttons`
        btn_clear = tk.Button(master=frm_buttons, text="Close", command=Close())
        btn_clear.pack(side=tk.RIGHT, ipadx=10, padx=10)

        # Start the application
        window.mainloop()

        self.optimizer = optimizers[opt_list[var.get()]]

        return

    def load_saved_model(self):
        _, self.vocab_to_int, self.int_to_vocab, self.token_dict = helper.load_preprocess()
        self.trained_rnn = helper.load_model('./save/trained_rnn')

    def run(self, criterion=None, data_path=None):
        self.initialize_data_loader(data_dir=data_path) if data_path else self.initialize_data_loader()
        self.initialize_hyperparameters()
        self.select_optimizer()

        # create model and move to gpu if available
        self.model = RNN(self.vocab_size, self.output_size, self.embedding_dim, self.hidden_dim, self.n_layers,
                         dropout=0.5)
        if self.train_on_gpu:
            self.model.cuda()

        # defining loss and optimization functions for training
        # self.optimizer = optimizer if optimizer else torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = criterion if criterion else nn.CrossEntropyLoss()

        # training the model
        self.trained_rnn = self.train_rnn(self.model, self.batch_size, self.optimizer, self.criterion, self.num_epochs,
                                          self.show_every_n_batches)

        # saving the trained model
        helper.save_model('./save/trained_rnn', self.trained_rnn)
        print('Model Trained and Saved')

    def generate(self, prime_word=None, predict_len=100):
        """
        Generate text using the neural network
        :param prime_word: The word to start the first prediction
        :param predict_len: The length of text to generate
        :return: The generated text
        """
        self.trained_rnn.eval()

        names = ['jerry', 'elaine', 'george', 'kramer']
        random_index = random.randint(0, len(names) - 1)
        prime_word = prime_word if prime_word else names[random_index]

        prime_id = self.vocab_to_int[prime_word + ':']
        pad_word = helper.SPECIAL_WORDS['PADDING']
        pad_value = self.vocab_to_int[pad_word] - 1

        # create a sequence (batch_size=1) with the prime_id
        current_seq = np.full((1, self.sequence_length), pad_value)
        current_seq[-1][-1] = prime_id
        predicted = [self.int_to_vocab[prime_id]]

        for _ in range(predict_len):
            if self.train_on_gpu:
                current_seq = torch.LongTensor(current_seq).cuda()
            else:
                current_seq = torch.LongTensor(current_seq)

            # initialize the hidden state
            hidden = self.trained_rnn.init_hidden(current_seq.size(0))

            # get the output of the rnn
            output, _ = self.trained_rnn(current_seq, hidden)

            # get the next word probabilities
            p = F.softmax(output, dim=1).data
            if self.train_on_gpu:
                p = p.cpu()  # move to cpu

            # use top_k sampling to get the index of the next word
            top_k = 5
            p, top_i = p.topk(top_k)
            top_i = top_i.numpy().squeeze()

            # select the likely next word index with some element of randomness
            p = p.numpy().squeeze()
            word_i = np.random.choice(top_i, p=p / p.sum())

            # retrieve that word from the dictionary
            word = self.int_to_vocab[word_i]
            predicted.append(word)

            if self.train_on_gpu:
                current_seq = current_seq.cpu()  # move to cpu
            # the generated word becomes the next "current sequence" and the cycle can continue
            if self.train_on_gpu:
                current_seq = current_seq.cpu()
            current_seq = np.roll(current_seq, -1, 1)
            current_seq[-1][-1] = word_i

        gen_sentences = ' '.join(predicted)

        # Replace punctuation tokens
        for key, token in self.token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
        gen_sentences = gen_sentences.replace('\n ', '\n')
        gen_sentences = gen_sentences.replace('( ', '(')

        # return all the sentences
        return gen_sentences
