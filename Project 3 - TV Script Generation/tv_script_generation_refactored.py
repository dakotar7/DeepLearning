{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import numpy as np\
import torch\
import torch.nn as nn\
import torch.optim as optim\
from torch.utils.data import TensorDataset, DataLoader\
import torch.nn.functional as F\
from collections import Counter\
\
# DataProcessor class for preprocessing tasks\
class DataProcessor:\
    def __init__(self, sequence_length, batch_size):\
        self.sequence_length = sequence_length\
        self.batch_size = batch_size\
\
    def create_lookup_tables(self, text):\
        """Create lookup tables for vocabulary."""\
        word_counts = Counter(text)\
        sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)\
        int_to_vocab = \{i: word for i, word in enumerate(sorted_vocab, 1)\}\
        vocab_to_int = \{word: i for i, word in int_to_vocab.items()\}\
        return vocab_to_int, int_to_vocab\
\
    def token_lookup(self):\
        """Generate a dict to turn punctuation into a token."""\
        return \{\
            '.': '||period||', ',': '||comma||', ';': '||semicolon||',\
            '"': '||quotation_mark||', '!': '||exclamation_point||',\
            '?': '||question_mark||', '(': '||left_parentheses||',\
            ')': '||right_parentheses||', '-': '||dash||', '\\n': '||return||'\
        \}\
\
    def batch_data(self, words):\
        """Batch the data using DataLoader."""\
        words = np.array(words)\
        features, targets = [], []\
\
        for idx in range(0, len(words) - self.sequence_length):\
            feature = words[idx: idx + self.sequence_length]\
            target = words[idx + self.sequence_length]\
            features.append(feature)\
            targets.append(target)\
\
        features, targets = np.array(features), np.array(targets)\
        train_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))\
        return DataLoader(train_data, shuffle=True, batch_size=self.batch_size)\
\
\
# RNNModel class for neural network definition\
class RNNModel(nn.Module):\
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):\
        super(RNNModel, self).__init__()\
        self.output_size = output_size\
        self.hidden_dim = hidden_dim\
        self.n_layers = n_layers\
\
        self.embedding = nn.Embedding(vocab_size, embedding_dim)\
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)\
        self.fc = nn.Linear(hidden_dim, output_size)\
        self.dropout = nn.Dropout(dropout)\
\
    def forward(self, x, hidden):\
        embeds = self.embedding(x)\
        lstm_out, hidden = self.lstm(embeds, hidden)\
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)\
        out = self.fc(self.dropout(lstm_out))\
        return out, hidden\
\
    def init_hidden(self, batch_size):\
        weight = next(self.parameters()).data\
        if torch.cuda.is_available():\
            return (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),\
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())\
        else:\
            return (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),\
                    weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())\
\
\
# Trainer class to handle training\
class Trainer:\
    def __init__(self, model, train_loader, criterion, optimizer, clip=5):\
        self.model = model\
        self.train_loader = train_loader\
        self.criterion = criterion\
        self.optimizer = optimizer\
        self.clip = clip\
\
    def train(self, epochs, show_every_n_batches=100):\
        self.model.train()\
        for epoch in range(1, epochs + 1):\
            hidden = self.model.init_hidden(self.train_loader.batch_size)\
            for batch_i, (inputs, targets) in enumerate(self.train_loader):\
                hidden = tuple([each.data for each in hidden])\
                if torch.cuda.is_available():\
                    inputs, targets = inputs.cuda(), targets.cuda()\
\
                self.model.zero_grad()\
                output, hidden = self.model(inputs, hidden)\
                loss = self.criterion(output, targets)\
                loss.backward()\
\
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)\
                self.optimizer.step()\
\
                if batch_i % show_every_n_batches == 0:\
                    print(f'Epoch: \{epoch\}/\{epochs\}, Loss: \{loss.item()\}')\
\
        return self.model\
\
\
# ScriptGenerator class to generate new TV script\
class ScriptGenerator:\
    def __init__(self, model, int_to_vocab, token_dict, sequence_length):\
        self.model = model\
        self.int_to_vocab = int_to_vocab\
        self.token_dict = token_dict\
        self.sequence_length = sequence_length\
\
    def generate(self, prime_id, predict_len=100, top_k=5):\
        self.model.eval()\
        current_seq = np.full((1, self.sequence_length), prime_id)\
        predicted = [self.int_to_vocab[prime_id]]\
\
        hidden = self.model.init_hidden(1)\
        for _ in range(predict_len):\
            current_seq = torch.LongTensor(current_seq)\
            output, hidden = self.model(current_seq, hidden)\
            p = F.softmax(output, dim=1).data\
            top_i = p.topk(top_k)[1].numpy()[0]\
            word_i = np.random.choice(top_i)\
            word = self.int_to_vocab[word_i]\
            predicted.append(word)\
            current_seq = np.roll(current_seq, -1, 1)\
            current_seq[0, -1] = word_i\
\
        return ' '.join(predicted)\
\
\
# Example of training and generating text\
def main():\
    # Parameters\
    vocab_size = 5000  # Assume vocab size\
    output_size = vocab_size\
    embedding_dim = 128\
    hidden_dim = 256\
    n_layers = 2\
    batch_size = 64\
    sequence_length = 10\
    epochs = 20\
    learning_rate = 0.001\
\
    # Data loading\
    processor = DataProcessor(sequence_length, batch_size)\
    train_loader = processor.batch_data(np.random.randint(0, 5000, 10000))  # Dummy data\
\
    # Model, criterion, optimizer\
    model = RNNModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)\
    criterion = nn.CrossEntropyLoss()\
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)\
\
    # Training\
    trainer = Trainer(model, train_loader, criterion, optimizer)\
    trainer.train(epochs)\
\
    # Generate script\
    script_gen = ScriptGenerator(model, \{i: str(i) for i in range(vocab_size)\}, processor.token_lookup(), sequence_length)\
    print(script_gen.generate(100, predict_len=100))\
\
\
if __name__ == "__main__":\
    main()\
}