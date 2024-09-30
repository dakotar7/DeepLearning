{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import os\
import pickle\
import torch\
import numpy as np\
import torch.nn as nn\
import pandas as pd\
from sklearn.utils import shuffle\
from nltk.corpus import stopwords\
from bs4 import BeautifulSoup\
import re\
import nltk\
from torch.utils.data import DataLoader, TensorDataset\
\
# Data processing class\
class DataProcessor:\
    def __init__(self, data_dir, cache_dir, cache_file="preprocessed_data.pkl", vocab_size=5000):\
        self.data_dir = data_dir\
        self.cache_dir = cache_dir\
        self.cache_file = cache_file\
        self.vocab_size = vocab_size\
        nltk.download("stopwords", quiet=True)\
    \
    def read_imdb_data(self):\
        data = \{\}\
        labels = \{\}\
        for data_type in ['train', 'test']:\
            data[data_type] = \{\}\
            labels[data_type] = \{\}\
            for sentiment in ['pos', 'neg']:\
                data[data_type][sentiment] = []\
                labels[data_type][sentiment] = []\
                path = os.path.join(self.data_dir, data_type, sentiment, '*.txt')\
                files = glob.glob(path)\
                for f in files:\
                    with open(f) as review:\
                        data[data_type][sentiment].append(review.read())\
                        labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)\
        return data, labels\
\
    def prepare_data(self, data, labels):\
        data_train = data['train']['pos'] + data['train']['neg']\
        data_test = data['test']['pos'] + data['test']['neg']\
        labels_train = labels['train']['pos'] + labels['train']['neg']\
        labels_test = labels['test']['pos'] + labels['test']['neg']\
        data_train, labels_train = shuffle(data_train, labels_train)\
        data_test, labels_test = shuffle(data_test, labels_test)\
        return data_train, data_test, labels_train, labels_test\
\
    def preprocess_data(self, data_train, data_test, labels_train, labels_test):\
        cache_data = None\
        if self.cache_file is not None:\
            try:\
                with open(os.path.join(self.cache_dir, self.cache_file), "rb") as f:\
                    cache_data = pickle.load(f)\
            except:\
                pass\
        \
        if cache_data is None:\
            words_train = [self.review_to_words(review) for review in data_train]\
            words_test = [self.review_to_words(review) for review in data_test]\
            if self.cache_file is not None:\
                cache_data = dict(words_train=words_train, words_test=words_test,\
                                  labels_train=labels_train, labels_test=labels_test)\
                with open(os.path.join(self.cache_dir, self.cache_file), "wb") as f:\
                    pickle.dump(cache_data, f)\
        else:\
            words_train, words_test, labels_train, labels_test = cache_data.values()\
        \
        return words_train, words_test, labels_train, labels_test\
\
    def review_to_words(self, review):\
        stemmer = nltk.PorterStemmer()\
        text = BeautifulSoup(review, "html.parser").get_text() \
        text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())\
        words = text.split()\
        words = [w for w in words if w not in stopwords.words("english")]\
        words = [stemmer.stem(w) for w in words]\
        return words\
\
    def build_dict(self, data):\
        word_count = \{\}\
        for review in data:\
            for word in review:\
                word_count[word] = word_count.get(word, 0) + 1\
        \
        sorted_words = sorted(word_count, key=word_count.get, reverse=True)\
        word_dict = \{word: idx + 2 for idx, word in enumerate(sorted_words[:self.vocab_size - 2])\}\
        return word_dict\
\
    def save_word_dict(self, word_dict, data_dir):\
        os.makedirs(data_dir, exist_ok=True)\
        with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:\
            pickle.dump(word_dict, f)\
\
    def convert_and_pad(self, word_dict, sentence, pad=500):\
        NOWORD = 0\
        INFREQ = 1\
        working_sentence = [NOWORD] * pad\
        for word_index, word in enumerate(sentence[:pad]):\
            working_sentence[word_index] = word_dict.get(word, INFREQ)\
        return working_sentence, min(len(sentence), pad)\
\
    def convert_and_pad_data(self, word_dict, data, pad=500):\
        result = []\
        lengths = []\
        for sentence in data:\
            converted, length = self.convert_and_pad(word_dict, sentence, pad)\
            result.append(converted)\
            lengths.append(length)\
        return np.array(result), np.array(lengths)\
\
# Model class\
class SentimentModel(nn.Module):\
    def __init__(self, embedding_dim, hidden_dim, vocab_size):\
        super(SentimentModel, self).__init__()\
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)\
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)\
        self.dense = nn.Linear(hidden_dim, 1)\
        self.sigmoid = nn.Sigmoid()\
\
    def forward(self, x):\
        x = x.t()\
        lengths = x[0, :]\
        reviews = x[1:, :]\
        embeds = self.embedding(reviews)\
        lstm_out, _ = self.lstm(embeds)\
        out = self.dense(lstm_out)\
        out = out[lengths - 1, range(len(lengths))]\
        return self.sigmoid(out.squeeze())\
\
# Training class\
class ModelTrainer:\
    def __init__(self, model, device, learning_rate=0.001):\
        self.model = model.to(device)\
        self.device = device\
        self.criterion = nn.BCELoss()\
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)\
\
    def train(self, train_loader, epochs=10):\
        self.model.train()\
        for epoch in range(epochs):\
            total_loss = 0\
            for batch in train_loader:\
                batch_X, batch_y = batch\
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)\
                self.model.zero_grad()\
                output = self.model(batch_X)\
                loss = self.criterion(output, batch_y.float())\
                loss.backward()\
                self.optimizer.step()\
                total_loss += loss.item()\
            print(f"Epoch: \{epoch+1\}/\{epochs\}, Loss: \{total_loss/len(train_loader)\}")\
\
# Evaluation class\
class ModelEvaluator:\
    def __init__(self, model, device):\
        self.model = model.to(device)\
        self.device = device\
\
    def evaluate(self, test_loader):\
        self.model.eval()\
        total_correct = 0\
        total_samples = 0\
        with torch.no_grad():\
            for batch_X, batch_y in test_loader:\
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)\
                output = self.model(batch_X)\
                predictions = (output > 0.5).int()\
                total_correct += (predictions == batch_y).sum().item()\
                total_samples += len(batch_y)\
        accuracy = total_correct / total_samples\
        print(f"Accuracy: \{accuracy:.4f\}")\
        return accuracy\
\
# Deployment class (for SageMaker)\
class SageMakerDeployer:\
    def __init__(self, role, model_data, instance_type="ml.m4.xlarge"):\
        self.role = role\
        self.model_data = model_data\
        self.instance_type = instance_type\
\
    def deploy(self, entry_point, source_dir):\
        from sagemaker.pytorch import PyTorchModel\
        from sagemaker.predictor import RealTimePredictor\
        \
        class StringPredictor(RealTimePredictor):\
            def __init__(self, endpoint_name, sagemaker_session):\
                super().__init__(endpoint_name, sagemaker_session, content_type='text/plain')\
\
        model = PyTorchModel(model_data=self.model_data,\
                             role=self.role,\
                             framework_version='0.4.0',\
                             entry_point=entry_point,\
                             source_dir=source_dir,\
                             predictor_cls=StringPredictor)\
\
        return model.deploy(initial_instance_count=1, instance_type=self.instance_type)\
\
# Sample main function to use the classes\
def main():\
    data_processor = DataProcessor(data_dir="../data/aclImdb", cache_dir="../cache")\
    data, labels = data_processor.read_imdb_data()\
    train_X, test_X, train_y, test_y = data_processor.prepare_data(data, labels)\
\
    # Preprocess the data\
    train_X, test_X, train_y, test_y = data_processor.preprocess_data(train_X, test_X, train_y, test_y)\
    word_dict = data_processor.build_dict(train_X)\
    data_processor.save_word_dict(word_dict, data_dir="../data/pytorch")\
\
    # Convert to padded format\
    train_X, train_X_len = data_processor.convert_and_pad_data(word_dict, train_X)\
    test_X, test_X_len = data_processor.convert_and_pad_data(word_dict, test_X)\
\
    # Prepare PyTorch Datasets\
    train_dataset = TensorDataset(torch.tensor(train_X), torch.tensor(train_y))\
    test_dataset = TensorDataset(torch.tensor(test_X), torch.tensor(test_y))\
\
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)\
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)\
\
    # Model Initialization and Training\
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\
    model = SentimentModel(embedding_dim=32, hidden_dim=100, vocab_size=5000)\
\
    trainer = ModelTrainer(model, device)\
    trainer.train(train_loader, epochs=10)\
\
    # Model Evaluation\
    evaluator = ModelEvaluator(model, device)\
    evaluator.evaluate(test_loader)\
\
if __name__ == "__main__":\
    main()\
}