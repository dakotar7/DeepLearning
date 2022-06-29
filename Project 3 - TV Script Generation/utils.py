from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    # TODO: Implement Function
    # words_no_punct = ' '.join([c for c in text if c not in punctuation])

    # words = words_no_punct.split()

    word_counts = Counter(text)

    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)

    int_to_vocab = {i: word for i, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: i for i, word in int_to_vocab.items()}

    # return tuple
    return (vocab_to_int, int_to_vocab)


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenized dictionary where the key is the punctuation and the value is the token
    """
    # TODO: Implement Function
    punctuation_tokens = {
        '.': '||period||',
        ',': '||comma||',
        ';': '||semicolon||',
        '"': '||quotation_mark||',
        '!': '||exclamation_point||',
        '?': '||question_mark||',
        '(': '||left_parentheses||',
        ')': '||right_parentheses||',
        '-': '||dash||',
        '\n': '||return||'
    }

    return punctuation_tokens


def batch_data(words, sequence_length, batch_size):
    """
    Batch the neural network data using DataLoader
    :param words: The word ids of the TV scripts
    :param sequence_length: The sequence length of each batch
    :param batch_size: The size of each batch; the number of sequences in a batch
    :return: DataLoader with batched data
    """
    # TODO: Implement function

    words = np.array(words)

    total_chars_batch = batch_size * sequence_length

    n_batches = words.shape[0] // total_chars_batch

    words = words[:total_chars_batch * n_batches]

    features = []

    targets = []

    for idx in range(0, len(words) - sequence_length):
        feature = words[idx:idx + sequence_length]

        target = words[idx + sequence_length]

        features.append(feature)
        targets.append(target)

    features, targets = np.array(features), np.array(targets)

    train_data = TensorDataset(torch.from_numpy(features), torch.from_numpy(targets))
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    return train_loader

