import random
import torch
import torch.nn as nn
from collections import Counter

import json
import codecs
import fasttext

import numpy as np
import argparse


def parse_args():

    # construct the argument parser and parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True,
    	                help="path to input text")

    parser.add_argument("-m", "--method", required=True,
    	                help="Method to use - NER, NER_Entity, MOEE,Entity_MOEE, NER_ENTITY_MOEE")

    parser.add_argument("--sent_vocab", required=True,
    	                help="Path to sentence vocab json file")

    parser.add_argument("--ner_tag_vocab", required=True,
    	                help="Path to NER tag vocab json file")

    parser.add_argument("--entity_tag_vocab", required=True,
    	                help="Path to Entity tag vocab json file")

    parser.add_argument("--model_path", required=False,
    	                help="path to the saved model to load for evaluation")

    parser.add_argument('--dropout_rate', type=float, default=0.5,
                        help="dropout rate [default: 0.5]")

    parser.add_argument('--embed_size', type=int, default=256,
                        help="size of word embedding [default: 256]")

    parser.add_argument('--hidden_size', type=int, default=256,
                        help="size of hidden state [default: 256]")

    parser.add_argument('--batch_size', type=int, default=32,
                        help="batch-size [default: 32]")

    parser.add_argument('--max_epoch', type=int, default=10,
                        help="max epoch [default: 10]")

    parser.add_argument('--clip_max_norm', type=float, default=5.0,
                        help="clip max norm [default: 5.0]")

    parser.add_argument('--lr', type=float, default=0.001,
                        help="learning rate [default: 0.001]")

    parser.add_argument('--log_every', type=int, default=10,
                        help="log every [default: 10]")

    parser.add_argument('--validation_every', type=int, default=250,
                        help="validation every [default: 250]")

    parser.add_argument('--patience_threshold', type=float, default=0.98,
                        help="patience threshold [default: 0.98]")

    parser.add_argument('--max_patience', type=int, default=4,
                        help="time of continuous worse performance to decay lr [default: 4]")

    parser.add_argument('--max_decay', type=int, default=4,
                        help="time of lr decay to early stop [default: 4]")

    parser.add_argument('--lr_decay', type=float, default=0.5,
                        help="decay rate of lr [default: 0.5]")

    parser.add_argument('--model_save_path', type=str, default='./model/model.pth',
                        help="model save path [default: ./model/model.pth]")

    parser.add_argument('--optimizer_save_path', type=str, default='./model/optimizer.pth',
                        help="optimizer save path [default: ./model/optimizer.pth]")

    parser.add_argument('--cuda', action='store_true',
                        help="use GPU")

    return parser.parse_args()


def batch_iter(data, batch_size=32, shuffle=True):
    """ Yield batch of (sent, tag), by the reversed order of source length.
    Args:
        data: list of tuples, each tuple contains a sentence and corresponding tag.
        batch_size: batch size
        shuffle: bool value, whether to random shuffle the data
    """
    data_size = len(data)
    indices = list(range(data_size))
    if shuffle:
        random.shuffle(indices)
    batch_num = (data_size + batch_size - 1) // batch_size
    for i in range(batch_num):
        batch = [data[idx] for idx in indices[i * batch_size: (i + 1) * batch_size]]
        batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)
        sentences = [x[0] for x in batch]
        tags_ner = [x[1] for x in batch]
        tags_entity = [x[2] for x in batch]
        yield sentences, tags_ner, tags_entity


def calculate_loss(model, data, batch_size, sent_vocab, tag_vocab_ner, tag_vocab_entity, device, method):
    """ Calculate loss on the development data
    Args:
        model: the model being trained
        data: development data
        batch_size: batch size
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        device: torch.device on which the model is trained
    Returns:
        the average loss on the data
    """
    is_training = model.training
    model.eval()
    loss, n_sentences = 0, 0
    with torch.no_grad():
        for sentences, tags_ner, tags_entity in utils.batch_iter(data, batch_size, shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags_ner, _ = utils.pad(tags_ner, tag_vocab_ner[sent_vocab.PAD], device)
            tags_entity, _ = utils.pad(tags_entity, tag_vocab_entity[sent_vocab.PAD], device)
            batch_loss = model(sentences, tags_ner, tags_entity, sent_lengths, method)  # shape: (b,)
            loss += batch_loss.sum().item()
            n_sentences += len(sentences)
    model.train(is_training)
    return loss / n_sentences


def create_vocab(unique_tags_ner, unique_tags_entity, unique_words):
    # For tags NER
    unique_tags_dict = {unique_tags_ner[i]: i for i in range(len(unique_tags_ner))}
    tag_vocab = {"word2id": unique_tags_dict, "id2word": unique_tags_ner}
    json_object = json.dumps(tag_vocab)
    with open("./vocab/tag_vocab_ner.json", "w") as outfile:
        outfile.write(json_object)
    # For tags entity
    unique_tags_dict = {unique_tags_entity[i]: i for i in range(len(unique_tags_entity))}
    tag_vocab = {"word2id": unique_tags_dict, "id2word": unique_tags_entity}
    json_object = json.dumps(tag_vocab)
    with open("./vocab/tag_vocab_entity.json", "w") as outfile:
        outfile.write(json_object)
    # For words
    unique_words_dict = {unique_words[i]: i for i in range(len(unique_words))}
    sent_vocab = {"word2id": unique_words_dict, "id2word": unique_words}
    json_object = json.dumps(sent_vocab)
    with open("./vocab/sent_vocab.json", "w") as outfile:
        outfile.write(json_object)

    # Write the unique words into a text file
    with open('./data/data.txt', 'w', encoding='utf-8') as f:
        for word in unique_words:
            f.write(word + " ")

    # Train the fasttext model
    # model = fasttext.train_unsupervised('./data/data.txt', model='skipgram', minCount=1, dim=300)
    # model.save_model('./data/my_model.bin')

    return unique_words_dict


def entity_or_not(tags):
    new_tags = []
    for curr_set in tags:
        temp_tags = []
        for j, tag in enumerate(curr_set):
            if tag == 'O':
                temp_tags.append("O")
            elif tag == '<START>':
                temp_tags.append(tag)
            elif tag == '<END>':
                temp_tags.append(tag)
            elif tag == '<PAD>':
                temp_tags.append(tag)
            elif tag == '-DOCSTART-':
                temp_tags.append(tag)
            else:
                temp_tags.append("Y")
        new_tags.append(temp_tags)
    return new_tags


def generate_train_test_data(filepath, sent_vocab, tag_vocab_ner, tag_vocab_entity, train_proportion=0.8):
    """ Read corpus from given file path and split it into train and dev parts
    Args:
        filepath: file path
        sent_vocab: sentence vocab
        tag_vocab: tag vocab
        train_proportion: proportion of training data
    Returns:
        train_data: data for training, list of tuples, each containing a sentence and corresponding tag.
        test_data: data for testing, list of tuples, each containing a sentence and corresponding tag.
    """
    sentences, tags = read_corpus(filepath)
    sentences = words2indices(sentences, sent_vocab)
    tags_ner = words2indices(tags, tag_vocab_ner)
    tags_entity = words2indices(entity_or_not(tags), tag_vocab_entity)
    data = list(zip(sentences, tags_ner, tags_entity))
    random.shuffle(data)
    n_train = int(len(data) * train_proportion)
    train_data, test_data = data[: n_train], data[n_train:]
    return train_data, test_data


def getStats(tag, predicted_tag, tag_vocab):
    """ Calculate TN, FN, FP for the given true tag and predicted tag.
    Args:
        tag (list[int]): true tag
        predicted_tag (list[int]): predicted tag
        tag_vocab: tag vocab
    Returns:
        tp: true positive
        fp: false positive
        fn: false negative
    """
    tp, fp, fn = 0, 0, 0

    def func(tag1, tag2):
        a, b, i = 0, 0, 0
        while i < len(tag1):
            if tag1[i] == tag_vocab['O']:
                i += 1
                continue
            begin, end = i, i
            while end + 1 < len(tag1) and tag1[end + 1] != tag_vocab['O']:
                end += 1
            equal = True
            for j in range(max(0, begin - 1), min(len(tag1), end + 2)):
                if tag1[j] != tag2[j]:
                    equal = False
                    break
            a, b = a + equal, b + 1 - equal
            i = end + 1
        return a, b
    t, f = func(tag, predicted_tag)
    tp += t
    fn += f
    t, f = func(predicted_tag, tag)
    fp += f
    return tp, fp, fn


def indices2words(origin, vocab):
    """ Transform a sentence or a list of sentences from int to str
    Args:
        origin: a sentence of type list[int], or a list of sentences of type list[list[int]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with str
    """
    if isinstance(origin[0], list):
        result = [[vocab.id2word(w) for w in sent] for sent in origin]
    else:
        result = [vocab.id2word(w) for w in origin]
    return result

def iobes_iob(tags):
    """
    IOBES -> IOB
    """
    new_tags = []
    for curr_set in tags:
        temp_tags = []
        for j, tag in enumerate(curr_set):
            if tag.split('-')[0] == 'B':
                temp_tags.append(tag)
            elif tag.split('-')[0] == 'I':
                temp_tags.append(tag)
            elif tag.split('-')[0] == 'S':
                temp_tags.append(tag.replace('S-', 'B-'))
            elif tag.split('-')[0] == 'E':
                temp_tags.append(tag.replace('E-', 'I-'))
            elif tag.split('-')[0] == 'O':
                temp_tags.append(tag)
            else:
                temp_tags.append(tag)
                # raise Exception('Invalid format!')
        new_tags.append(temp_tags)
    return new_tags

def pad(data, padded_token, device):
    """ pad data so that each sentence has the same length as the longest sentence
    Args:
        data: list of sentences, List[List[word]]
        padded_token: padded token
        device: device to store data
    Returns:
        padded_data: padded data, a tensor of shape (max_len, b)
        lengths: lengths of batches, a list of length b.
    """
    lengths = [len(sent) for sent in data]
    max_len = lengths[0]
    padded_data = []
    for s in data:
        padded_data.append(s + [padded_token] * (max_len - len(s)))
    return torch.tensor(padded_data, device=device), lengths


def preprocess_data(args):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    """
    sentences = []
    sentence = []
    for line in codecs.open(args.input_path, 'r', 'utf8'):
        line = line.rstrip()
        if not line:
            if len(sentence) > 0:
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            word = line.split()
            assert len(word) >= 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)

    tags_ner = ['<START>',  '<END>', '<PAD>', '-DOCSTART-']
    tags_entity = ['<START>',  '<END>', '<PAD>', '-DOCSTART-']
    words = ['<START>',  '<END>', '<PAD>', '-DOCSTART-']

    for sentence in sentences:
        for sent in sentence:
            words.append(sent[0])
            tags_ner.append(sent[1])
            if sent[1] == 'O':
                tags_entity.append('O')
            else:
                tags_entity.append('Y')
    unique_tags_ner = list(Counter(tags_ner).keys())
    unique_tags_entity = list(Counter(tags_entity).keys())
    unique_words = list(Counter(words).keys())

    return unique_tags_ner, unique_tags_entity, unique_words


def pretrained(target_vocab, emb_dim=300):
    # Load pre-trained model
    # model = fasttext.load_model('./data/Pre-trained embeddings/crawl-300d-2M-subword.bin')
    model = fasttext.load_model('./data/my_model.bin')
    print("Done loading the pre-trained model.")

    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    words_found = 0

    for word, i in target_vocab.items():
        try:
            weights_matrix[i] = np.array(model[word]).astype(np.float)
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim,))
    print("Total number of words are ", len(target_vocab))
    print("Total number of words found in pre-trained embeddings are ", words_found)

    b = weights_matrix.tolist()
    file_path = "./data/weights_matrix.json"
    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

    return weights_matrix


def print_var(**kwargs):
    for k, v in kwargs.items():
        print(k, v)


def read_corpus(filepath):
    """ Read corpus from the given file path.
    Args:
        filepath: file path of the corpus
    Returns:
        sentences: a list of sentences, each sentence is a list of str
        tags: corresponding tags
    """
    sentences, tags = [], []
    sent, tag = ['<START>'], ['<START>']
    with open(filepath, 'r', encoding='utf8') as f:
        for line in f:
            if line == '\n':
                if len(sent) > 1:
                    sentences.append(sent + ['<END>'])
                    tags.append(tag + ['<END>'])
                sent, tag = ['<START>'], ['<START>']
            else:
                line = line.split()
                sent.append(line[0])
                tag.append(line[1])
    return sentences, tags


def words2indices(origin, vocab):
    """ Transform a sentence or a list of sentences from str to int
    Args:
        origin: a sentence of type list[str], or a list of sentences of type list[list[str]]
        vocab: Vocab instance
    Returns:
        a sentence or a list of sentences represented with int
    """
    if isinstance(origin[0], list):
        result = [[vocab[w] for w in sent] for sent in origin]
    else:
        result = [vocab[w] for w in origin]
    return result
