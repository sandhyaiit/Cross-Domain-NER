from vocab import Vocab
import time

import torch
import torch.nn as nn
import model
import utils
import random

import codecs
import json
import fasttext
import numpy as np


def test(args, weights_matrix):
    """ Testing the model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args.sent_vocab)
    tag_vocab = Vocab.load(args.ner_tag_vocab)
    sentences, tags = utils.read_corpus(args.input_path)
    sentences = utils.words2indices(sentences, sent_vocab)

    # Method
    method = args.method

    # # Convert to binary tags (if there is a tag or not)
    tags_entity = utils.entity_or_not(tags)

    # Convert from IOBES to IOB
    tags = utils.iobes_iob(tags)

    tags = utils.words2indices(tags, tag_vocab)
    test_data = list(zip(sentences, tags, tags_entity))
    print('num of test samples: %d' % (len(test_data)))

    device = torch.device('cuda' if args.cuda else 'cpu')
    bilstmcrf_model = model.BiLSTMCRF.load(weights_matrix, args.model_path, device)
    print('start testing...')
    print('using device', device)

    start = time.time()
    n_iter, num_words = 0, 0
    tp, fp, fn = 0, 0, 0

    bilstmcrf_model.eval()
    with torch.no_grad():
        for sentences, tags, tags_entity in utils.batch_iter(test_data, batch_size=int(args.batch_size), shuffle=False):
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            predicted_tags = bilstmcrf_model.predict(sentences, sent_lengths, method)
            n_iter += 1
            num_words += sum(sent_lengths)
            for tag, predicted_tag in zip(tags, predicted_tags):
                current_tp, current_fp, current_fn = utils.getStats(tag, predicted_tag, tag_vocab)
                tp += current_tp
                fp += current_fp
                fn += current_fn
            if n_iter % int(args.log_every) == 0:
                print('log: iter %d, %.1f words/sec, precision %f, recall %f, f1_score %f, time %.1f sec' %
                      (n_iter, num_words / (time.time() - start), tp / (tp + fp), tp / (tp + fn),
                       (2 * tp) / (2 * tp + fp + fn), time.time() - start))
                num_words = 0
                start = time.time()
    print('tp = %d, fp = %d, fn = %d' % (tp, fp, fn))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = (2 * tp) / (2 * tp + fp + fn)
    print('Precision: %f, Recall: %f, F1 score: %f' % (precision, recall, f1_score))

def main():
    args = utils.parse_args()
    # Load the weights matrix file generated while training
    file_path = "./data/weights_matrix.json"
    obj_text = codecs.open(file_path, 'r', encoding='utf-8').read()
    b_new = json.loads(obj_text)
    weights_matrix = np.array(b_new)

    # Get the unique words and unique tags from the test file
    unique_tags_ner, unique_tags_entity, unique_words = utils.preprocess_data(args)
    # Add the unique words from the test data (not present in train data) to the dictionary
    # Load the train vocab
    with open('./vocab/sent_vocab.json') as json_file:
        train_vocab = json.load(json_file)
    train_words = train_vocab["id2word"]
    model = fasttext.load_model('./data/my_model.bin')
    final_words = []
    for word in unique_words:
        if word in train_words:
            continue
        else:
            final_words.append(word)
    # If there are new words
    if len(final_words) > 0:
        unique_words_dict = {final_words[i]: i+len(weights_matrix) for i in range(len(final_words))}

        # Update the weights_matrix
        matrix_len = len(unique_words_dict)+len(weights_matrix)
        final_weights_matrix = np.zeros((matrix_len, 300))
        # Rewrite the train weights
        for i in range(len(weights_matrix)):
            final_weights_matrix[i] = weights_matrix[i]
        # Write the test weights
        for word, i in unique_words_dict.items():
            try:
                final_weights_matrix[i] = np.array(model.get_word_vector(word)).astype(np.float)
            except KeyError:
                final_weights_matrix[i] = np.random.normal(scale=0.6, size=(300,))

        final_dict = {**unique_words_dict, **train_vocab["word2id"]}
        final_id2word = train_words+final_words
        sent_vocab = {"word2id": final_dict, "id2word": final_id2word}
        json_object = json.dumps(sent_vocab)
        with open("./vocab/sent_vocab.json", "w") as outfile:
            outfile.write(json_object)

        print("Finally here!!")
        b = final_weights_matrix.tolist()
        file_path = "./data/weights_matrix.json"
        json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)

        test(args, final_weights_matrix)

if __name__ == '__main__':
    main()
