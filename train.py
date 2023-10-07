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


def train(args, weights_matrix):
    """ Training BiLSTMCRF model
    Args:
        args: dict that contains options in command
    """
    sent_vocab = Vocab.load(args.sent_vocab)
    tag_vocab_ner = Vocab.load(args.ner_tag_vocab)
    tag_vocab_entity = Vocab.load(args.entity_tag_vocab)
    method = args.method
    train_data, test_data = utils.generate_train_test_data(args.input_path, sent_vocab, tag_vocab_ner, tag_vocab_entity)
    print('num of train examples: %d' % (len(train_data)))
    print('num of test examples: %d' % (len(dev_data)))

    max_epoch = int(args.max_epoch)
    log_every = int(args.log_every)
    validation_every = int(args.validation_every)
    model_save_path = args.model_save_path
    optimizer_save_path = args.optimizer_save_path
    min_dev_loss = float('inf')
    device = torch.device('cuda' if args.cuda else 'cpu')
    patience, decay_num = 0, 0

    bilstmcrf_model = model.BiLSTMCRF(weights_matrix, sent_vocab, tag_vocab_ner, tag_vocab_entity, float(args.dropout_rate), int(args.embed_size),
                                 int(args.hidden_size)).to(device)
    print(bilstmcrf_model)
    for name, param in bilstmcrf_model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, 0, 0.01)
        else:
            nn.init.constant_(param.data, 0)

    optimizer = torch.optim.Adam(bilstmcrf_model.parameters(), lr=float(args.lr))
    train_iter = 0  # train iter num
    record_loss_sum, record_tgt_word_sum, record_batch_size = 0, 0, 0  # sum in one training log
    cum_loss_sum, cum_tgt_word_sum, cum_batch_size = 0, 0, 0  # sum in one validation log
    record_start, cum_start = time.time(), time.time()

    print('start training...')
    for epoch in range(max_epoch):
        for sentences, tags_ner, tags_entity in utils.batch_iter(train_data, batch_size=int(args.batch_size)):
            train_iter += 1
            current_batch_size = len(sentences)
            sentences, sent_lengths = utils.pad(sentences, sent_vocab[sent_vocab.PAD], device)
            tags_ner, _ = utils.pad(tags_ner, tag_vocab_ner[tag_vocab_ner.PAD], device)
            tags_entity, _ = utils.pad(tags_entity, tag_vocab_entity[tag_vocab_entity.PAD], device)

            # back propagation
            optimizer.zero_grad()
            batch_loss = bilstmcrf_model(sentences, tags_ner, tags_entity, sent_lengths, method)  # shape: (b,)
            loss = batch_loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bilstmcrf_model.parameters(), max_norm=float(args.clip_max_norm))
            optimizer.step()

            record_loss_sum += batch_loss.sum().item()
            record_batch_size += current_batch_size
            record_tgt_word_sum += sum(sent_lengths)

            cum_loss_sum += batch_loss.sum().item()
            cum_batch_size += current_batch_size
            cum_tgt_word_sum += sum(sent_lengths)

            if train_iter % log_every == 0:
                print('log: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, record_tgt_word_sum / (time.time() - record_start),
                       record_loss_sum / record_batch_size, time.time() - record_start))
                record_loss_sum, record_batch_size, record_tgt_word_sum = 0, 0, 0
                record_start = time.time()

            if train_iter % validation_every == 0:
                print('dev: epoch %d, iter %d, %.1f words/sec, avg_loss %f, time %.1f sec' %
                      (epoch + 1, train_iter, cum_tgt_word_sum / (time.time() - cum_start),
                       cum_loss_sum / cum_batch_size, time.time() - cum_start))
                cum_loss_sum, cum_batch_size, cum_tgt_word_sum = 0, 0, 0

                dev_loss = utils.calculate_loss(bilstmcrf_model, dev_data, 64, sent_vocab, tag_vocab_ner, tag_vocab_entity, device, method)
                if dev_loss < min_dev_loss * float(args.patience_threshold):
                    min_dev_loss = dev_loss
                    bilstmcrf_model.save(model_save_path)
                    torch.save(optimizer.state_dict(), optimizer_save_path)
                    print('Reached %d epochs, Save result model to %s' % (epoch, model_save_path))
                    patience = 0
                    # Save the word embeddings
                    print("Saving the model")
                    params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                    new_weights_matrix = params['state_dict']['embedding.weight']
                    b = new_weights_matrix.tolist()
                    file_path = "./data/weights_matrix.json"
                    json.dump(b, codecs.open(file_path, 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4)
                else:
                    patience += 1
                    if patience == int(args.max_patience):
                        decay_num += 1
                        if decay_num == int(args.max_decay):
                            return
                        lr = optimizer.param_groups[0]['lr'] * float(args.lr_decay)
                        bilstmcrf_model = model.BiLSTMCRF.load(weights_matrix, model_save_path, device)
                        optimizer.load_state_dict(torch.load(optimizer_save_path))
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        patience = 0
                print('dev: epoch %d, iter %d, dev_loss %f, patience %d, decay_num %d' %
                      (epoch + 1, train_iter, dev_loss, patience, decay_num))
                cum_start = time.time()
                if train_iter % log_every == 0:
                    record_start = time.time()
    bilstmcrf_model.save(model_save_path)
    print('Reached %d epochs, Save result model to %s' % (max_epoch, model_save_path))

def main():
    args = utils.parse_args()
    random.seed(0)
    torch.manual_seed(0)

    if args.cuda:
        torch.cuda.manual_seed(0)

    unique_tags_ner, unique_tags_entity, unique_words = utils.preprocess_data(args)
    unique_words_dict = utils.create_vocab(unique_tags_ner, unique_tags_entity, unique_words)
    print("Done preprocessing the data")
    weights_matrix = utils.pretrained(unique_words_dict)
    print("Done computing the weights matrix")
    train(args, weights_matrix)

if __name__ == '__main__':
    main()
