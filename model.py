import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vocab import Vocab
import utils

import numpy as np

class BiLSTMCRF(nn.Module):
    def __init__(self, weights_matrix, sent_vocab, tag_vocab_ner, tag_vocab_entity, dropout_rate=0.5, embed_size=300, hidden_size=256):
        """ Initialize the model
        Args:
            sent_vocab (Vocab): vocabulary of words
            tag_vocab (Vocab): vocabulary of tags
            embed_size (int): embedding size
            hidden_size (int): hidden state size
        """
        super(BiLSTMCRF, self).__init__()
        self.dropout_rate = dropout_rate
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.sent_vocab = sent_vocab
        self.tag_vocab_entity = tag_vocab_entity
        self.tag_vocab_ner = tag_vocab_ner

        # freeze=True is default. Change it to false to learn embeddings during training
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(weights_matrix), freeze=False)

        self.dropout = nn.Dropout(dropout_rate)
        self.encoder = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bidirectional=True)

        self.hidden2emit_score_ner = nn.Linear(hidden_size * 2, len(self.tag_vocab_ner))
        self.hidden2emit_score_entity = nn.Linear(hidden_size * 2, len(self.tag_vocab_entity))

        self.transition_ner = nn.Parameter(torch.randn(len(self.tag_vocab_ner), len(self.tag_vocab_ner)))  # shape: (K, K)
        self.transition_entity = nn.Parameter(torch.randn(len(self.tag_vocab_entity), len(self.tag_vocab_entity)))  # shape: (K, K)

        self.entity_experts_layer = nn.ModuleList([nn.Linear(2*self.hidden_size, len(self.tag_vocab_ner)) for _ in range(len(self.tag_vocab_ner))])
        self.moee_linear = nn.Linear(2*self.hidden_size, len(self.tag_vocab_ner))

    def MoEE_Block(self, hidden_states):
        """Mixture of Entity Experts"""
        m = []
        hidden_states = hidden_states.transpose(0, 1)
        for h in hidden_states: #shape of h - (b, 2h)
            expt = [self.dropout(l(h)) for l in self.entity_experts_layer]
            expt = torch.stack(expt).transpose(0, 1)
            alpha = nn.Softmax(self.dropout(self.moee_linear(h)))
            T = torch.bmm(expt.reshape(expt.shape[0], expt.shape[1], expt.shape[2]), alpha.dim.view(alpha.dim.shape[0], alpha.dim.shape[1], 1))[:, :, -1]
            m.append(T)
        m = torch.stack(m)
        m = m.transpose(0, 1)
        return m

    def forward(self, sentences, tags_ner, tags_entity, sen_lengths, method):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            tags (tensor): corresponding tags, shape (b, len)
            sen_lengths (list): sentence lengths
        Returns:
            loss (tensor): loss on the batch, shape (b,)
        """
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD]).to(self.device)  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        # MTL
        hidden_states = self.encode(sentences, sen_lengths)  # shape: (b, len, 2h)

        if method == 'NER':
            # For NER
            emit_score_ner = self.hidden2emit_score_ner(hidden_states)  # shape: (b, len, K)
            emit_score_ner = self.dropout(emit_score_ner)  # shape: (b, len, K)
            loss = self.calculate_loss(tags_ner, mask, emit_score_ner, self.transition_ner)  # shape: (b,)

        elif method == 'NER_Entity':
            # For NER
            emit_score_ner = self.hidden2emit_score_ner(hidden_states)  # shape: (b, len, K)
            emit_score_ner = self.dropout(emit_score_ner)  # shape: (b, len, K)
            loss_ner = self.calculate_loss(tags_ner, mask, emit_score_ner, self.transition_ner)  # shape: (b,)
            # For Entity
            emit_score_entity = self.hidden2emit_score_entity(hidden_states)  # shape: (b, len, K)
            emit_score_entity = self.dropout(emit_score_entity)  # shape: (b, len, K)
            loss_entity = self.calculate_loss(tags_entity, mask, emit_score_entity, self.transition_entity)  # shape: (b,)
            loss = loss_ner + loss_entity

        elif method == 'MOEE':
            # For MoEE
            moee_output = self.MoEE_Block(hidden_states)  # shape: (b,)
            loss_moee = self.calculate_loss(tags_ner, mask, moee_output, self.transition_ner)  # shape: (b,)
            loss = loss_moee

        elif method == "Entity_MOEE":
            # For Entity
            emit_score_entity = self.hidden2emit_score_entity(hidden_states)  # shape: (b, len, K)
            emit_score_entity = self.dropout(emit_score_entity)  # shape: (b, len, K)
            loss_entity = self.calculate_loss(tags_entity, mask, emit_score_entity, self.transition_entity)  # shape: (b,)
            # For MoEE
            moee_output = self.MoEE_Block(hidden_states)  # shape: (b,)
            loss_moee = self.calculate_loss(tags_ner, mask, moee_output, self.transition_ner)  # shape: (b,)
            loss = loss_entity+loss_moee

        elif method == "NER_ENTITY_MOEE":
            # For NER
            emit_score_ner = self.hidden2emit_score_ner(hidden_states)  # shape: (b, len, K)
            emit_score_ner = self.dropout(emit_score_ner)  # shape: (b, len, K)
            loss_ner = self.calculate_loss(tags_ner, mask, emit_score_ner, self.transition_ner)  # shape: (b,)
            # For Entity
            emit_score_entity = self.hidden2emit_score_entity(hidden_states)  # shape: (b, len, K)
            emit_score_entity = self.dropout(emit_score_entity)  # shape: (b, len, K)
            loss_entity = self.calculate_loss(tags_entity, mask, emit_score_entity, self.transition_entity)  # shape: (b,)
            # For MoEE
            moee_output = self.MoEE_Block(hidden_states)  # shape: (b,)
            loss_moee = self.calculate_loss(tags_ner, mask, moee_output, self.transition_ner)  # shape: (b,)
            loss = loss_ner + loss_entity + loss_moee

        return loss

    def encode(self, sentences, sent_lengths):
        """ BiLSTM Encoder
        Args:
            sentences (tensor): sentences with word embeddings, shape (len, b, e)
            sent_lengths (list): sentence lengths
        Returns:
            emit_score (tensor): emit score, shape (b, len, K)
        """
        padded_sentences = pack_padded_sequence(sentences, sent_lengths)
        hidden_states, _ = self.encoder(padded_sentences)
        hidden_states, _ = pad_packed_sequence(hidden_states, batch_first=True)  # shape: (b, len, 2h)
        # emit_score = hidden2emit_score(hidden_states)  # shape: (b, len, K)
        # emit_score = self.dropout(emit_score)  # shape: (b, len, K)
        return hidden_states



    def calculate_loss(self, tags, mask, emit_score, transition):
        """ Calculate CRF loss
        Args:
            tags (tensor): a batch of tags, shape (b, len)
            mask (tensor): mask for the tags, shape (b, len), values in PAD position is 0
            emit_score (tensor): emit matrix, shape (b, len, K)
        Returns:
            loss (tensor): loss of the batch, shape (b,)
        """
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    def predict(self, sentences, sen_lengths, method):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            sen_lengths (list): sentence lengths
        Returns:
            tags (list[list[str]]): predicted tags for the batch
        """
        batch_size = sentences.shape[0]
        mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        sentences = self.embedding(sentences)  # shape: (len, b, e)
        hidden_states = self.encode(sentences, sen_lengths)  # shape: (b, len, 2*h)
        # For MOEE
        if method == 'MOEE' or method == 'Entity_MOEE' or method == "NER_ENTITY_MOEE":
            emit_score = self.MoEE_Block(hidden_states)  # shape: (b,)
        # For everything
        else:
            emit_score = self.hidden2emit_score_ner(hidden_states)  # shape: (b, len, K)
            emit_score = self.dropout(emit_score)  # shape: (b, len, K)

        tags = [[[i] for i in range(len(self.tag_vocab_ner))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition_ner + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags

    def save(self, filepath):
        params = {
            'sent_vocab': self.sent_vocab,
            'tag_vocab_ner': self.tag_vocab_ner,
            'tag_vocab_entity': self.tag_vocab_entity,
            'args': dict(dropout_rate=self.dropout_rate, embed_size=self.embed_size, hidden_size=self.hidden_size),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(weights_matrix, filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BiLSTMCRF(weights_matrix, params['sent_vocab'], params['tag_vocab_ner'], params['tag_vocab_entity'], **params['args'])
        weights_matrix = torch.DoubleTensor(np.array(weights_matrix))
        params['state_dict']['embedding.weight'] = weights_matrix
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model

    @property
    def device(self):
        return self.embedding.weight.device
