#! /usr/bin/env python3

"""
    ABLTagger: Augmented BiDirectional LSTM Tagger

    Main module

    Copyright (C) 2019 Örvar Kárason and Steinþór Steingrímsson.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
__license__ = "Apache 2.0"


import random
import numpy as np
import sys
from collections import defaultdict, Counter
from itertools import count

import dynet_config
dynet_config.set(mem=4096, random_seed=42)
random.seed(42)
import dynet as dy
import gc


class Utils:
    @staticmethod
    def build_word_dict(train_words):
        words = []
        word_frequency = Counter()
        for lina in train_words:
            for w, p in lina:
                words.append(w)
                word_frequency[w] += int(p)
        return Vocab.from_corpus([words]), word_frequency

    @staticmethod
    def build_vocab_tags(tag_file):
        train_tags = []
        for tagline in open(tag_file):
            tags = tagline.split('\t')
            for t in tags:
                train_tags.append(t)
        return Vocab.from_corpus([train_tags])

    @staticmethod
    def create_vocabularies(training_file):
        word_frequency = Counter()
        words = []
        tags = []
        train = Utils.read(training_file)
        for sent in train:
            for w, p in sent:
                words.append(w)
                tags.append(p)
                word_frequency[w] += 1

        words = list(dict.fromkeys(words))
        tags = list(dict.fromkeys(tags))
        return words, word_frequency, tags

    @staticmethod
    def read(fname, cols=2):
        sent = []
        for line in open(fname):
            line = line.strip().split()
            if not line:
                if sent: yield sent
                sent = []
            elif cols == 3:
                w, p, t = line
                sent.append((w, p, t))
            else:
                try:
                    w, p = line
                    sent.append((w, p))
                except:
                    print(line)
                    sys.exit(1)
        if sent: yield sent

    @staticmethod
    def save_args(arguments, filename):
        arg_dict = arguments.__dict__
        args_keys = arg_dict.keys()
        with open(filename, "w") as arg_file:
            for i in args_keys:
                if i not in ('model', 'tag_type', 'training_type'):
                    arg_file.write('--' + i + '\n' + str(arg_dict[i]) + '\n')


class Vocab:
    def __init__(self, w2i=None):
        if w2i is None: w2i = defaultdict(count(0).__next__)
        self.w2i = dict(w2i)
        self.i2w = {i: w for w, i in w2i.items()}

    @classmethod
    def from_corpus(cls, corpus):
        w2i = defaultdict(count(0).__next__)
        for sentence in corpus:
            [w2i[word] for word in sentence]
        return Vocab(w2i)

    def size(self): return len(self.w2i.keys())


class Embeddings:
    def __init__(self, emb_file):
        self.vocab = {}
        with open(emb_file) as f:
            for i, l in enumerate(f):
                pass
        key_number = i + 1
        vector_length = 0

        with open(emb_file) as f:
            f.readline()
            for i, line in enumerate(f):
                fields = line.strip().split(";")
                vector_length = len(eval(fields[1]))
                break

        self.embeddings = np.zeros((key_number, vector_length))

        with open(emb_file) as f:
            f.readline()
            for i, line in enumerate(f):
                fields = line.strip().split(";")
                self.vocab[fields[0]] = i
                self.embeddings[i] = eval(fields[1])
                if key_number > 100:
                    if i % int(key_number/100) == 0:
                        print(str(int((i*100)/key_number)) + '%   ', end='\r')
            print('100%', end='\r')


# Layer Dimensions for Combined emb. model
class CombinedDims:
    def __init__(self):
        self.hidden = 32
        self.hidden_input = 128
        self.char_input = 20
        self.word_input = 256
        self.tags_input = 30
        self.char_output = 64
        self.word_output = 64
        self.word_lookup = 128
        self.char_lookup = 20
        self.morphlex_lookup = 65
        self.word_class_lookup = 14


class ABLTagger():
    START_OF_WORD = "<w>"
    END_OF_WORD = "</w>"
    
    def __init__(self, vocab_chars, vocab_words, vocab_tags, word_freq, morphlex_embeddings=None, coarse_embeddings=None, hyperparams=None):
        self.model = dy.Model()
        self.trainer = None
        self.word_frequency = word_freq

        self.vw = vocab_words
        self.vt = vocab_tags
        self.vc = vocab_chars

        self.morphlex_flag = False
        self.coarse_flag = False
        if coarse_embeddings is not None:
            self.coarse_flag = True
            self.coarse_embeddings = coarse_embeddings.embeddings
            self.coarse = coarse_embeddings.vocab
        if morphlex_embeddings is not None:
            self.morphlex_flag = True
            self.morphlex_embeddings = morphlex_embeddings.embeddings
            self.morphlex = morphlex_embeddings.vocab
        if hyperparams is not None:
            self.hp = hyperparams
            self.set_trainer(self.hp.optimization)

        self.dim = CombinedDims()
        self.create_network()

    def create_network(self):
        assert self.vw.size(), "Need to build the vocabulary (build_vocab) before creating the network."

        self.dim.word_input = self.dim.word_lookup + self.dim.char_output * 2
        if self.morphlex_flag:
            self.dim.word_input += self.dim.morphlex_lookup

        if self.coarse_flag:
            self.dim.word_input += self.dim.word_class_lookup

        self.WORDS_LOOKUP = self.model.add_lookup_parameters((self.vw.size(), self.dim.word_lookup))
        self.CHARS_LOOKUP = self.model.add_lookup_parameters((self.vc.size(), self.dim.char_lookup))

        if self.morphlex_flag:
            self.MORPHLEX_LOOKUP = self.model.add_lookup_parameters((len(self.morphlex_embeddings), self.dim.morphlex_lookup))
            self.MORPHLEX_LOOKUP.init_from_array(self.morphlex_embeddings)
        if self.coarse_flag:
            self.WORD_CLASS_LOOKUP = self.model.add_lookup_parameters((11, self.dim.word_class_lookup))
            self.WORD_CLASS_LOOKUP.init_from_array(self.coarse_embeddings)

        # MLP on top of biLSTM outputs, word/char out -> hidden -> num tags
        self.pH = self.model.add_parameters((self.dim.hidden, self.dim.hidden_input))  # hidden-dim, hidden-input-dim
        self.pO = self.model.add_parameters((self.vt.size(), self.dim.hidden))  # vocab-size, hidden-dim
        gc.collect()

        # word-level LSTMs
        self.fwdRNN = dy.LSTMBuilder(1, self.dim.word_input, self.dim.word_output, self.model) # layers, input-dim, output-dim
        self.bwdRNN = dy.LSTMBuilder(1, self.dim.word_input, self.dim.word_output, self.model)
        gc.collect()

        # char-level LSTMs
        self.cFwdRNN = dy.LSTMBuilder(1, self.dim.char_input, self.dim.char_output, self.model)
        self.cBwdRNN = dy.LSTMBuilder(1, self.dim.char_input, self.dim.char_output, self.model)

    def set_trainer(self, optimization):
        if optimization == 'MomentumSGD':
            self.trainer = dy.MomentumSGDTrainer(self.model, learning_rate=self.hp.learning_rate)
        if optimization == 'CyclicalSGD':
            self.trainer = dy.CyclicalSGDTrainer(self.model, learning_rate_max=self.hp.learning_rate_max, learning_rate_min=self.hp.learning_rate_min)
        if optimization == 'Adam':
            self.trainer = dy.AdamTrainer(self.model)
        if optimization == 'RMSProp':
            self.trainer = dy.RMSPropTrainer(self.model)
        else:
            self.trainer = dy.SimpleSGDTrainer(self.model, learning_rate=self.hp.learning_rate)

    def dynamic_rep(self, w, cf_init, cb_init):
        if self.word_frequency[w] >= self.hp.words_min_freq:
            return self.word_rep(w)
        else:
            return self.char_rep(w, cf_init, cb_init)

    def char_rep(self, w, cf_init, cb_init):
        char_ids = [self.vc.w2i[self.START_OF_WORD]] + [self.vc.w2i[c] if c in self.vc.w2i else -1 for c in w] + [self.vc.w2i[self.END_OF_WORD]]
        char_embs = [self.CHARS_LOOKUP[cid] if cid != -1 else dy.zeros(self.dim.char_lookup) for cid in char_ids]
        fw_exps = cf_init.transduce(char_embs)
        bw_exps = cb_init.transduce(reversed(char_embs))
        return dy.concatenate([fw_exps[-1], bw_exps[-1]])

    def word_rep(self, w):
        if self.word_frequency[w] == 0:
            return dy.zeros(self.dim.word_lookup)
        w_index = self.vw.w2i[w]
        return self.WORDS_LOOKUP[w_index]

    def morphlex_rep(self, w):
        if w not in self.morphlex.keys():
            return dy.zeros(self.dim.morphlex_lookup)
        else:
            return self.MORPHLEX_LOOKUP[self.morphlex[w]]

    def coarse_rep(self, t):
        if t not in self.coarse.keys():
            return dy.zeros(self.dim.word_class_lookup)
        else:
            return self.WORD_CLASS_LOOKUP[self.coarse[t]]

    def word_and_char_rep(self, w, cf_init, cb_init, t='0'):
        wembs = self.word_rep(w)
        cembs = self.char_rep(w, cf_init, cb_init)
        if self.coarse_flag:
            coarse = self.coarse_rep(t)
        if self.morphlex_flag:
            morphlex = self.morphlex_rep(w)

        if self.coarse_flag and self.morphlex_flag:
            return dy.concatenate([wembs, cembs, morphlex, coarse])
        elif self.coarse_flag:
            return dy.concatenate([wembs, cembs, coarse])
        elif self.morphlex_flag:
            return dy.concatenate([wembs, cembs, morphlex])
        else:
            return dy.concatenate([wembs, cembs])

    def build_tagging_graph(self, words, tags):
        dy.renew_cg()

        # Initialize the LSTMs
        f_init = self.fwdRNN.initial_state()
        b_init = self.bwdRNN.initial_state()

        cf_init = self.cFwdRNN.initial_state()
        cb_init = self.cBwdRNN.initial_state()

        # Get the word vectors, a 128-dim vector expression for each word.
        if self.coarse_flag:
            wembs = [self.word_and_char_rep(w, cf_init, cb_init, t[0]) for w, t in zip(words, tags)]
        else:
            wembs = [self.word_and_char_rep(w, cf_init, cb_init) for w in words]

        if self.hp.noise > 0:
            wembs = [dy.noise(we, self.hp.noise) for we in wembs]

        # Feed word vectors into biLSTM
        fw_exps = f_init.transduce(wembs)
        bw_exps = b_init.transduce(reversed(wembs))

        # biLSTM states
        bi_exps = [dy.concatenate([f, b]) for f, b in zip(fw_exps, reversed(bw_exps))]

        # Feed each biLSTM state to an MLP
        return [self.pO * (dy.tanh(self.pH * x)) for x in bi_exps]

    def sent_loss(self, sent):
        words, tags = map(list, zip(*sent))
        vecs = self.build_tagging_graph(words, tags)
        errs = []
        for v, t in zip(vecs, tags):
            tid = self.vt.w2i[t]
            err = dy.pickneglogsoftmax(v, tid)
            errs.append(err)
        return dy.esum(errs)

    def tag_sent(self, words, in_tags=None):
        vecs = self.build_tagging_graph(words, in_tags)
        vecs = [dy.softmax(v) for v in vecs]
        probs = [v.npvalue() for v in vecs]
        tags = []
        for prb in probs:
            tag = np.argmax(prb)
            tags.append(self.vt.i2w[tag])
        return zip(words, tags)

    def update_trainer(self):
        self.trainer.update()

    def train(self, epochs, training_data):
        self.setup(training_data)

        for _ in range(epochs):
            random.shuffle(training_data)
            for sent in training_data:
                loss_exp = self.sent_loss(sent)
                loss_exp.backward()
                self.update_trainer()


    def save_trained_embeddings(self, emb_file):
        self.model.save(emb_file)

    def load_model(self, emb_file):
        self.model.populate(emb_file)
