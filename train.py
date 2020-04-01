#! /usr/bin/env python3

"""
    ABLTagger: Augmented BiDirectional LSTM Tagger

    Script for training models for later use

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
import os, sys, shutil
from itertools import cycle
from time import time
import argparse
from ABLTagger import ABLTagger, Embeddings, Vocab, Utils

spinner = cycle('⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏')


def update_progress_notice(i, epoch, total_epochs, start_time, epoch_start_time, avg_loss, evaluation = None, morphlex_flag = False):
    now_time = time()
    if morphlex_flag:
        print(" ",
            next(spinner),
            "{:>2}/{}".format(epoch, total_epochs),
            ("  {:>4}/{:<5}".format(int(now_time - start_time), str(int(now_time - epoch_start_time)) + 's') if i % 100 == 0 or evaluation else ""),
            ("  AVG LOSS: {:.3}".format(avg_loss) if i % 1000 == 0 or evaluation else ""),
            ("  EVAL: tags {:.3%} sent {:.3%} tr. only {:.3%} dmii only {:.3%} both {:.3%} knw {:.3%} unk {:.3%}".format(*evaluation) if evaluation else ""),
            end='\r'
        )
    else:
        print(" ",
            next(spinner),
            "{:>2}/{}".format(epoch, total_epochs),
            ("  {:>4}/{:<5}".format(int(now_time - start_time), str(int(now_time - epoch_start_time)) + 's') if i % 100 == 0 or evaluation else ""),
            ("  AVG LOSS: {:.3}".format(avg_loss) if i % 1000 == 0 or evaluation else ""),
            ("  EVAL: tags {:.3%} sent {:.3%} knw {:.3%} unk {:.3%}".format(*evaluation) if evaluation else ""),
            end='\r'
        )


def train_tagger(tagger, training_data, model_folder, training_type, morphlex=None, epochs=20):
    '''
    Train the tagger and report progress to console.
    '''

    if args.dropout:
        tagger.fwdRNN.set_dropouts(args.dropout, 0)
        tagger.bwdRNN.set_dropouts(args.dropout, 0)
        tagger.cFwdRNN.set_dropouts(args.dropout, 0)
        tagger.cBwdRNN.set_dropouts(args.dropout, 0)

    start_time = time()
    for ITER in range(epochs):
        cum_loss = num_tagged = 0
        epoch_start_time = time()
        random.shuffle(training_data)
        for i, sent in enumerate(training_data, 1):
            # Training
            loss_exp = tagger.sent_loss(sent)
            cum_loss += loss_exp.scalar_value()
            num_tagged += len(sent)
            loss_exp.backward()
            tagger.update_trainer()

            if i % 10 == 0:
                if morphlex is not None:
                    update_progress_notice(i, ITER + 1, epochs, start_time, epoch_start_time, cum_loss / num_tagged, None, True)
                else:
                    update_progress_notice(i, ITER + 1, epochs, start_time, epoch_start_time, cum_loss / num_tagged)

        # decay
        if args.learning_rate_decay:
            tagger.trainer.learning_rate = tagger.trainer.learning_rate * (1-args.learning_rate_decay)

    # Show hyperparameters used when we are done
    print("\nHP opt={} epochs={} emb_noise={} ".format(args.optimization, epochs, args.noise))

    tagger.save_trained_embeddings(model_folder + 'model.' + training_type)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # HYPERPARAMETERS
    parser.add_argument('--optimization', '-o', help="Which optimization algorithm",
                        choices=['SimpleSGD', 'MomentumSGD', 'CyclicalSGD', 'Adam', 'RMSProp'], default='SimpleSGD')
    parser.add_argument('--learning_rate', '-lr', help="Learning rate", type=float, default=0.13)
    parser.add_argument('--learning_rate_decay', '-lrd', help="Learning rate decay", type=float, default=0.05)
    parser.add_argument('--learning_rate_max', '-l_max', help="Learning rate max for Cyclical SGD", type=float, default=0.1)
    parser.add_argument('--learning_rate_min', '-l_min', help="Learning rate min for Cyclical SGD", type=float, default=0.01)
    parser.add_argument('--dropout', '-d', help="Dropout rate", type=float, default=0.05)
    parser.add_argument('--noise', '-n', help="Noise in embeddings", type=float, default=0.1)
    # EXTERNAL DATA
    parser.add_argument('--use_morphlex', '-morphlex', help="File with morphological lexicon embeddings in ./extra folder. Example file: ./extra/dmii.or", default='./extra/dmii.vectors')
    parser.add_argument('--load_characters', '-load_chars', help="File to load characters from", default='./extra/characters_training.txt')
    parser.add_argument('--load_coarse_tagset', '-load_coarse', help="Load embeddings file for coarse grained tagset", default='./extra/word_class_vectors.txt')
    parser.add_argument('--training_type', '-type', help='Select training type: coarse, fine or combined.', choices=['coarse', 'fine', 'combined'], default="combined")
    # TRAIN MODEL
    parser.add_argument('--epochs_coarse_grained', '-ecg', help="How many epochs for coarse grained training? (12 is default)", type=int, default=12)
    parser.add_argument('--epochs_fine_grained', '-efg', help="How many epochs for fine grained training? (20 is default)", type=int, default=20)

    requiredNamed = parser.add_argument_group('Required named arguments')
    requiredNamed.add_argument('--model', '-m', help="Name of new model")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)
    args = parser.parse_args()

    char_list = []
    for charline in open(args.load_characters):
        characters = charline.strip().split('\t')
        for c in characters:
            char_list.append(c)
    VocabCharacters = Vocab.from_corpus([char_list])

    if args.use_morphlex is not None:
        print("Building morphological lexicon embeddings...")
        morphlex_embeddings = Embeddings(args.use_morphlex)
        print("Morphological lexicon embeddings in place")
    else:
        morphlex_embeddings = None

    model_folder = './models/' + args.model + '/'
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    shutil.copy2(args.load_characters, model_folder + 'characters.txt')
    args.load_characters = 'characters.txt'
    shutil.copy2(args.load_coarse_tagset, model_folder + 'coarse_tagset.txt')
    args.load_coarse_tagset = 'coarse_tagset.txt'

    if args.training_type == 'combined':
        training_file_coarse = './data/' + args.model + '.coarse'
        training_file_fine = './data/' + args.model + '.fine'
        print('Reading coarse-grained training file...')
        train_coarse = list(Utils.read(training_file_coarse))
        print('Reading fine-grained training file...')
        train_fine = list(Utils.read(training_file_fine))
        print('Creating vocabularies...')
        words, word_frequency, tags_coarse = Utils.create_vocabularies(training_file_coarse)
        words, word_frequency, tags_fine = Utils.create_vocabularies(training_file_fine)
        with open(model_folder + 'words.txt', "w") as word_freq_file:
            for i in words:
                word_freq_file.write(i.strip() + '\t' + str(word_frequency[i]) + '\n')
        with open(model_folder + 'tags_coarse.txt', "w") as tag_file_coarse:
            for i in tags_coarse:
                tag_file_coarse.write(i.strip() + '\t')
        with open(model_folder + 'tags_fine.txt', "w") as tag_file_fine:
            for i in tags_fine:
                tag_file_fine.write(i.strip() + '\t')
        shutil.copy2(args.use_morphlex, model_folder + 'morphlex.txt')
        VocabWords, WordFrequency = Utils.build_word_dict(list(Utils.read(model_folder + 'words.txt')))
        print("Known tokens loaded")
        VocabTagsCoarse = Utils.build_vocab_tags(model_folder + 'tags_coarse.txt')
        VocabTagsFine = Utils.build_vocab_tags(model_folder + 'tags_fine.txt')
        coarse_embeddings = Embeddings(model_folder + args.load_coarse_tagset)
        Utils.save_args(args, model_folder + 'args.' + args.training_type)
        tagger_coarse = ABLTagger(VocabCharacters, VocabWords, VocabTagsCoarse, WordFrequency, morphlex_embeddings, None, args)
        train_tagger(tagger_coarse, train_coarse, model_folder, 'combined_coarse', morphlex_embeddings, args.epochs_coarse_grained)
        tagger_fine = ABLTagger(VocabCharacters, VocabWords, VocabTagsFine, WordFrequency, morphlex_embeddings, coarse_embeddings, args)
        train_tagger(tagger_fine, train_fine, model_folder, 'combined_fine', morphlex_embeddings, args.epochs_fine_grained)
    else:
        if args.training_type == 'coarse':
            training_file = './data/' + args.model + '.coarse'
            suffix = 'coarse'
            epochs = args.epochs_coarse_grained
        elif args.training_type == 'fine':
            training_file = './data/' + args.model + '.fine'
            suffix = 'fine'
            epochs = args.epochs_fine_grained

        train = list(Utils.read(training_file))
        words, word_frequency, tags = Utils.create_vocabularies(training_file)
        with open(model_folder + 'words.txt', "w") as word_freq_file:
            for i in words:
                word_freq_file.write(i.strip() + '\t' + str(word_frequency[i]) + '\n')
        with open(model_folder + 'tags_' + suffix + '.txt', "w") as tag_file:
            for i in tags:
                tag_file.write(i.strip() + '\t')
        shutil.copy2(args.use_morphlex, model_folder + 'morphlex.txt')

        VocabWords, WordFrequency = Utils.build_word_dict(list(Utils.read(model_folder + 'words.txt')))
        print("Known tokens loaded")
        VocabTags = Utils.build_vocab_tags(model_folder + 'tags_' + suffix + '.txt')
        print("Tags loaded")
        Utils.save_args(args, model_folder + 'args.' + args.training_type)
        tagger = ABLTagger(VocabCharacters, VocabWords, VocabTags, WordFrequency, morphlex_embeddings, None, args)
        train_tagger(tagger, train, model_folder, suffix, morphlex_embeddings, epochs)
