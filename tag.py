#! /usr/bin/env python3

"""
    ABLTagger: Augmented BiDirectional LSTM Tagger

    Script for tagging using trained models

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

import argparse
import os, sys, shutil
from colored import fg, attr
import nltk


def read_arguments_from_file(filename):
    parser2 = argparse.ArgumentParser(fromfile_prefix_chars='@')
    parser2.add_argument("--optimization")
    parser2.add_argument("--noise", type=float)
    parser2.add_argument("--learning_rate", type=float)
    parser2.add_argument("--learning_rate_decay", type=float)
    parser2.add_argument("--learning_rate_max", type=float)
    parser2.add_argument("--learning_rate_min", type=float)
    parser2.add_argument("--dropout", type=float)
    parser2.add_argument("--epochs_coarse_grained", type=int)
    parser2.add_argument("--epochs_fine_grained", type=int)
    parser2.add_argument("--use_morphlex")
    parser2.add_argument("--load_characters")
    parser2.add_argument("--load_coarse_tagset")
    return parser2.parse_args(['@'+filename])


def tag_simple(input, output, tagger):
    input_file = open(input, 'rt')
    input_text = input_file.readlines()
    input_file.close()
    output_file = input + output

    with open(output_file, "w") as f:
        for i in input_text:
            simple_tokens = nltk.word_tokenize(i)
            if simple_tokens[0][0].isupper() and not simple_tokens[0] in tagger.vw.w2i:
                simple_tokens[0] = simple_tokens[0][0] + simple_tokens[0][1:]
            f.write("\n".join([x[0] + "\t" + x[1] for x in tagger.tag_sent(simple_tokens)]) + '\n')
            f.write("\n")


def tag_augmented(input, output, tagger):
    input_file = open(input, 'rt')
    input_text = input_file.readlines()
    input_file.close()
    output_file = input + output

    with open(output_file, "w") as f:
        tokens = []
        tags = []
        for i in input_text:
            if i.strip() == '':
                if tokens[0][0].isupper() and not tokens[0] in tagger.vw.w2i:
                    tokens[0] = tokens[0][0] + tokens[0][1:]
                f.write("\n".join([x[0] + "\t" + x[1] for x in tagger.tag_sent(tokens, tags)]) + '\n')
                f.write("\n")
                tokens = []
                tags = []
            else:
                tokens.append(i.split()[0])
                tags.append(i.split()[1].strip())


if __name__ == '__main__':
    # reading input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', help="Select model name. Saved in ./models/[model-name]/", default="Full")
    parser.add_argument('--output', '-o', help='Select suffix for output files.', default=".tagged")
    parser.add_argument('--tag_type', '-type', help='Select tagging type', choices=['coarse', 'fine', 'combined'], default="combined")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--input', '-i', nargs='+', required=True, default=argparse.SUPPRESS,
                               help="File(s) to tag. Files should include tokenized sentences. One sentence per line. Each token followed by whitespace. (Example: Þetta er tókað .) [Required]")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    from ABLTagger import ABLTagger, Embeddings, Vocab, Utils

    # model file names
    model_folder = './models/' + args.model + '/'
    chars_file = model_folder + 'characters.txt'
    words_file = model_folder + 'words.txt'
    tags_coarse_file = model_folder + 'tags_coarse.txt'
    tags_fine_file = model_folder + 'tags_fine.txt'
    morphlex = model_folder + 'morphlex.txt'
    coarse_emb_file = model_folder + 'coarse_tagset.txt'

    # load training time model arguments
    args = argparse.Namespace(**vars(args), **vars(read_arguments_from_file(model_folder + 'args.' + args.tag_type)))

    # load words and tagsets used at training time
    if args.tag_type in ('combined', 'coarse'):
        VocabTagsCoarse = Utils.build_vocab_tags(tags_coarse_file)

    if args.tag_type in ('combined', 'fine'):
        coarse_embeddings = Embeddings(coarse_emb_file)
        VocabTagsFine = Utils.build_vocab_tags(tags_fine_file)

    train_words = list(Utils.read(words_file))
    char_list = []
    for charline in open(chars_file):
        characters = charline.strip().split('\t')
        for c in characters:
            char_list.append(c)

    morphlex_embeddings = Embeddings(morphlex)
    print("Morphological lexicon embeddings in place")
    VocabCharacters = Vocab.from_corpus([char_list])
    VocabWords, WordFrequency = Utils.build_word_dict(train_words)

    # Select tagging type
    if args.tag_type == 'combined':
        tagger_coarse = ABLTagger(VocabCharacters, VocabWords, VocabTagsCoarse, WordFrequency, morphlex_embeddings, None, args)
        print(fg('blue'), "Loading pre-tagger..." + attr('reset') + attr('reset'), end='\r')
        tagger_coarse.load_model(model_folder + 'model.combined_coarse')

        tagger_fine = ABLTagger(VocabCharacters, VocabWords, VocabTagsFine, WordFrequency, morphlex_embeddings, coarse_embeddings, args)
        print(fg('blue'), "Loading tagger...    " + attr('reset') + attr('reset'), end='\r')
        tagger_fine.load_model(model_folder + 'model.combined_fine')

        for i in args.input:
            tag_simple(i, args.output, tagger_coarse)
            tag_augmented(i+args.output, args.output, tagger_fine)
            os.remove(i+args.output)
            shutil.move(i+args.output+args.output, i+args.output)
        print('Done!              ')

    else:
        if args.tag_type == 'coarse':
            tagger = ABLTagger(VocabCharacters, VocabWords, VocabTagsCoarse, WordFrequency, morphlex_embeddings, None, args)
            print(fg('blue'), "Loading tagger..." + attr('reset') + attr('reset'), end='\r')
            tagger.load_model(model_folder + 'model.coarse')

            for i in args.input:
                tag_simple(i, args.output, tagger)
            print('Done!              ')

        elif args.tag_type == 'fine':
            tagger = ABLTagger(VocabCharacters, VocabWords, VocabTagsFine, WordFrequency, morphlex_embeddings, None, args)
            print(fg('blue'), "Loading tagger..." + attr('reset') + attr('reset'), end='\r')
            tagger.load_model(model_folder + 'model.fine')

            for i in args.input:
                tag_augmented(i+args.output, args.output, tagger)
            print('Done!              ')
