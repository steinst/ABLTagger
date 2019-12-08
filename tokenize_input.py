#! /usr/bin/env python3

"""
    ABLTagger: Augmented BiDirectional LSTM Tagger

    Script for tokenizing input, using the tokenizer from Reynir.
    (https://github.com/mideind/Tokenizer)

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

from tokenizer import split_into_sentences
import argparse
import sys

# This script tokenizes an input file. Using the tokenizer module from Mideind, this script does the same
# as invoking the tokenizer directly from the command line:
#       $ tokenize input.txt output.txt
#

if __name__ == '__main__':
    # reading input parameters
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', '-o', help='Select suffix for output files.', default=".tokenized")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--input', '-i', nargs='+', required=True, default=argparse.SUPPRESS,
                               help="File(s) to tokenize before tagging.")

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)
    try:
        args = parser.parse_args()
    except:
        sys.exit(0)

    filename = args.input

    for current_file in args.input:
        with open(current_file + args.output, "w") as f:
            line_list = []
            for line in open(current_file):
                if len(line.strip()) > 0:
                    g = split_into_sentences(line.strip())
                    for sentence in g:
                        f.write(sentence + '\n')
