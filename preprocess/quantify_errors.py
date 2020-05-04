import argparse
import sys
import csv
from prettytable import PrettyTable

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_folder', '-data', help="Folder with evaluation files", default='../data/')
parser.add_argument('--count', '-cnt', help="Number of most common errors", default=20)
parser.add_argument('--output_type', '-output', help='Select output type: top-errors, all-errors, top-incorrect, all-incorrect.',
                    choices=['top-errors', 'all-errors', 'top-incorrect', 'all-incorrect'], default="top-errors")

args = parser.parse_args()

tagger_tags_ending = 'PM.txt__tagger_out'

correct_tag_ctr = 0
tag_dict = {}
incorrect_tag_ctr = 0
incorrect_tag_dict = {}
count = int(args.count)
tag_correctly = {}

i = 0
while i < 10:
    i += 1
    tagger_tags_file = open(args.data_folder + str(format(i, '02')) + tagger_tags_ending, 'r')
    tagger_tags = tagger_tags_file.readlines()

    j = 0
    while j < len(tagger_tags):
        if len(tagger_tags[j].strip()) > 0:
            ot = tagger_tags[j].strip().split()[1]
            tt = tagger_tags[j].strip().split()[2]
            try:
                tag_dict[ot] += 1
            except:
                tag_dict[ot] = 1

            if ot == tt:
                correct_tag_ctr += 1
                try:
                    tag_correctly[ot] += 1
                except:
                    tag_correctly[ot] = 1
            else:
                incorrect_tag_ctr += 1
                incorrect_str = ot + ' -> ' + tt
                try:
                    incorrect_tag_dict[incorrect_str] += 1
                except:
                    incorrect_tag_dict[incorrect_str] = 1
        j += 1

error_list = [(k, v) for k, v in incorrect_tag_dict.items()]
error_sorted = sorted(error_list, key=lambda x: int(x[1]), reverse=True)

if args.output_type == 'top-errors':

    t = PrettyTable(['correct -> incorrect', 'count', '% of errors', 'Correct count total', '% of tag'])

    print('Total: ' + str(correct_tag_ctr + incorrect_tag_ctr))
    print('Incorrect: ' + str(incorrect_tag_ctr))
    print('Correct: ' + str("{:.2f}%".format(float(100*(correct_tag_ctr) / (correct_tag_ctr + incorrect_tag_ctr)))))


    #for k in error_sorted:
    for i in range(count):
        k = error_sorted[i]
        error_precentage = float((100*k[1]) / incorrect_tag_ctr)
        correct_tag = k[0].split()[0]
        error_percentage_of_correct_tag = float(100*k[1]/tag_dict[correct_tag])
        t.add_row([str(k[0]), str(k[1]), str("{:.2f}%".format(error_precentage)), str(correct_tag) + ':' + str(tag_dict[correct_tag]), str("{:.2f}%".format(error_percentage_of_correct_tag))])
    print(t)

elif args.output_type == 'all-errors':
    for k in error_sorted:
        error_precentage = float((100*k[1]) / incorrect_tag_ctr)
        print(str(k[0]) + '\t' + str(k[1]) + '\t' + str("{:.2f}%".format(error_precentage)))

elif args.output_type == 'top-incorrect':
    t = PrettyTable(['tag', 'occurrance count', '% correctly tagged'])

    incorrect_list = []
    for tag in tag_dict.keys():
        try:
            incorrect_list.append((tag, tag_dict[tag], 100*(tag_correctly[tag]/tag_dict[tag])))
        except:
            incorrect_list.append((tag, tag_dict[tag], 0))
    incorrect_sorted = sorted(incorrect_list, key=lambda x: int(x[2]))
    for i in range(count):
        t.add_row([incorrect_sorted[i][0], str(incorrect_sorted[i][1]), str("{:.2f}%".format(incorrect_sorted[i][2]))])
    print(t)

elif args.output_type == 'all-incorrect':
    incorrect_list = []
    for tag in tag_dict.keys():
        try:
            incorrect_list.append((tag, tag_dict[tag], 100*(tag_correctly[tag]/tag_dict[tag])))
        except:
            incorrect_list.append((tag, tag_dict[tag], 0))
    incorrect_sorted = sorted(incorrect_list, key=lambda x: int(x[2]))
    for i in incorrect_sorted:
        print(i[0] + '\t' + str(i[1]) + '\t' + str("{:.2f}%".format(i[2])))

