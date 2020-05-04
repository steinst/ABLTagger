import argparse
import csv
from prettytable import PrettyTable

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_folder', '-data', help="Folder with evaluation files", default='../evaluate/')
parser.add_argument('--corpus', '-c', help="Name of training corpus", default='')
parser.add_argument('--training_parameters', '-tp', help="parameters in file name", default='combined_SimpleSGD_0.13')
args = parser.parse_args()

i = 0

total_accuracy = {}
known_accuracy = {}
unknown_accuracy = {}
word_count = {}

total_count = {}
while i < 10:
    i += 1
    word_list = open('../data/' + str(format(i, '02')) + 'PM.txt', 'r')
    word_lines = word_list.readlines()
    i_counter = 0
    for curr_line in word_lines:
        if len(curr_line.strip()) > 0:
            i_counter += 1

    try:
        with open(args.data_folder + args.corpus + '_' + str(format(i, '02')) + '_' + args.training_parameters, newline='') as csvfile:
            filereader = csv.reader(csvfile, delimiter='\t')
            for row in filereader:
                try:
                    total_accuracy[row[2]]['count'] += 1
                    total_accuracy[row[2]]['accuracy'] += float(float(row[3])*float(i_counter))
                    known_accuracy[row[2]]['count'] += 1
                    known_accuracy[row[2]]['accuracy'] += float(float(row[5])*float(i_counter))
                    unknown_accuracy[row[2]]['count'] += 1
                    unknown_accuracy[row[2]]['accuracy'] += float(float(row[6])*float(i_counter))
                    total_count[row[2]] += i_counter
                except Exception as e:
                    total_accuracy[row[2]] = {'count':1, 'accuracy':float(float(row[3])*float(i_counter))}
                    known_accuracy[row[2]] = {'count':1, 'accuracy':float(float(row[5])*float(i_counter))}
                    unknown_accuracy[row[2]] = {'count':1, 'accuracy':float(float(row[6])*float(i_counter))}
                    total_count[row[2]] = i_counter
    except:
        pass

t = PrettyTable(['Epoch', 'Accuracy', 'Known words', 'Unknown words', 'Count'])
for i in total_accuracy.keys():
    #accuracy = 100*float(total_accuracy[i]['accuracy']/total_accuracy[i]['count'])
    accuracy = 100*float(total_accuracy[i]['accuracy']/total_count[i])
    #known = 100*float(known_accuracy[i]['accuracy']/known_accuracy[i]['count'])
    known = 100*float(known_accuracy[i]['accuracy']/total_count[i])
    #unknown = 100*float(unknown_accuracy[i]['accuracy']/unknown_accuracy[i]['count'])
    unknown = 100*float(unknown_accuracy[i]['accuracy']/total_count[i])
    t.add_row([str(i),str("%.2f" % accuracy),str("%.2f" % known),str("%.2f" % unknown),str(total_accuracy[i]['count'])])
    #print(str(i) + '\t' + str("%.2f" % accuracy) + '\t' + str("%.2f" % known) + '\t' + str("%.2f" % unknown) + '\t' + str(total_accuracy[i]['count']))
print(t)

