import numpy

data_folder = '../data/'
dim_folder = '../extra/'
training_files = ['01PM.txt','01TM.txt']
training_set = 'otb'
dim_in = 'dmii.or'
dim_out = 'dmii.or.' + training_set

training_dict = {}
for i in training_files:
    curr_file = open(data_folder + i, 'r')
    training_lines = curr_file.readlines()
    for j in training_lines:
        try:
            word_split = j.split()
            training_dict[word_split[0]] = 1
        except:
            pass

print(len(training_dict))
dim_file = open(dim_folder + dim_in, 'r')
dim_lines = dim_file.readlines()

with open(dim_folder + dim_out, "w") as f:
    for dim_wordform in dim_lines:
        try:
            temp = dim_wordform.split(';')[0]
            if temp in training_dict.keys():
                f.write(dim_wordform)
        except:
            pass
