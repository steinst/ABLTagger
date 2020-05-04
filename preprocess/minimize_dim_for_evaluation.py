import argparse

data_folder = '../data/mim_gull/'
dim_folder = '../extra/'
train_test_files = ['01PM.plain','01TM.plain']
training_set = 'mim_gull'
dim_in = 'dmii.vectors'
dim_out = 'dmii.vectors.' + training_set

training_dict = {}
for i in train_test_files:
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

