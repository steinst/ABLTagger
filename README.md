# ABLTagger
ABLTagger is a bidirectonal LSTM Part-of-Speech Tagger with combined Word and Character embeddings, augmented with a morphological lexicon and a lexical category identification step. The work is described in the paper [Augmenting a BiLSTM Tagger with a Morphological Lexicon and a Lexical Category Identification Step](https://www.aclweb.org/anthology/R19-1133/)

If you find this work useful in your research, please cite the paper: 

@inproceedings{steingrimsson-etal-2019-augmenting,
    title = "Augmenting a {B}i{LSTM} Tagger with a Morphological Lexicon and a Lexical Category Identification Step",
    author = {Steingr{\'\i}msson, Stein{\th}{\'o}r  and
      K{\'a}rason, {\"O}rvar  and
      Loftsson, Hrafn},
    booktitle = "Proceedings of the International Conference on Recent Advances in Natural Language Processing (RANLP 2019)",
    month = sep,
    year = "2019",
    address = "Varna, Bulgaria",
    url = "https://www.aclweb.org/anthology/R19-1133",
    doi = "10.26615/978-954-452-056-4_133",
    pages = "1161--1168",
}

The paper describes a method for achieving high accuracy in part-of-speech tagging a fine grained tagset. We show how the method is used to reach the highest accuracy reported for PoS-tagging Icelandic. The tagger is augmented by using a morphological lexicon, [The Database of Icelandic Morphology (DIM)](https://www.aclweb.org/anthology/W19-6116/), and by running a pre-tagging step using a very coarse grained tagset induced from the fine grained data.   

# Training
Before training make sure the requirements in `requirements.txt` are set up.
## Preparing the data
### Training set
The training data is a text file in the ./data/ folder. The file contains PoS-tagged sentences. The file has one token per line, as well as its corresponding tag. The sentences are separated by an empty line. 

```
Við     fp1fn
höfum   sfg1fn
góða    lveosf
aðstöðu nveo
fyrir   af
barnavagna      nkfo
og      c
kerrur  nvfo
.       pl

Börnin  nhfng
geta    sfg3fn
sofið   sþghen
úti     aa
ef      c
vill    sfg3en
.       pl
```

In the paper we use the training sets from [The Icelandic Frequency Dictionary](http://www.malfong.is/index.php?lang=en&pg=ordtidnibok) and the [MIM-GOLD](http://www.malfong.is/index.php?lang=en&pg=gull). We download the training files and to make sure they are correctly formed (no spaces or extra symbols in the lines that should be empty), we run `./preprocess/generate_fine_training_set.py` on the training file. In order to run the lexical category identification step we also generate a coarse grained training set from the data, by running `./preprocess/generate_coarse_training_set.py`.
```
python3 ./preprocess/generate_fine_training_set.py 
```
The script can take two parameters:
| Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --input 	       |	./data/Full.txt           |The name of the original gold standard file
| -o  --output          | ./data/Full.fine.txt           |The name of the file which will be used for training. `Full` will be the name of the model to be trained.

```
python3 ./preprocess/generate_coarse_training_set.py 
```
The script can take two parameters:
| Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --input 	       |	./data/Full.fine.txt           |The name of file containing the training set and fine grained tags.
| -o  --output          | ./data/Full.coarse.txt           |The output file containing coarse grained tags.

When training with the coarse grained data, the file `word_class_vectors.txt` is needed. It is stored in the `./extra/` directory. The file contains one-hot vectors for all possible coarse-grained tags. 

### Morphological lexicon
We represent the information contained in the morphological lexicon with n-hot vectors. To generate the n-hot vectors, different scripts will have to be written for different morphological lexicons. We use the DIM morphological lexicon for Icelandic. The `./preprocess/` folder contains a script, `vectorize_dim.py`, to create n-hot vectors from DIM. We first [download the data in SHsnid format](https://bin.arnastofnun.is/django/api/nidurhal/?file=SHsnid.csv.zip). After unpacking the `SHsnid.csv` file is copied into `./data/`. To generate the n-hot vectors we run the script:
```
python3 ./preprocess/vectorize_dim.py 
```
The script can take two parameters:
| Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --input 	       |	./data/SHsnid.csv           |The file containing the DIM morphological lexicon in SHsnid format.
| -o  --output          | ./extra/dmii.vectors           |The file containing the DIM n-hot vectors.

## Training models
A model can be trained with the script `train.py`. The program requires input corpora to be in the same format as the IFD-training/testing sets, as described above.
Running `./train.py -h` gives information on all possible parameters. The default parameters are the ones used in the paper. One parameter, model, is required. It is for the name of the model. In out example we call the model `Full`.
```
python3 ./train.py -m Full 
```

| Required Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -m --model 	       |	None           | The name of the model being trained.

| Optional Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -o --optimization 	       |	SimpleSGD           | Optimization algorithm to use. Available algorithms are: SimpleSGD, MomentumSGD, CyclicalSGD, Adam, RMSProp.
| -lr --learning_rate 	       |	0.13           | Learning rate
| -lrd --learning_rate_decay 	       |	0.05           | Learning rate decay
| -l_max --learning_rate_max 	       |	0.1           | Learning rate max for Cyclical SGD
| -l_min --learning_rate_min 	       |	0.01           | Learning rate min for Cyclical SGD
| -d --dropout 	       |	0.05           | Dropout rate
| -n --noise 	       |	0.1           | Noise in embeddings
| -morphlex --use_morphlex 	       |	./extra/dmii.vectors           | File with morphological lexicon embeddings in ./extra folder.
| -load_chars --load_characters 	       |	./extra/characters_training.txt           | File to load characters from
| -load_coarse --load_coarse_tagset 	       |	./extra/word_class_vectors.txt           | Load embeddings file for coarse grained tagset
| -type --training_type 	       |	combined           | Select training type: coarse, fine or combined.
| -ecg --epochs_coarse_grained 	       |	12           | Number of epochs for coarse grained training.
| -efg --epochs_fine_grained 	       |	20           | Number of epochs for fine grained training.

The program runs on a CPU and training with default settings and the two Icelandic corpora combined takes approximately 5 hours on an Intel i9-9900K CPU @ 3.60GHz. As some parts of the training process are memory hungry, 32GB of memory are recommended. 

## Tagging texts
Texts can be tagged using the script `train.py`. The program loads a model stored in the `./models` folder. It can be a model trained by the user or a pre-trained model. A model trained on the IFD and MIM-GOLD combined can be downloaded:

* Full (Best model to use for tagging)
    - A model trained on all training data used in the paper cited above, taking advantage of the whole DMII morphological lexicon. This model needs at least 16GB RAM to load.
    - Download link: https://www.dropbox.com/s/8oya0nse3o4xxoy/models_Full.zip?dl=0 (360 MB download - 6.7 GB uncompressed)
    - The model should go into a folder called ./models/Full

The model needs the contents of https://www.dropbox.com/s/ysmn9or0n0zytwi/extra.zip?dl=0 to be in the ./extra folder.


Running `./tag.py -h` gives information on all possible parameters. At minimum the input file(s) have to be specified, and normally the model is also specified. 
```
python3 ./tag.py -m Full -i text_file.txt
```

| Required Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -i --input 	       |	None           | File(s) to tag. Files should include tokenized sentences. One sentence per line. Each token followed by whitespace.

| Optional Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -m --model 	       |	Full           | Select model. It should be stored in ./models/[model-name]/
| -o --output 	       |	.tagged           | Select suffix for output files.
| -type --tag_type 	       |	combined           | Select tagging type: coarse, fine or combined'.
| --tokenize 	       |	None           | Use the Reynir tokenizer to tokenize input text. Action is invoked by using the parameter.

## Evaluating models
Training/testing sets can be evaluated with the script `evaluate.py`. Before evaluation a script to minimize the DIM, `minimize_dim_for_evaluation.py`, can be run to reduce time spent in training and testing the model. The script finds all word forms in the training/testing data and removes n-hot vectors from the DIM file for words that are not in the training/testing data.
Before evaluating the models the `./preprocess/generate_fine_training_set.py` and `./preprocess/generate_coarse_training_set.py` should be run as described in the previous section, on all train/test files.
To evaluate the accuracy of the tagger on fold number 1 in a set of 10 folds from the mim_gold corpus, the following command does that with all the same settings as used in the paper.
```
python3 ./evaluate.py -c mim_gold -fold 1 -morphles dmii.vectors.mim_gold
```
Running `./evaluate.py -h` gives information on all possible parameters.

| Optional Parameters                 | Default       | Description   |	
| :------------------------ |:-------------:| :-------------|
| -o --optimization 	       |	SimpleSGD           | Optimization algorithm to use. Available algorithms are: SimpleSGD, MomentumSGD, CyclicalSGD, Adam, RMSProp.
| -lr --learning_rate 	       |	0.13           | Learning rate
| -lrd --learning_rate_decay 	       |	0.05           | Learning rate decay
| -l_max --learning_rate_max 	       |	0.1           | Learning rate max for Cyclical SGD
| -l_min --learning_rate_min 	       |	0.01           | Learning rate min for Cyclical SGD
| -d --dropout 	       |	0.0           | Dropout rate
| -data --data_folder 	       |	./data/           | Folder containing training data.
| -morphlex --use_morphlex 	       |	None           | File with morphological lexicon embeddings in ./extra folder.
| -load_chars --load_characters 	       |	./extra/characters_training.txt           | File to load characters from
| -load_coarse --load_coarse_tagset 	       |	./extra/word_class_vectors.txt           | Load embeddings file for coarse grained tagset
| -coarse --coarse_type 	       |	word_class           | Select type of coarse data.
| -type --training_type 	       |	combined           | Select training type: coarse, fine or combined.
| -c --corpus 	       |	otb           | Name of training corpus
| -fold --dataset_fold 	       |	1           | select which dataset to use (1-10)
| -ecg --epochs_coarse_grained 	       |	12           | Number of epochs for coarse grained training.
| -efg --epochs_fine_grained 	       |	20           | Number of epochs for fine grained training.
| -n --noise 	       |	0.1           | Noise in embeddings

The script writes results to files in the `./evaluate/` folder. `./preprocess/calc_accuracy.py` reads these files and gives you the average accuracy over all the folds in a 10-fold validation. `./preprocess/quantify_errors.py` gives you a list of the most common errors made by the tagger.
