# ABLTagger
ABLTagger is a bidirectonal LSTM Part-of-Speech Tagger with combined Word and Character embeddings, augmented with a morphological lexicon and a lexical category identification step.

ABLTagger is described in: _Steinþór Steingrímsson, Örvar Kárason and Hrafn Loftsson. 2019. Augmenting a BiLSTM tagger with a Morphological Lexicon and a Lexical Category Identification Step_

In the paper we show how we achieve the highest accuracy reported for part-of-speech tagging using a fine grained Icelandic tagset, by augmenting the tagger with a morphological lexicon, The Database of Modern Icelandic Inflection (DMII), and running a pre-tagging step using a very coarse grained tagset.   

## Training models
A bidirectional LSTM model can be trained with the script `train.py`.

![training](https://user-images.githubusercontent.com/24220374/50144035-11532880-02a6-11e9-8999-893e7ab94c2d.gif)

Running `./train.py -h` gives information on all possible parameters.

The program requires input corpora to be in the same format as the IFD-training/testing sets, available at http://www.malfong.is/index.php?lang=en&pg=ordtidnibok.
If the DMII is used it has to be formatted in a specific way for the tagger to be able to use it. Running `./prep_dmii.py -h` gives information on the preprocessing.
If the lexical category identification step is used, or other coarse grained tags for identification, the data has to be prepared beforehand.  Running `./prep_dmii.py -h` gives information on how to replicate what we did in the experiments described in the paper.

## Tagging texts
For tagging Icelandic texts two models are provided in the `./models/` folder: _Full_ and _Light_. _Full_ is trained on the whole DMII but _Light_ is trained on a subset of DMII, to reduce memory requirements and time when rebuilding and populating the tagging models.
Running `./test.py -h` gives information on all possible parameters.


## Evaluating models
Training/testing sets can be evaluated with the script `evaluate.py`.

Running `./evaluate.py -h` gives information on all possible parameters.

The script writes results to a file in the `./evaluate/` folder. During long running training sessions `get_results.py` can be used to fetch the latest results and calculate average scores for all sets for each epoch.

![get_results](https://user-images.githubusercontent.com/24220374/50121118-a8d95c80-024f-11e9-9064-41cca53c97c5.png)


## Trying out a PoS tagger

The script `interactive.py` loads a model and then allow the user to try out the tagger on sentences he enters. Running `./interactive.py -h` gives information on all possible parameters.

![interactive](https://user-images.githubusercontent.com/24220374/50121119-a8d95c80-024f-11e9-902f-44364692cea7.png)

Trained models are available:

* Full (Best model to use for tagging)
    - A model trained on all training data used in the paper cited above, taking advantage of the whole DMII morphological lexicon. This model needs at least 16GB RAM to load.
    - Download link: https://www.dropbox.com/s/8oya0nse3o4xxoy/models_Full.zip?dl=0 (360 MB download - 6.7 GB uncompressed)
    - The model should go into a folder called ./models/Full

* Light (Only for testing the tagger)
    - A model trained on all training data used in the paper cited above, but only taking advantage of a subset of the DMII morphological lexicon. This model needs less than 8GB RAM to load.
    - Download link: https://www.dropbox.com/s/hdp0kjkb46n5i1r/models_Light.zip?dl=0 (275 MB download - 770 MB uncompressed)
    - The model should go into a folder called ./models/Light

Both models need the contents of https://www.dropbox.com/s/ysmn9or0n0zytwi/extra.zip?dl=0 to be in the ./extra folder.


## Input formats

[training data]

[text to tag]

[morphological lexicon]

[information on coarse grained tagset]

...
