# Autextification

## Feature `word_frequency` dependencies:

This feature requires word frequency matrix from Google Book Corpus, all the data can be automatically downloaded and processed using the python file `download_book_corpus_ngram.pyp`, however this is time consuming.

The concatenated and processed files can be downloaded from the folowing links:
- Spanish: https://drive.google.com/file/d/1jsFPoYlCf9U8BfKBASnrs-OtNEqySnft/view?usp=share_link
- English: https://drive.google.com/file/d/1PwcEHgR8jU3M9_2QHCW0_NO0rrc2kRw1/view?usp=sharing

Please, create a folder in the root of the repository named `resources/` with two subfolders `en`and `es`, and store each of the tsv.gz files with the matrixes in each subfolder, the feature class `word_ferquency.py` will read the matrix from those folders and it will creatre a defaultdict.

## Feature `grammar` dependencies:

This feature requires pre-computed grammar check, all the data can be downloaded from:
- both subtasks and train+test and en+es are in: https://drive.google.com/file/d/1-JM_k18UukKqp6-rMO15qpc_MG8yu4BG/view?usp=share_link

Please, store the tsv.gz file in `resources/`  the feature class `grammar.py` will read the file from the folder and it will creatre a dict.
