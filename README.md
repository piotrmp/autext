# Autextification

## Feature `word_frequency` dependencies:

This feature requires word frequency matrix from Google Book Corpus, all the data can be automatically downloaded and processed using the python file `download_book_corpus_ngram.pyp`, however this is time consuming.

The concatenated and processed files can be downloaded from the folowing links:
- Spanish: https://drive.google.com/file/d/1jsFPoYlCf9U8BfKBASnrs-OtNEqySnft/view?usp=share_link
- English: https://drive.google.com/file/d/1PwcEHgR8jU3M9_2QHCW0_NO0rrc2kRw1/view?usp=sharing

Please, create a folder in the root of the repository named `Resources/` with two subfolders `en`and `es`, and store each of the tsv.gz files with the matrixes in each subfolder, the feature class `word_ferquency.py` will read the matrix from those folders and it will creatre a defaultdict.