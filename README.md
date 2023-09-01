# Autextification

## Feature `word_frequency` dependencies:

This feature requires word frequency matrix from Google Book Corpus, all the data can be automatically downloaded and processed using the python file `download_book_corpus_ngram.pyp`, however this is time consuming.

The concatenated and processed files can be downloaded from the folowing links:
- Spanish: https://drive.google.com/file/d/1jsFPoYlCf9U8BfKBASnrs-OtNEqySnft/view?usp=share_link
- English: https://drive.google.com/file/d/1PwcEHgR8jU3M9_2QHCW0_NO0rrc2kRw1/view?usp=sharing

Please, create a folder in the root of the repository named `resources/` with two subfolders `en`and `es`, and store each of the tsv.gz files with the matrixes in each subfolder, the feature class `word_ferquency.py` will read the matrix from those folders and it will creatre a defaultdict.

## Feature `grammar` dependencies:

This feature requires running [LanguageTool](https://github.com/languagetool-org/languagetool) in local server, port 8010. We use the dockerised version , [Erikvl87/docker-languagetool](https://github.com/Erikvl87/docker-languagetool) to start the grammar check service.

Here the steps to run LanguageTool in local server: 
```
docker pull erikvl87/languagetool
docker run --rm -p 8010:8010 erikvl87/languagetool
```

We use [language-tool-python](https://pypi.org/project/language-tool-python/#description) python wrapper to query the local API.

Please, store the tsv.gz file in `resources/`  the feature class `grammar.py` will read the file from the folder and it will creatre a dict.
