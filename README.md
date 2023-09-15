# Automatically-Generated Text Identification

This repository contains the source code for training a model identifying text that is automatically generated thourgh a language model (e.g. ChatGPT), rather than written by a human, as wellas  identifying the models used for generation.

The solution presented here was prepared within [AuTexTification: Automated Text Identification shared task](https://sites.google.com/view/autextification/home), a part of [IberLEF 2023, the 5th Workshop on Iberian Languages Evaluation Forum](http://sepln2023.sepln.org/en/iberlef-en/) at the SEPLN 2023 Conference, held in Jaén, Spain on the 26th of September, 2023. The [results](https://sites.google.com/view/autextification/results?authuser=0) show our solution (*TALN-UPF*) ranked as the 1st in differentiating between human- and machine-generated text, both for Spanish and English.

The full description of our work is available in the article presented at the conference ([I've Seen Things You Machines Wouldn't Believe: Measuring Content Predictability to Identify Automatically-Generated Text](TODO)). The source code is configured to reproduce the best-performing *Pred+FLM+Add* variant, but can be adjusted to test other configurations.

The research was done within the [ERINIA](https://www.upf.edu/web/erinia) project realised at the
[TALN lab](https://www.upf.edu/web/taln/) of [Universitat Pompeu Fabra](https://www.upf.edu).

## Using the code

1. *Dependencies*: You will need to install the packages we depend on: ```torch```, ```numpy```, ```transformers```, ```tqdm```, ```language_tool_python```, ```pandas``` and ```sklearn```.
2. *Resources*: The additional features (grammar and word frequency need special resources, see below).
3. *Data*: If you want to replicate the AuTexTification experiments, you will need the datasets for [training](https://zenodo.org/record/7692961) and [test](https://zenodo.org/record/7846000).
4. *Data split*: You might also need to set some data aside as a development portion. This is done through ```traindev_folds_path``` file (see details in code). We used LDA topic analysis for this purpose.
5. *Training and evaluation* is performed by the ```main_final.py``` script.

### Resources for word frequency features

This feature requires word frequency matrix from Google Book Corpus, all the data can be automatically downloaded and processed using the python file `download_book_corpus_ngram.py`, however this is time-consuming.

The concatenated and processed files can be downloaded from the folowing links:
- Spanish: https://drive.google.com/file/d/1jsFPoYlCf9U8BfKBASnrs-OtNEqySnft/view?usp=share_link
- English: https://drive.google.com/file/d/1PwcEHgR8jU3M9_2QHCW0_NO0rrc2kRw1/view?usp=sharing

Create a folder in the root of the repository named `resources/` with two subfolders `en`and `es`, and store each of the tsv.gz files with the matrixes in each subfolder, the feature class `word_frequency.py` will read the matrix from these folders.

### Resources for grammar features

This feature requires running [LanguageTool](https://github.com/languagetool-org/languagetool) in local server, port 8010. We recommend to use the following dockerised version [Erikvl87/docker-languagetool](https://github.com/Erikvl87/docker-languagetool) to start the grammar check service.

These are the steps to run LanguageTool in your local server: 
```
docker pull erikvl87/languagetool
docker run --rm -p 8010:8010 erikvl87/languagetool
```

Then, the `autext` functions use [language-tool-python](https://pypi.org/project/language-tool-python/#description) python wrapper to query the local API.

## Licence

BODEGA code is released under the [GNU GPL 3.0](https://www.gnu.org/licenses/gpl-3.0.html) licence.

## Funding

The [ERINIA](https://www.upf.edu/web/erinia) project has received funding from the European Union’s Horizon Europe
research and innovation programme under grant agreement No 101060930.

Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the
European Union. Neither the European Union nor the granting authority can be held responsible for them.