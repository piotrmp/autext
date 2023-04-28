from bs4 import BeautifulSoup
import requests   
import regex as re
from tqdm import tqdm
import pandas as pd
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import os
import argparse
import gzip
import glob

## Execution examples for English and Spanish
# python download_book_corpus_ngram.py --lang es --url http://storage.googleapis.com/books/ngrams/books/20200217/spa/spa-1-ngrams_exports.html
# python download_book_corpus_ngram.py --lang en --url http://storage.googleapis.com/books/ngrams/books/20200217/eng/eng-1-ngrams_exports.html

parser = argparse.ArgumentParser(description='Download data from Google Books Ngram')
parser.add_argument('-l','--lang', help='language', required=True, choices=['en','es'])
parser.add_argument('-url','--url', help='url', required=True)
args = vars(parser.parse_args())

url = args['url']
lang = args['lang']

# Request page with links to download
response = requests.get(url)
soup = BeautifulSoup(response.text, features="lxml")
# Search for all li elements in the page and get the href attribute (links)
list_links = soup.find_all('li')
my_links = [link.a['href'] for link in list_links]

# Download all files, checking if they already exist
print("Downloading files from Google Books Ngram")
for link in tqdm(my_links):
	filename = link.split("/")[-1]
	if os.path.exists(f"resources/{lang}/{filename}")==False:
		with open(f"resources/{lang}/{filename}", "wb") as f:
			r = requests.get(link)
			f.write(r.content)

# Unzip files, read the content and write it to a new file
print("Processing files from Google Books Ngram")
file = open(f"resources/{lang}/matrix.txt", "w")

for fname in tqdm(glob.glob(f'resources/{lang}/*.gz')):
    with gzip.open(fname,'rt') as f:
        for line in f:
            word = line.strip().split('\t')[0]
            freq = sum([int(l.split(',')[1]) for l in line.strip().split('\t')[1:]])
            file.write(f"{word}\t{freq}\n")

# To read the file with all the words and frequencies from all the ziped files
print("Cleaning files from Google Books Ngram")

lang='es'
with open(f"resources/{lang}/matrix.txt") as f:
    lines = f.readlines()

# Split by lines and by tab, take the first part of the line (word) and the second part (freq), casting to int freq and removing POS tag from word
lines_list = [(str(l.strip().split('\t')[0]).split('_')[0],int(l.strip().split('\t')[1])) for l in lines]
lines_df = pd.DataFrame(lines_list, columns=['word','freq'])
# Group by word and sum freq
lines_grouped_df = lines_df.groupby('word').freq.sum().reset_index()
# Store the file compressed
lines_grouped_df.to_csv(f"resources/{lang}/word_freq_matrix.tsv.gz", compression='gzip', sep='\t', quotechar='"', index=False)
