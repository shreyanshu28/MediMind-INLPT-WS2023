import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import ray

#read txt file 
# print(os.getcwd())


articles_dict_keys = ['PMID', 'OWN', 'STAT', 'DCOM', 'LR', 'IS', 'VI', 'IP', 'DP', 'TI', 'PG', 'LID', 'AB', 'FAU', 'AU', 'AD', 'LA', 'GR', 'PT', 'DEP', 'PL', 'TA', 'JT', 'JID', 'SB', 'MH', 'PMC', 'MID', 'COIS', 'EDAT', 'MHDA', 'CRDT', 'PHST', 'AID', 'PST', 'SO', 'AUID', 'CIN', 'CI', 'OTO', 'OT']
articles_dict = dict([(key, []) for key in articles_dict_keys])

# print(articles_dict)
# articles_dict = []


#loop over all file names

# path = "pubmed-intelligen-set.txt"
# for filename in os.listdir(path):

with open('pubmed-intelligen-set.txt','r', encoding="utf-8") as f:
    text = f.read()

# Split the text into articles
articles = text.split("\n\n")

# Split each article into lines
for i, article in tqdm(enumerate(articles)):
    lines = article.split("\n")

    # Create an empty dictionary
    dictionary = dict.fromkeys(articles_dict_keys)

    # Loop through each line
    for line in lines:
        # Split the line into key and value using the first hyphen as the delimiter
        # key, value = line.split("-", 1)
        if len(line) > 4 and line[4] == '-':
            key, value = line.split("-", 1)
            key = key.strip()
            value = value.strip()
            dictionary[key] = value
        else:
            key = key.strip()
            value = value.strip()
            dictionary[key] = dictionary[key] + line
        # Strip any whitespace from the key and value
        # Add the key-value pair to the dictionary
    
    # dump the dictionary
    for K in articles_dict_keys:
        articles_dict[K].append(dictionary[K])


df = pd.DataFrame(articles_dict)
print(df.describe())

#select only required columns

ds = ray.data.from_pandas(df)

# print(ds.take_all())
print(len(ds))

