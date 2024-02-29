
# Overview

In this project we have decided to work with PubMed data to build a Question Answering system to provide a accurate and cocise answers to users based on the downloaded PubMed data during a specific duration from 2013 -> 2024.We are using in this project NLP techniques for instance embedings, storing, loading, chunking, indexing, information retrival, conversation retrival chain to capture semantics about the text data which allows the model to understand, interpret and generate a new answer from the retrived question.

## Our approach has several steps

- Gathering the data from a websource is a challenging part. we have implmented a function called search_and_download_abstracts function automates the process of searching for and downloading abstracts from the PubMed database. It utilizes the Entrez API to perform a search query based on specified parameters and then downloads the abstracts found within a given date range. This function is particularly useful for researchers needing to gather large volumes of scientific journals efficiently. given that there is a limitation on the website 10,000 record per request. API_KEYS & EMAIL are stored in local file instead of being hard coded thats why we are using 'dotenv package' for loading the environment variables.

- cleaning the raw data, there are several steps in cleaning the raw data:

1. lowercase to all text which allows to have only one representation to for each word.
2. tokenization: transform raw text into a format that assits in creating embeddings.
3. Normalization: preform stemming or lemmatization, which reduce words to their base or root form

- data loading, processing, embedding, and storing results in a vector database for efficient retrieval.

1. vectore database creation: Setting up a directory for the vector database.
2. loading and processing PubMed Data: Reading and processing PubMed data for analysis.
3. Extraction and chunking of text sections: Extracting sections from text files and chunking them for processing.
4. Embedding text chunks: the process of transforming segments of text into numerical vectors. which captures the semantic meaning of the text, allows machines to interpret and process.
5. Storing the FAISS index Results: Saving the embedded chunks to a vector database for querying.
