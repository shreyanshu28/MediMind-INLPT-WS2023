import os
import re
# from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import ray
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from functools import partial
from langchain.document_loaders import ReadTheDocsLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from functools import partial
from ray.data import ActorPoolStrategy
import logging
# import psycopg2
# from pgvector.psycopg2 import register_vector
from langchain.vectorstores import FAISS

# Suppress FutureWarnings
import warnings
warnings.filterwarnings("ignore")

# Initialize Ray
os.environ['RAY_worker_register_timeout_seconds'] = '1000'
# ray.init(num_gpus=1, num_cpus = 6, logging_level = logging.ERROR)
ray.init()

ray.data.DataContext.get_current().execution_options.verbose_progress = True

# Step 1: Vector DB creation
# Set the desired output directory for vector DB creation
# EFS_DIR = "pubMed"

# # Create Path object for the directory
# DOCS_DIR = Path(EFS_DIR, "docs.ray.io/en/master/")


# Step 2: Loading Pubmed data and processing
# Read Pubmed data and process
#TODO: resolve multiple tags for authors, locations, MHs etc. add _digit to the end of the key
articles_dict_keys = ['PMID', 'OWN', 'STAT', 'DCOM', 'LR', 'IS', 'VI', 'IP', 'DP', 'TI', 'PG', 'LID', 'AB', 'FAU', 'AU', 'AD', 'LA', 'GR', 'PT', 'DEP', 'PL', 'TA', 'JT', 'JID', 'SB', 'MH', 'PMC', 'MID', 'COIS', 'EDAT', 'MHDA', 'CRDT', 'PHST', 'AID', 'PST', 'SO', 'AUID', 'CIN', 'CI', 'OTO', 'OT']
articles_dict = dict([(key, []) for key in articles_dict_keys])

document_folder = os.path.join("PubMedDataPMFormat","PubMedDataPMFormat")


print("Reading documents...")
#iterate over files inside documents folder:
for file_name in tqdm(os.listdir(document_folder)):
    with open(os.path.join(document_folder,file_name), 'r', encoding="utf-8") as f:
        text = f.read()

    articles = text.split("\n\n")


    for i, article in tqdm(enumerate(articles)):
        lines = article.split("\n")
        dictionary = dict.fromkeys(articles_dict_keys)

        for line in lines:
            if len(line) > 4 and line[4] == '-':
                key, value = line.split("-", 1)
                key = key.strip()
                value = value.strip()
                dictionary[key] = value
            else:
                key = key.strip()
                value = value.strip()
                dictionary[key] = dictionary[key] + line

        for K in articles_dict_keys:
            articles_dict[K].append(dictionary[K])



print("Read complete, generating DataFrame...")
print(f"Length of dataset is: {len(articles_dict['PMID'])}")
df = pd.DataFrame(articles_dict,columns=articles_dict_keys)

#preprocess 
#remove duplicate articles with same title (there are some articles with different PMIDs but same title)
df = df.drop_duplicates(subset='TI', keep='first') 
#remove duplicate articles with same abstract since there are some articles with different PMIDs AND different Titles but same abstract
df = df.drop_duplicates(subset='AB', keep='first')
#drop na for title and abstract
df = df.dropna(subset=['AB','TI'])  #drop na for title and abstract
#drop based on Not available in a title column
df = df[df['TI']!='[Not Available].'] 

#save dataframe to parquet 
# df.to_parquet('pubmed.parquet')

# Convert DataFrame to Ray Dataset
ds = ray.data.from_pandas(df)

print(df[['TI','PMID','AB','AU','AD','LA','MH','PHST','DP','TI','OT']].describe(include='all'), len(set(articles_dict['PMID'])))
# Step 3: Print the number of documents in the Ray Dataset
# Print the number of records in the dataset
print(f"Number of records in the dataset: {ds.count()}")

# print("Current article:", article)

def extract_sections(item):
    sections = []

    # 'AB' is the key containing the abstract text in dictionary
    #item is a dictionary with all keys
    title = item['TI']
    abstract = item['AB']
    pmid = item['PMID']

    #metadata keys 
    metadata_keys = ['AU','AD','LA','MH','PHST','DP','TI','OT']

    # Save the abstract as a section
    if abstract:
        section = {
            'source': "pubmed"+str(pmid),  # Using pmid as the anchor id for the abstract
            'text': "title: " + title + "abstract: " + abstract,
            'metadata': {K: item[K] for K in metadata_keys if item[K]} #add selected columns for metadata, can be used for advanced search at documents retrieval stage
        }
        sections.append(section)

    return sections


# # Step 3: Extract sections from text files in parallel
sections_ds = ds.flat_map(extract_sections, num_cpus=6)

# sections = sections_ds.take_all()
# Print the number of extracted sections
# print(f"Number of extracted sections: {len(sections)}")

# Function to chunk a section
def chunk_section(section, chunk_size, chunk_overlap):
    # Extract sections
    # print(type(section), section)

    # print("section:", section["source"])

    # Chunk each section
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)

    
    chunks = text_splitter.create_documents(
        texts=[section["text"]], 
        metadatas=[{"source": section["source"], "metadata": section["metadata"]}])

    return [{"text": chunk.page_content, "source": chunk.metadata["source"], "metadata": chunk.metadata["metadata"]} for chunk in chunks]


# Apply chunking to the entire dataset
chunk_size = 500
chunk_overlap = 50
chunks_ds = sections_ds.flat_map(partial(
    chunk_section, 
    chunk_size=chunk_size, 
    chunk_overlap=chunk_overlap))

""" # Chunk a sample section
chunk_size = 300
chunk_overlap = 50
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len,
)
sample_section = sections_ds.take(1)[0]
chunks = text_splitter.create_documents(
    texts=[sample_section["text"]], 
    metadatas=[{"source": sample_section["source"]}])
print (chunks[0]) """

# Display the count of chunks and a sample chunk
print(f"{chunks_ds.count()} chunks")
chunks_ds.show(1)

def get_embedding_model(embedding_model_name, model_kwargs, encode_kwargs):
    #we use hugingface model
    embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,  # also works with model_path
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs)
    return embedding_model

# Define the EmbedChunks class
class EmbedChunks:
    def __init__(self, model_name):
        self.embedding_model = get_embedding_model(
            embedding_model_name=model_name,
            model_kwargs={"device": "cuda"},
            encode_kwargs={"device": "cuda", "batch_size": 100})
    def __call__(self, batch):
        embeddings = self.embedding_model.embed_documents(batch["text"])
        return {"text": batch["text"], "source": batch["source"], "metadata": batch["metadata"], "embeddings": embeddings}


# Embed chunks
embedding_model_name = "BAAI/bge-large-en-v1.5"
embedded_chunks = chunks_ds.map_batches(
    EmbedChunks,
    fn_constructor_kwargs={"model_name": embedding_model_name},
    batch_size=100, 
    num_gpus=1,
    compute=ActorPoolStrategy(size=1))

# Display the count of embedded chunks and a sample embedded chunk
print(f"{embedded_chunks.count()} embedded chunks")
# embedded_chunks.show(1)

# Sample
sample = embedded_chunks.take(1)
print ("embedding size:", len(sample[0]["embeddings"]))
print (sample[0]["text"])

# #setting up vector DB
# os.environ["MIGRATION_FP"] = f"vector-pubmed_bge_large.sql"
# os.environ["SQL_DUMP_FP"] = f"{EFS_DIR}/sql_dumps/{embedding_model_name.split('/')[-1]}_{chunk_size}_{chunk_overlap}.sql"
# #write DB connection string to env variable
# # os.environ["DB_CONNECTION_STRING"] = "postgresql://postgres:postgres@localhost:5432/postgres"

# class StoreResults:
#     def __call__(self, batch):
#         # with psycopg2.connect("dbname='pubmedtest' user='dbuser' host='localhost' password='dbpass'") as conn:
#         #     register_vector(conn)
#         #     with conn.cursor() as cur:
#         #         for text, source, embedding in zip(batch["text"], batch["source"], batch["embeddings"]):
#         #             cur.execute("INSERT INTO document (text, source, embedding) VALUES (%s, %s, %s)", (text, source, embedding,))
#         # return {}
#         #use FAISS for vector storage
#         vector_store = FAISS("postgresql://postgres:password@localhost:5432/nlp_embed")
#         vector_store.store_vectors(batch["source"], batch["embeddings"])

# # Index data
# embedded_chunks.map_batches(
#     StoreResults,
#     batch_size=128,
#     num_cpus=1,
#     compute=ActorPoolStrategy(size=2),
# ).count()

#extract text and embeddings from embedded_chunks and save it using indexed FAISS vectorstore
text_and_embeddings = []
metadatas = []

for output in embedded_chunks.iter_rows():
    text_and_embeddings.append(tuple([output['text'],output['embeddings']]))
    metadatas.append(output['metadata'])

print("Creating FAISS Vector Index.")
vector_store = FAISS.from_embeddings(
    text_and_embeddings,
    metadatas=metadatas,
    # Provide the embedding model to embed the query.
    # The documents are already embedded.
    embedding=HuggingFaceEmbeddings(model_name=embedding_model_name),
)



print("Saving FAISS index locally.")
# Persist the vector store.
vector_store.save_local("faiss_index")

# Shutdown Ray
ray.shutdown()
