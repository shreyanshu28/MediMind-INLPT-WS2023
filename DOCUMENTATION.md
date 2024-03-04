# Title: Question Answering RAG System for PubMed Data
# Team Members:
* @ozgeberktas: Insaf Ozge Berktas (insaf.berktas@stud.uni-heidelberg.de)
* @jeromepatel: Jyot Makadiya (jyot.makadiya@stud.uni-heidelberg.de)
* @a-sameh1: Ahmed Abdelraouf (ahmed.abdelraouf@stud.uni-heidelberg.de)
# Member Contribution:
* Insaf Ozge Berktas: Preprocessing, Split & Chunk, Embeddings, Create a vectore store, alternative chain
* Jyot Makadiya: Preprocessing, Embeddings, alternative experimental vectordatabase, Conversational chain, Evaluation
* Ahmed: Data Acquisition, Split & Chunk, Embeddings, Create a vectore store, advanced chain and retrieval system

# Advisor:
* Satya Almasian 

# Anti-Plagiarism Statement:
This project is entirely our own work, except where I have clearly acknowledged the contribution of others. It has not been submitted for any other degree or diploma at any institution. We have cited all sources used in its creation, adhering to Heidelberg University's academic integrity and anti-plagiarism policies. By submitting this work for the Project for Natural Language processing course, We consent to its verification for originality and understand the consequences of any found violation of plagiarism standards.

# Overview

In this project, we leverage PubMed data to develop a Question Answering (QA) system that delivers precise and concise answers based on PubMed articles published from 2013 to 2024. Utilizing the **RAG Architecture**, we implement various NLP techniques such as embeddings creation, storage, loading, chunking, indexing, and information retrieval, along with a conversation retrieval chain. These methods enable our model to comprehend, interpret, and generate new answers from retrieved questions, capturing the semantic essence of the text data effectively.

## Approach


### Data Acquisition

Acquiring data from a web source poses significant challenges. Our `search_and_download_abstracts` function automates the process of searching for and downloading abstracts from the PubMed database using the `Entrez` API. It performs search queries based on specified parameters and downloads abstracts within a given date range. This function is essential for efficiently gathering large volumes of scientific journals, considering the 10,000 record per request limit on the website. To enhance security and maintainability, API keys and email addresses are stored using the 'dotenv package' for environment variable management, rather than being hard-coded.

### Preprocessing (Data Cleaning)

The raw data undergoes cleaning to ensure quality and relevance:

* Duplicate PMIDs are dropped.
* Rows missing abstract values are dropped.
* Column data types are converted to strings.
* Select the significat columns in our case PMID, Abstract, Title, place of publication and date of publication etc...
* Save the dataframe in Parquet format to leverage its storage efficiency advantages.

### Additional Techniques (Not Implemented)

These methods were considered but not implemented in our project:

* Lowercasing: Ensures a single representation for each word by converting all text to lowercase.
* Tokenization: Transforms raw text into a format conducive to embedding creation.
* Normalization: Applies stemming or lemmatization to reduce words to their base or root form.
* Alphabetical Filtering: Removes numbers and special characters, retaining only alphabetic characters.

### Downloading the data

* we have uploaded data starting from 2013->2019 becuase it has small volume becuase github has file size limitation. Therefore, we have used HeiBox to store the rest of the data from 2020-> 2024 in HEIBOX in the follwoing linke : <https://heibox.uni-heidelberg.de/d/692badba5bfa44f889c6/>

* To use go to the follwoing website and create your API_KEYS and store EMAIL and API_KEYS inside config/.env you have to create a API_KEYS. <https://account.ncbi.nlm.nih.gov/settings/>

* To download the FAISS_Index files: <https://heibox.uni-heidelberg.de/d/3f7644ce7dba4db4bfc2//>

* data loading, processing, embedding, and storing results in a vector database for efficient retrieval.

1. loading and processing PubMed Data: Reading and processing PubMed data for analysis.
2. Extraction and chunking of text sections: Extracting sections from text files and chunking them for processing.
3. vectore store creation: Setting up a directory for the vector.
4. Embedding text chunks: the process of transforming segments of text into numerical vectors. which captures the semantic meaning of the text, allows machines to interpret and process.
5. Storing the FAISS index Results: Saving the embedded chunks to a vector database for querying.

### Preprocessing

* Change the datatype for each column

* Select the significat columns in our case PMID, Abstract, Title, place of publication and date of publication etc...

* drop rows that has duplicate PMID.

* drop rows that has none values for Abstract column

* Save the dataframe in Parquet format to leverage its storage efficiency advantages.

### Split & Chunk

* In split and chunking process, we use `RecursiveCharacterTextSplitter`. to break down data into smaller segments.
* Each document is divided recursively, ensuring that the resulting segments have an appropriate size for efficient processing and analysis.

## Embeddings

1. **Load Documents**: Use `DataFrameLoader` to load documents from a DataFrame.
2. **Generate Document Embeddings**: Convert documents to embeddings using`GPT4AllEmbeddings`.

## Create a vectore store

* We used the FAISS library to index and store the vector representations(embeddings) of the text documents.

* We saved the vector store locally using the save_local method. To load the vector store from the saved directory, we use the FAISS.load_local method. With the help of that, indexed data is available for similarity search and retrieval.

## Chain

* We initialized an instance of the `Ollama` and use llam2 language model `llm` that helped us generate responses based on the provided context, metadata, and user queries.

* We created a prompt template that specifies how the system should generate responses. Then, We turned vector store into a retriever. This retriever will fetch relevant documents based on user queries `get_relevant_documents(query)`.

* We created a `document_chain` that processes documents. This chain uses the language model (llm) to generate responses based on the provided context, metadata, and user input.

* We created another `retriever_chain` that combines the history of conversations with document retrieval. This chain ensures that retrieved documents are relevant to the ongoing conversation.

* We created a retrieval chain that combines a history-aware retriever and a document chain. The history-aware retriever considers the conversation history, while the document chain processes the retrieved documents. Then, we invoke the retrieval chain with the appropriate inputs to generate responses based on the provided context, metadata, and user query.

### Evaluation

The goal of the evaluation is to assess the performance of the system in retrieving relevant documents and generating accurate responses to user queries. We divided the evaluation in two parts: automated and human evaluation.

#### Automated Evaluation

This evaluation is done using generated question answer pairs using a high quality language model. We used the `Mistral-7b-Instruct` model to generate questions and answers for the documents in the dataset. We then used the generated question answer pairs to evaluate the performance of the system.

* Choice of model: Instruct model is a high quality language model that is trained on a large dataset and is capable of generating high quality questions and answers by following the instructions provided accurately.
* Generating questions: We used the Instruct model to generate questions for the documents in the dataset. We used prompt engineering methods to craft a prompt which gives one question per context (in our case abstract).
* Generating answers: We used the Instruct model to generate answers for the questions generated in the previous step. Here we also used prompt engineering to curate one answer given question and context both.
* Scalabiliy: Since we use LLM to generate questions and answers, the system is scalable and can be used to generate questions and answers for large datasets. For example, leveraging OpenAI API (for GPT 4) or Huggingface API (for Mistral-8x7b) to generate questions and answers high quality for large datasets.
* Our methodology created abstractive question for each context/document, we further used raga to evaluate our pipeline output quality. We used the generated question answer pairs to evaluate the performance of the system.

* Evaluation metrics:
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
Using these metrics, we can evaluate the performance of the system in retrieving relevant documents and generating accurate responses to user queries.

#### Results from automated evaluation

TODO
[add one plot here]

#### Human Evaluation

TODO

#### Results from human evaluation

TODO

## Frontend

* Unfortuantly, we didn't have much time to integrate the Questioning and Answering system with the frontend but we have built the user interface.

1. `cd frontend`

2. `npm install`

3. `npm start`

4. You can now view frontend in the browser. Local: <http://localhost:3000>

![Alt text](pictures/interface_Screenshot.png "Optional title")

## Future work

### User Feedback and Iteration

* Establish mechanisms for collecting user feedback on the frontend interface and functionality. This feedback will be invaluable for identifying areas for improvement and refining the user experience.

* Plan for iterative development cycles focused on implementing user feedback, fixing issues, and introducing new features based on user needs and technological advancements.

### Multilingual Data Processing and Analysis

* Extend the system's capabilities to process and analyze documents in multiple languages addressing the current limitation of English-only support.

* Implement NLP tools and models that are optimized for multilingual processing, such as transformer-based models with multilingual capabilities (e.g., mBERT, XLM-R).

* Develop or integrate translation services to allow users to submit queries in their language and receive translated results, maintaining the semantic integrity of the content.

* Conduct research on language-specific nuances and cultural contexts to ensure accurate interpretation and analysis of foreign language documents.

## Refrences

* LangChain. (n.d.). Langchain: A framework for developing applications powered by language models. <https://python.langchain.com/docs/get_started/introduction>

* LangChain. (n.d.). Retrieving relevant data with LangChain. <https://python.langchain.com/docs/use_cases/chatbots/retrieval>

* LangChain. (n.d.). Supported Language Models. <https://python.langchain.com/docs/modules/model_io/llms/>

* LangChain. (n.d.). Vector Stores. <https://python.langchain.com/docs/integrations/vectorstores>

* LangChain. (n.d.). Retrievers. <https://python.langchain.com/docs/integrations/retrievers>

* LangChain. (n.d.). VectorDB Q&A Chain. <https://js.langchain.com/docs/integrations/vectorstores>

* LangChain. (n.d.). VectorStore Retriever. <https://python.langchain.com/docs/integrations/retrievers>

* Ola LLAMA. (n.d.). OLAMA: Open Large Language Model Archive. <https://github.com/ollama/ollama>

* Langchain. (n.d.). LangGraph RAG example notebook. <https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_self_rag.ipynb?ref=blog.langchain.dev>

* Deci.Ai (n.d.). Deci.Ai: AI for the Enterprise. <https://deci.ai/blog>
* Wolf, T., Debut, L., Sanh, V., Chaumond, J., Delangue, C., Moi, A., ... & Rush, A. M. (2019). HuggingFace's Transformers: State-of-the-art Natural Language Processing. ArXiv, abs/1910.03771.
* Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S., Casas, D. D. L., ... & Sayed, W. E. (2023). Mistral 7B. arXiv preprint arXiv:2310.06825.
* Anyscale. (n.d.). Ray: A distributed execution framework. <https://docs.ray.io/en/master/index.html>
* FAISS. (n.d.). FAISS: A library for efficient similarity search. 
* 