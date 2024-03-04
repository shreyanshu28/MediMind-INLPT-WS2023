# MediMind-INLPT-WS2023

# Title: Question Answering RAG System for PubMed Data


## Team Members

* @ozgeberktas: Insaf Ozge Berktas (<insaf.berktas@stud.uni-heidelberg.de>)
* @jeromepatel: Jyot Makadiya (<jyot.makadiya@stud.uni-heidelberg.de>)
* @a-sameh1: Ahmed Abdelraouf (<ahmed.abdelraouf@stud.uni-heidelberg.de>)


## Steps to run the project

1. Clone the repository
2. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

3. [Optional] Download the dataset files manually from [HeiBox](<https://heibox.uni-heidelberg.de/d/692badba5bfa44f889c6/>) or run the script `PubMed_raw_data.ipynb`.

4. Before running the file `PubMed_raw_data.ipynb` go to the follwoing website and create your API_KEYS and store EMAIL and API_KEYS inside `config/.env`.

5. download the Vector store that includes the FAISS vectorbase from [HeiBox](<https://heibox.uni-heidelberg.de/d/3f7644ce7dba4db4bfc2//>) (put inside the directory `vectorStore` folder both the files) 

6. API_KEYS are created from the following [website](<https://account.ncbi.nlm.nih.gov/settings/>)

7. Run the following steps to run Ollama:

    download ollama app from the [ollama website](https://ollama.ai/download)
    run app and use `ollama pull llama2` (go to directory of downloaded Ollama app and execute when encountered error for ollama) to download the model
    run `ollama run llama2` to start the model

8.  Run `python build_pipeline.py` to have a conversation chat with the bot. (It will prompt to give query till "exit")
    
9.  A) [Optional] To generate the evaluation dataset:
 download and place **mistral-7b-Instruct-v0.1** and place inside `.cache/hugginface` local directory (alternatively run the following command to download the model: `huggingface-cli download mistral-7b-Instruct-v0.1` </br>
 Now run the command to generate the evaluation dataset: `python evaluate/generate_evaluation_dataset.py`

10.  To fianlly run evalution with given QA dataset:
 finally run using `python evaluate.py`




Details about weekly meeting and Progress of the project:
[Google Docs](https://docs.google.com/document/d/1s9WYkriT6fogZpYWcFGu1aMR-8EMgOY1pBHmL2Up5j8/edit?usp=sharing)
