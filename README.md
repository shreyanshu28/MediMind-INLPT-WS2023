# MediMind-INLPT-WS2023

QA system for pub med abstracts collection dataset

## Steps to run the project

1. Clone the repository
2. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Project Prerequisites

1. [Optional] Download the dataset files manually from [HeiBox](<https://heibox.uni-heidelberg.de/d/692badba5bfa44f889c6/>) or run the script `PubMed_raw_data.ipynb`.

2. Before running the file `PubMed_raw_data.ipynb` go to the follwoing website and create your API_KEYS and store EMAIL and API_KEYS inside `config/.env`.

3. download the Vector store that includes the FAISS Index from [HeiBox](<https://heibox.uni-heidelberg.de/d/3f7644ce7dba4db4bfc2//>)

4. API_KEYS are created from the following [website](<https://account.ncbi.nlm.nih.gov/settings/>)

5. Run the from `main.py` file and call the need function and pass the parammters, comments are add for illustration in each file :

6. Run the following steps to run Ollama:

7. download ollama app from the [ollama website](https://ollama.ai/download)
run app and use `ollama pull llama2` to download the model
run `ollama run llama2` to start the model

8. [Optional] To generate the evaluation dataset:

* download and place **mistral-7b-Instruct-v0.1** and place inside `.cache/hugginface` local directory (alternatively run the following command to download the model: `huggingface-cli download mistral-7b-Instruct-v0.1`

* run the command to generate the evaluation dataset: `python generate_evaluation_dataset.py`

* finally run using `python evaluate.py`

[Optional] To evaluate the model, run the following command:

```bash


Team Members:
* @shreyanshu28: Shreyanshu Vyas
* @ozgeberktas: Insaf Ozge Berktas
* @jeromepatel: Jyot Makadiya
* @a-sameh1: Ahmed Abdelraouf 

Details about weekly meeting and Progress of the project:
[Google Docs](https://docs.google.com/document/d/1s9WYkriT6fogZpYWcFGu1aMR-8EMgOY1pBHmL2Up5j8/edit?usp=sharing)
