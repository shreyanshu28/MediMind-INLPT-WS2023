# MediMind-INLPT-WS2023

# Title: Question Answering RAG System for PubMed Data


## Team Members

* @ozgeberktas: Insaf Ozge Berktas (<insaf.berktas@stud.uni-heidelberg.de>)
* @jeromepatel: Jyot Makadiya (<jyot.makadiya@stud.uni-heidelberg.de>)
* @a-sameh1: Ahmed Abdelraouf (<ahmed.abdelraouf@stud.uni-heidelberg.de>)


## Steps to run the project

1. Clone the repository
2. Python version 3.10, CUDA version 12.1, Windows 11 (MacOS also tested)
3. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

3. Download the dataset files manually from [HeiBox](<https://heibox.uni-heidelberg.de/d/692badba5bfa44f889c6/>) and place all 4 files inside `data/PubMed_Format` (Recommended and Easier method) or run the script `PubMed_raw_data.ipynb`.

4. [Optional] To run the file `PubMed_raw_data.ipynb`, it requires API_KEYS and EMAIL for PubMed. For that, go to the following [website](<https://account.ncbi.nlm.nih.gov/settings/>) and create your API_KEYS. Now place EMAIL and API_KEYS inside `config/.env`.

5. [Optional] Not recommended, however if you want to generate FAISS Vectorbase using GPT4ALL embeddings model yourself, run through cells until **Create a Vector Store** section. You will a vectorStore folder created with two files. Alternatively, follow next step;
   
6. download the Vector store that includes the FAISS vectorbase from [HeiBox](<https://heibox.uni-heidelberg.de/d/3f7644ce7dba4db4bfc2//>) (put inside the directory `vectorStore` folder both the files). 

<!--6. API_KEYS are created from the following [website](<https://account.ncbi.nlm.nih.gov/settings/>)-->

7. Run the following steps to run Ollama:

    download ollama app from the [ollama website](https://ollama.ai/download) Depending upon your OS.
    Run app and use `ollama pull llama2` (for Windows: go to directory of downloaded Ollama app and execute if encountered error `ollama command not found`) to download the model
    run `ollama run llama2` to start the model

8.  Run `python build_pipeline.py` to have a conversation chat with the bot. (It will prompt to give query till "exit")
    
9.  A) [Optional] To generate the evaluation dataset:
 download and place **mistral-7b-Instruct-v0.1** and place inside `.cache/hugginface` local directory (alternatively run the following command to download the model: `huggingface-cli download mistral-7b-Instruct-v0.1` </br>
 Now run the command to generate the evaluation dataset: `python evaluate/generate_evaluation_dataset.py`

10.  To fianlly run evalution with given QA dataset:
 finally run using `python evaluate.py`

**Note:**
* The code was tested on MacOS and Windows (The linux version gives error with GPT4All library, however if you can overcome the loading of GPT4AllEmbeddings(), it should be find.)
* System without sudo command cannot run Ollama build at the moment, (for example a university remote server) (Revert to simpler model commented in the `build_pipeline.py`)
* It will atleast take 30 minutes even for a powerful computer to install `requirements.txt` so please be patient, unfortunately we had lot of dependencies due to external libraries such as `ollama` and `GPT4All`




Details about weekly meeting and Progress of the project:
[Google Docs](https://docs.google.com/document/d/1s9WYkriT6fogZpYWcFGu1aMR-8EMgOY1pBHmL2Up5j8/edit?usp=sharing)
