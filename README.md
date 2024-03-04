# MediMind-INLPT-WS2023
QA system for pub med abstracts collection dataset

## Steps to run the project:
1. Clone the repository
2. Install the required packages using the following command:
```bash
pip install -r requirements.txt
```
3. [Optional] Download the dataset files
4. [Alternative to 3] Run the following command to download the dataset files:
```bash
python download_dataset.py
```
5. Run the following command to preprocess dataset and create the embeddings vector store:
```bash
python preprocess.py
```
6. Run the following steps to run Ollama:
```
download ollama app from the [ollama website](https://ollama.ai/download)
run app and use 'ollama pull llama2' to download the model
run 'ollama run llama2' to start the model
```
7. Run the following file to interact with the model:
```bash
python run_llm.py
```
8. [Optional] To generate the evaluation dataset:
```
* download and place mistral-7b-Instruct-v0.1 and place inside .cache/hugginface local directory (alternatively run the following command to download the model: 'huggingface-cli download mistral-7b-Instruct-v0.1')

* run the command to generate the evaluation dataset: `python generate_evaluation_dataset.py`
* finally run using `python evaluate.py`
```
9. [Optional] To evaluate the model, run the following command:
```bash


Team Members:
* @shreyanshu28: Shreyanshu Vyas
* @ozgeberktas: Insaf Ozge Berktas
* @jeromepatel: Jyot Makadiya
* @a-sameh1: Ahmed Abdelraouf 

Details about weekly meeting and Progress of the project:
[Google Docs](https://docs.google.com/document/d/1s9WYkriT6fogZpYWcFGu1aMR-8EMgOY1pBHmL2Up5j8/edit?usp=sharing)


