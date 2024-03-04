from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate
from tqdm import tqdm
import pandas as pd
from build_pipeline import get_retrieval_chain, get_relenvant_documents, get_document_chain
from langchain_core.messages import HumanMessage, AIMessage
from datasets import Dataset

# @title
import matplotlib.pyplot as plt


def create_eval_dataset(rag_pipeline, eval_dataset):
  '''
  Create a dataset for evaluation of the RAG model
  Call the RAG model to get the answers for the questions in the eval_dataset
  Use the context from the retrieved documents to get the answers
  Combine the answers and the context to create the dataset
  '''
  rag_dataset = []
  retrieval_chain, retriever = rag_pipeline
  chat_history = [HumanMessage(content=""), AIMessage(content="Answered!")]
  for i,row in tqdm(eval_dataset.iterrows()):
    documents = get_relenvant_documents(retriever, row["context"])
    context = documents[0]
    chat_history = [HumanMessage(content="Question:"), AIMessage(content="Answered!")]
    answer =  retrieval_chain.invoke({
            "chat_history": chat_history,
            "input": row["question"],
            "context": context,
        })['answer']
    
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["answer"],
         "contexts" : context,
         "ground_truths" : [row["ground_truth"]]
         }
    )
    if i > 10:
      #sample only 10 answers (for faster evaluation)
      break
  df =  pd.DataFrame(rag_dataset)
  eval_dataset = Dataset.from_pandas(df)
  return eval_dataset
  

def evaluate_rag_model(rag_dataset):
  '''
  Evaluate the RAG model using the metrics
  
  '''
  result = evaluate(
    rag_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result




def plot_metrics_with_values(metrics_dict, title='RAG Metrics'):
    """
    Plots a bar chart for metrics contained in a dictionary and annotates the values on the bars.
    """
    names = list(metrics_dict.keys())
    values = list(metrics_dict.values())

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, values, color='skyblue')

    # Adding the values on top of the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01,  # x-position
                 bar.get_y() + bar.get_height() / 2,  # y-position
                 f'{width:.4f}',  # value
                 va='center')

    plt.xlabel('Score')
    plt.title(title)
    plt.xlim(0, 1)  # Setting the x-axis limit to be from 0 to 1
    plt.show()




#read evaldataset
eval_dataset = pd.read_csv("evaluate/qa_generated.csv")

#get rag chain

rag_chain = get_retrieval_chain()


basic_qa_ragas_dataset = create_eval_dataset(rag_chain, eval_dataset)

basic_qa_result = evaluate_rag_model(basic_qa_ragas_dataset)
