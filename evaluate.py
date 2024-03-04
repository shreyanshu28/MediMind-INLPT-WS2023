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

# @title
import matplotlib.pyplot as plt


def create_eval_dataset(rag_pipeline, eval_dataset):
  rag_dataset = []
  for row in tqdm(eval_dataset):
    answer = rag_pipeline.invoke({"question" : row["question"]})
    rag_dataset.append(
        {"question" : row["question"],
         "answer" : answer["response"],
         "contexts" : [context.page_content for context in answer["context"]],
         "ground_truths" : [row["ground_truth"]]
         }
    )
  return pd.DataFrame(rag_dataset)
  

def evaluate_rag_model(rag_dataset):
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


#read evaldataset
eval_dataset = pd.read_csv("qa_generated.csv")

basic_qa_ragas_dataset = create_eval_dataset(retrieval_augmented_qa_chain, eval_dataset)

basic_qa_result = evaluate_rag_model(basic_qa_ragas_dataset)


def plot_metrics_with_values(metrics_dict, title='RAG Metrics'):
    """
    Plots a bar chart for metrics contained in a dictionary and annotates the values on the bars.

    Args:
    metrics_dict (dict): A dictionary with metric names as keys and values as metric scores.
    title (str): The title of the plot.
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



