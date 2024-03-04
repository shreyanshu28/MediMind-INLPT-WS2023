
import pandas as pd

from tqdm import tqdm
import random
import time

from langchain.prompts import ChatPromptTemplate

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain_community.llms import HuggingFaceHub
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


device = "cuda" # the device to load the model onto


#define model and load the tokenizer

def set_up_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer


model_name = "mistralai/Mistral-7B-Instruct-v0.2"
model, tokenizer = set_up_model(model_name)




random.seed(42)


#start = time.time()
def read_docs():
    df = pd.read_parquet("pubmed.parquet")
    docs = df["AB"].tolist()
    return docs

docs = read_docs()



start = time.time()

def create_qa_pipeline(model, tokenizer, device):
  question_answerer = pipeline(
      "text-generation", 
      model=model, 
      device=device,
      tokenizer=tokenizer,
      max_length = 2048,
      return_tensors='pt',
      eos_token_id=tokenizer.eos_token_id,
      pad_token_id=tokenizer.eos_token_id
  )

  qa_generation_llm = HuggingFacePipeline(
      pipeline=question_answerer,
      model_kwargs={'temperature':0.8},
  )
  return qa_generation_llm

qa_generation_llm = create_qa_pipeline(model, tokenizer, device)


q_template = """\
Generate a question based only on the following context:
{context}
You are allowed to rephrase the question based on the context. 
Question:
"""

prompt_template = ChatPromptTemplate.from_template(template=q_template)

question_generation_chain = prompt_template | qa_generation_llm
response = question_generation_chain.invoke({"context" : docs[0]})

end = time.time()
execution_time = end-start
print(f"Execution time of generating one sample: {execution_time:.6f} seconds")

print(response)
#only use if using OpenAI API: 
#output_dict = question_output_parser.parse(response.content)

def generate_questions(docs, question_generation_chain):
    qac_triples = []
    for text in tqdm(random.sample(docs, 100)):
        response = question_generation_chain.invoke({"context" : text})
        output_dict = {"context": text, "question":response}
        qac_triples.append(output_dict)
    return qac_triples


qac_triples = generate_questions(docs, question_generation_chain)

print("questions generated..")
print(qac_triples[:2])


a_template = """\
Answer the question based only on the following context:
{context}
You are allowed to rephrase the answer based on the context. 
Question: {question},
Give the answer after this sentence. Answer:
"""

prompt_template = ChatPromptTemplate.from_template(template=a_template)

answer_generation_chain = prompt_template | qa_generation_llm

response = answer_generation_chain.invoke({"context" : qac_triples[0]["context"] ,"question": qac_triples[0]['question'] })

print(response)

print("generationg answers now....")

full_pairs = []
for triple in tqdm(qac_triples):
  response = answer_generation_chain.invoke({"context" : triple["context"] ,"question": triple['question'] })
  full_pairs.append(triple)
  full_pairs[-1]["answer"] = response

  
print(full_pairs[:2])


ground_truth_qac_set = pd.DataFrame(full_pairs)

#ground_truth_qac_set["context"] = ground_truth_qac_set["context"].map(lambda x: str(x.page_content))

ground_truth_qac_set = ground_truth_qac_set.rename(columns={"answer" : "ground_truth"})

ground_truth_qac_set.to_csv('qa_generated_dataset.csv',index=False)



