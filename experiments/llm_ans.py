import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig

torch.set_default_device("cuda")

USE_4bit = True


base_model_id = "microsoft/phi-2"
# base_model_id = "mobiuslabsgmbh/aanaphi2-v0.1"  #"Qwen/Qwen1.5-1.8B" #TODO: add all of these models to documentation card with pros/cons
#TODO: select a config file for all hyperparameters and model names
#TODO: Check for fine-tuning of phi2 if feasible within timeframe https://huggingface.co/blog/g-ronimo/phinetuning

if USE_4bit:
    compute_dtype = getattr(torch, "float16")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
            base_model_id, trust_remote_code=True, quantization_config=bnb_config, device_map={"": 0}, torch_dtype="auto"
    )
else:
    model = AutoModelForCausalLM.from_pretrained(base_model_id, trust_remote_code=True, torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)

# #prompt
# question = "procedure to operate on a patient with a broken leg"
# prompt = f"Instruct: {question}."


#Set Prompt format
instruction_template = "### Human: "
response_template    = "### Assistant: "
def prompt_format(prompt):
    out = instruction_template + prompt + '\n' + response_template
    return out
model.eval();


def generate(prompt, max_length=512):
    prompt_chat = prompt_format(prompt)
    inputs      = tokenizer(prompt_chat, return_tensors="pt", return_attention_mask=True).to('cuda')
    outputs     = model.generate(**inputs, max_length=max_length, eos_token_id= tokenizer.eos_token_id) 
    text        = tokenizer.batch_decode(outputs[:,:-1])[0]
    return text

#Generate
print(generate('If A+B=C and B=C, what would be the value of A?'))
