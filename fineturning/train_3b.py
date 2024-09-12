import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
os.environ["NCCL_P2P_DISABLE"] ="1"
os.environ["NCCL_IB_DISABLE"] ="1"
os.environ['WANDB_PROJECT'] = 'numfa-mark4'

import wandb

wandb.login()
wandb.init(project= 'numfa-mark5')

# NCCL_P2P_DISABLE="1"` and `NCCL_IB_DISABLE="1"


from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser, TrainingArguments, pipeline, logging, TextStreamer
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import os,torch, platform, warnings
from datasets import load_dataset
from trl import SFTTrainer
import torch


import glob


def print_system_specs():
    # Check if CUDA is available
    is_cuda_available = torch.cuda.is_available()
    print("CUDA Available:", is_cuda_available)
# Get the number of available CUDA devices
    num_cuda_devices = torch.cuda.device_count()
    print("Number of CUDA devices:", num_cuda_devices)
    if is_cuda_available:
        for i in range(num_cuda_devices):
            # Get CUDA device properties
            device = torch.device('cuda', i)
            print(f"--- CUDA Device {i} ---")
            print("Name:", torch.cuda.get_device_name(i))
            print("Compute Capability:", torch.cuda.get_device_capability(i))
            print("Total Memory:", torch.cuda.get_device_properties(i).total_memory, "bytes")
    # Get CPU information
    print("--- CPU Information ---")
    print("Processor:", platform.processor())
    print("System:", platform.system(), platform.release())
    print("Python Version:", platform.python_version())
print_system_specs()


# Pre trained model
model_name = "pythainlp/KhanomTanLLM-3B" 

# # Dataset name
# dataset_name = "vicgalle/alpaca-gpt4"

# Hugging face repository link to save fine-tuned model(Create new repository in huggingface,copy and paste here)
new_model = "mark5_3b_sft"


from datasets import Dataset,DatasetDict
import pandas as pd
#df = pd.concat([pd.read_csv(i) for i in dataset_train_path])
ds = load_dataset("wannaphong/KhanomTanLLM-Instruct-dataset",split="train")
dataset=ds

max_seq_length=2048

# Load base model(llama-2-7b-hf) and tokenizer
bnb_config = BitsAndBytesConfig(
    load_in_4bit= True,
    bnb_4bit_quant_type= "nf4",
    bnb_4bit_compute_dtype= torch.bfloat16,
    bnb_4bit_use_double_quant= False,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map={'':torch.cuda.current_device()},
    #device_map="auto",
    torch_dtype=torch.bfloat16,
    #load_in_4bit=True,
    #rope_scaling={"type":"dynamic","factor":2.0},
    #max_seq_length = max_seq_length,
)
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False # silence the warnings. Please re-enable for inference!
model.config.pretraining_tp = 1
# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.add_eos_token = True
tokenizer.padding_side = "right"
# tokenizer.add_bos_token, tokenizer.add_eos_token


tokenizer.chat_template="{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}"

print(tokenizer.chat_template)

tokenizer.eos_token

import datetime
#dataset.filter(lambda example: isinstance(eval(example['relevant_laws']),list))

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    inputs       = examples["messages"]
    # list_law= [i for i in examples["relevant_laws"]]
    # list_txt_law=[]
    # for i in list_law:
    #     list_txt_law.append(law_data[i["law_code"][str(i['sections'][0])]])
    texts = []
    for message in inputs:
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = tokenizer.apply_chat_template(message, tokenize=False) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

from datasets import load_dataset
dataset = dataset.map(formatting_prompts_func, batched = True,).filter(lambda example: isinstance(example['text'],str))


print(dataset["text"][2111])


# peft_config = LoraConfig(
#     lora_alpha= 256, # cr. https://www.reddit.com/r/LocalLLaMA/comments/17pw7bv/comment/k88zmab/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button
#     lora_dropout= 0.1,
#     r= 64,
#     bias="none",
#     task_type="CAUSAL_LM",
#     target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]#["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
# )
peft_config = LoraConfig(
    lora_alpha= 256,
    lora_dropout= 0.1,
    r= 64,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]#["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj"]
)

training_arguments = TrainingArguments(
    output_dir= new_model,
    num_train_epochs=3,
    per_device_train_batch_size= 8,
    gradient_accumulation_steps= 1,
    optim = "paged_adamw_8bit",
    save_steps= 1000,
    logging_steps= 10,
    learning_rate= 2e-4,
    #learning_rate= 5e-5,#2e-4,
    #warmup_steps=50,
    weight_decay= 0.001,
    fp16= False,
    bf16= True,
    max_grad_norm= 0.3,
    max_steps= -1,
    warmup_ratio = 0.03,
    group_by_length= True,
    lr_scheduler_type= "cosine",
    report_to="wandb",
    save_total_limit=2,
)

# Setting sft parameters
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length= max_seq_length,
    dataset_text_field="text",
    tokenizer=tokenizer,
    args=training_arguments,
    packing= True,
    neftune_noise_alpha=5
)

# Train model
trainer.train()