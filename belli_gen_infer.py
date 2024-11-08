import torch
from peft import (
    AutoPeftModelForCausalLM,
    LoraConfig,
    TaskType,
    get_peft_config,
    get_peft_model,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    Trainer,
    TrainingArguments,
)

import datasets
from datasets import Dataset, DatasetDict, load_dataset

model_path = "/mnt/media2/shared/models/farmaci/belli/5/checkpoint-500"


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
quantization_config = None
model = AutoPeftModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

model.eval()

istruzione = "Fornisci una risposta dettagliata spiegando a cosa serve il farmaco."
domanda = "A cosa serve il farmaco Abba?"

istruzione = "Descrivi brevemente il principio attivo del farmaco e la sua funzione principale."
domanda = "Qual è il principio attivo del farmaco Abba?"

istruzione = "Descrivi brevemente il principio attivo del farmaco e la sua funzione principale."
domanda = "Qual è il principio attivo del farmaco Abba?"



istruzione = "Sei un esperto di farmacologia e fornisci informazioni dettagliate sui farmaci come se stessi leggendo il foglietto illustrativo (bugiardino) ufficiale. Mantieni sempre un tono formale e rigoroso, senza fornire consigli medici personali. Le informazioni che fornisci devono essere neutre e oggettive."

domanda = "Quali sono le principali controindicazioni del farmaco Abba?"
domanda = "Come si deve somministrare il farmaco Abba?"
domanda= "Qual è il principio attivo del farmaco Dysport?"
domanda = "Qual è il principio attivo del farmaco Abba?"

# inputs = tokenizer.apply_chat_template(
#     conversation=[
#         {"role": "system", "content": istruzione},
#         {"role": "user", "content": domanda},
#     ],
#     tokenize=True,
#     truncation=True,
#     max_length=1024,
#     return_tensors="pt",
#     # return_overflowing_tokens=True,
#     # return_length=True,
# )
# inputs = {'input_ids': inputs}


text = tokenizer.apply_chat_template(
    conversation=[
        {"role": "system", "content": istruzione},
        {"role": "user", "content": domanda},
    ],
    tokenize=False,
    truncation=True,
    max_length=1024,
    add_generation_prompt=False,
    # return_overflowing_tokens=True,
    # return_length=True,
)
inputs = tokenizer(text, return_tensors="pt")





# generation_config = GenerationConfig(
#     repetition_penalty=1.0,
#     temperature=0.1,
#     num_beams=2,
#     do_sample=True,
# )
generation_config = None

outputs = model.generate(
    input_ids=inputs["input_ids"].to("cuda"),
    max_new_tokens=256,
    generation_config=generation_config,
)
print(tokenizer.batch_decode(outputs, skip_special_tokens=False)[0])





