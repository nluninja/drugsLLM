import os

from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

# Set HF_TOKEN environment variable before running
hf_token = os.environ.get("HF_TOKEN", "")


file_name = "dataset_farmaci_qaNEW 1.json"
dataset_all = load_dataset("json", data_files=file_name)
# Dataset({
#     features: ['istruzione', 'domanda', 'risposta'],
#     num_rows: 1705
# })


# calcola i caratteri totali per riga stampa i risultati ordinati per massimo

len_size = [
    len(row["istruzione"]) + len(row["domanda"]) + len(row["risposta"])
    for row in dataset_all["train"]
]
sorted(len_size)
max(len_size)


model_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
tokenizer_name_or_path = "meta-llama/Llama-3.1-8B-Instruct"
target_modules = ["o_proj", "qkv_proj"]
target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=32,
    lora_alpha=32,
    lora_dropout=0.05,
    # lora_alpha=32,
    # lora_dropout=0.1,
    target_modules=target_modules,
)
quantization_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
context_length = 512

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    quantization_config=quantization_config,
    device_map="auto",
    token=hf_token,
    # variant="fp16",
    # attn_implementation="flash_attention_2",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, token=hf_token)
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

tokenizer.chat_template


tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Sei un pirata"},
        {"role": "user", "content": "Chi sei?"},
        {"role": "assistant", "content": "Sono un pirata"},
    ],
    tokenize=False,
)

tokenizer.apply_chat_template(
    [
        {"role": "system", "content": "Sei un pirata"},
        {"role": "user", "content": "Chi sei?"},
        {"role": "assistant", "content": "Sono un pirata"},
    ],
)


# old version no instruction style
# def tokenize(element):
#     outputs = tokenizer(
#         element["text"],
#         truncation=True,
#         max_length=context_length,
#         return_overflowing_tokens=True,
#         return_length=True,
#     )
#     input_batch = []
#     for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
#         if length == context_length:
#             input_batch.append(input_ids)
#     return {"input_ids": input_batch}

# ['istruzione', 'domanda', 'risposta'],


def tokenize(element):
    return tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": element["istruzione"]},
            {"role": "user", "content": element["domanda"]},
            {"role": "assistant", "content": element["risposta"]},
        ],
        tokenize=True,
        truncation=True,
        max_length=context_length,
        # return_overflowing_tokens=True,
        # return_length=True,
    )


# def tokenize(batch):
#     tokenized_batch = []
#     for istruzione, domanda, risposta in zip(batch["istruzione"], batch["domanda"], batch["risposta"]):
#         tokenized_element = tokenizer.apply_chat_template(
#             conversation=[
#                 {"role": "system", "content": istruzione},
#                 {"role": "user", "content": domanda},
#                 {"role": "assistant", "content": risposta},
#             ],
#             tokenize=True,  # Ensure the input is tokenized
#             truncation=True,  # Truncate to max length
#             max_length=context_length,  # Set max token length
#         )

#         tokenized_batch.append(tokenized_element)

#     return tokenized_batch


def tokenize(element):
    # print(f"{element['istruzione'][:10]} {element['domanda'][:10]} {element['risposta'][:10]}")
    text = tokenizer.apply_chat_template(
        conversation=[
            {"role": "system", "content": element["istruzione"]},
            {"role": "user", "content": element["domanda"]},
            {"role": "assistant", "content": element["risposta"]},
        ],
        add_generation_prompt=False,
        tokenize=False,
        truncation=True,
        max_length=context_length,
        # return_overflowing_tokens=True,
        # return_length=True,
    )
    return tokenizer(
        text,
        truncation=True,
        max_length=context_length,
        # return_overflowing_tokens=True,
        # return_length=True,
    )


dataset_splitted = dataset_all["train"].train_test_split(test_size=0.02)


dataset_tokenized = dataset_splitted.map(
    tokenize,
    batched=False,
    remove_columns=dataset_splitted["train"].column_names,
)

print(dataset_tokenized)

# controllino che non guasta mai:
dataset_tokenized["train"][0]["input_ids"]
tokenizer.decode(dataset_tokenized["train"][-1]["input_ids"])


args = TrainingArguments(
    output_dir="output/belli/2",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="steps",
    eval_steps=250,
    logging_steps=10,
    gradient_accumulation_steps=1,
    max_steps=5000,
    weight_decay=0.1,
    warmup_steps=200,
    # lr_scheduler_type="cosine",
    learning_rate=2e-4,
    save_steps=500,
    fp16=True,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=dataset_tokenized["train"],
    eval_dataset=dataset_tokenized["test"],
)

trainer.train()
