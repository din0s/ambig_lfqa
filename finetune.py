from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainingArguments,
    Trainer,
)
from scoring import _rouge_calculation as rouge

import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_var_or_default(var, default):
    return os.environ[var] if var in os.environ else default

base_model = get_var_or_default("BASE_MODEL", "t5-base")
model_name = get_var_or_default("MODEL_NAME", base_model)
tokenizer_name = get_var_or_default("TOKENIZER_NAME", model_name)

dataset_hf_user = get_var_or_default("DATASET_HF_USER", "din0s")
dataset_name = get_var_or_default("DATASET_NAME", "asqa")
OPEN_BOOK = get_var_or_default("OPEN_BOOK", "false").lower() == "true"

print(f"Finetuning {model_name} on {dataset_name} ({'open' if OPEN_BOOK else 'closed'}-book variant)")

dataset = load_dataset(f"{dataset_hf_user}/{dataset_name}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

hf_base_name = f"{base_model}-pt" if "/" in model_name else model_name
ft_model_name = f"{hf_base_name}-{dataset_name}-{'ob' if OPEN_BOOK else 'cb'}"
train_batch_size = 8 if OPEN_BOOK else 16
eval_batch_size = 8

def get_context(example):
    if dataset_name == "asqa":
        context = [p["context"] for p in example["qa_pairs"] if p["context"] != "No context provided"]
    elif dataset_name == "msmarco-nlgen":
        context = [p["passage_text"] for p in example["passages"] if p["is_selected"]]
    else:
        raise Exception(f"Unknown dataset {dataset_name}")

    return context

def has_context(example):
    return len(get_context(example)) > 0

def tokenize_function(example):
    if dataset_name == "asqa":
        question = example["ambiguous_question"]
        answer = example["annotations"][0]["long_answer"]
    elif dataset_name == "msmarco-nlgen":
        question = example["query"]
        answer = example["answers"][0]
    else:
        raise Exception(f"Unknown dataset {dataset_name}")

    question = f"question: {question}"
    if OPEN_BOOK:
        if base_model == "bart":
            context = "<P> " + " <P> ".join(get_context(example))
        else:
            context = " | ".join(get_context(example))
        question = f"{question} context: {context}"

    question_tokenized = tokenizer(question, truncation=True, max_length=512)
    example['input_ids'] = question_tokenized['input_ids']

    ans_tokenized = tokenizer(text_target=answer, truncation=True, max_length=512)
    example["labels"] = ans_tokenized['input_ids']

    return example

def compute_metrics(pred):
    labels_ids = pred.label_ids
    preds_ids = pred.predictions

    # replace -100 back to <pad>
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    preds_ids[preds_ids == -100] = tokenizer.pad_token_id

    labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    return rouge(hypotheses=preds, references1=labels)

def get_argmax(logits, _):
    return torch.cat([distr.argmax(dim=-1) for distr in logits], dim=1)

if OPEN_BOOK:
    dataset = dataset.filter(has_context)

tokenized_datasets = dataset.map(tokenize_function)

custom_args = {}
if dataset_name == "asqa":
    custom_args = {
        "evaluation_strategy": "epoch",
        "save_strategy": "epoch",
        "num_train_epochs": 20,
        "learning_rate": 1e-5,
    }
elif dataset_name == "msmarco-nlgen":
    custom_args = {
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "num_train_epochs": 1,
        "eval_steps": 2500,
        "save_steps": 2500,
        "learning_rate": 1e-4,
    }

if base_model == "bart":
    custom_args["learning_rate"] = 5e-6

training_args = Seq2SeqTrainingArguments(
    output_dir=ft_model_name,
    predict_with_generate=True,
    generation_max_length=100,
    generation_num_beams=5,
    weight_decay=0.01,
    save_total_limit=1,
    report_to="wandb",
    remove_unused_columns=True,
    group_by_length=True,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    metric_for_best_model="loss",
    load_best_model_at_end = True,
    push_to_hub=True,
    fp16=True,
    **custom_args,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-3)],
    preprocess_logits_for_metrics=get_argmax,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.push_to_hub()
