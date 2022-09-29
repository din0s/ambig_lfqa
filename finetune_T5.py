from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast, Seq2SeqTrainingArguments, Trainer, set_seed, DataCollatorForSeq2Seq, EarlyStoppingCallback
import evaluate
import torch
import json

set_seed(42)
model_name = "google/t5-small-ssm-nq"
train_batch_size = 8
eval_batch_size = 8
# name the model to push to hub
ft_model_name = "t5-small-asqa-ob-bm25@5"

# Load passages (only for open book)
bm25_path = "./passages/bm25_passages_all.json"

with open(bm25_path, "r") as f:
    passages = json.load(f)

def has_context(example):
    context = [i["context"] for i in example["qa_pairs"]]
    context = list(filter(("No context provided").__ne__, context))
    if context:
        return True
    return False

def tokenize_function(example):
    # With ASQA context
    # context = [i["context"] for i in example["qa_pairs"]]
    # context = list(filter(("No context provided").__ne__, context))
    # context = ' | '.join(context)

    # With retrieved context
    sample_id = example['sample_id']
    context = ' | '.join(passages[sample_id][:3])
    
    question_tokenized = tokenizer(f"question: {example['ambiguous_question']} context: {context}", truncation=True, max_length=512)
    example['input_ids'] = question_tokenized['input_ids']

    ans_tokenized = tokenizer(text_target=example['annotations'][0]['long_answer'], truncation=True, max_length=512)
    example["labels"] = ans_tokenized['input_ids']
        
    example['long_answer'] = example['annotations'][0]['long_answer']

    return example

def compute_metrics(pred):
    labels_ids = pred.label_ids
    preds_ids = pred.predictions
    # pad token from -100 to 0
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    preds_ids[preds_ids == -100] = tokenizer.pad_token_id
    labels = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)
    preds = tokenizer.batch_decode(preds_ids, skip_special_tokens=True)
    return rouge.compute(predictions = preds, references = labels)

def get_argmax(logits, _):
    preds_ids = torch.cat([torch.argmax(i, dim=-1) for i in logits], dim=1)
    return preds_ids
    
dataset = load_dataset("din0s/asqa")

# Remove unnecessary columns
col_keep = ["ambiguous_question", "qa_pairs", "annotations", "sample_id"]
col_remove = list(set(dataset["train"].column_names).difference(col_keep))
dataset = dataset.remove_columns(col_remove)

tokenizer = T5TokenizerFast.from_pretrained(model_name)

# With ASQA context
# tokenized_datasets = dataset.filter(has_context).map(tokenize_function, remove_columns=["annotations"])
# With BM25 passages
tokenized_datasets = dataset.map(tokenize_function, remove_columns=["annotations"])

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

training_args = Seq2SeqTrainingArguments(
    output_dir=ft_model_name, 
    evaluation_strategy="epoch",
    remove_unused_columns=True,
    group_by_length=True,
    per_device_train_batch_size=train_batch_size, 
    per_device_eval_batch_size=eval_batch_size, 
    # eval_accumulation_steps=8,
    num_train_epochs=50,
    save_strategy="epoch",
    save_total_limit=1,
    report_to="wandb",
    learning_rate=5e-4,
    metric_for_best_model="rougeLsum",
    push_to_hub=True,
    load_best_model_at_end = True,
)

# Load metric
rouge = evaluate.load('rouge')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["dev"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5, early_stopping_threshold=1e-3)],
    preprocess_logits_for_metrics=get_argmax,
)

trainer.train()

trainer.push_to_hub()