from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import random

import json
import torch

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_var_or_default(var, default):
    return os.environ[var] if var in os.environ else default

model_name = get_var_or_default("MODEL_NAME", "t5-base")
tokenizer_name = get_var_or_default("TOKENIZER_NAME", model_name)

OPEN_BOOK = get_var_or_default("OPEN_BOOK", "false").lower() == "true"
RANDOM_RETRIEVAL = get_var_or_default("RANDOM_RETRIEVAL", "false").lower() == "true"
BATCH_SIZE = int(get_var_or_default("BATCH_SIZE", "8"))

SAVE_PATH = get_var_or_default("SAVE_PATH", "./output.json")

if OPEN_BOOK:
    top_k = int(get_var_or_default("TOP_K", "5"))
    passages_path = get_var_or_default("PASSAGES_PATH", "./passages/dpr_passages.json")

    with open(passages_path, "r") as f:
        passages = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
dataset = load_dataset("din0s/asqa")

variant = f"top-{top_k} from {passages_path}" if OPEN_BOOK else "closed-book"
print(f"Generating ASQA answers with {model_name} ({variant})")

predictions = {}
with torch.no_grad():
    N = len(dataset["dev"])
    for i in range(0, N, BATCH_SIZE):
        print(f"Batch #{(i // BATCH_SIZE)+1} of {(N // BATCH_SIZE)+1}")

        batch = dataset["dev"][i:i+BATCH_SIZE]
        ids = batch['sample_id']
        questions = [f"question: {q}" for q in batch['ambiguous_question']]

        if OPEN_BOOK:
            if RANDOM_RETRIEVAL:
                all_passages = [pasg for pasgs in passages.values() for pasg in pasgs]
                contexts = [random.sample(all_passages, top_k) for _ in range(BATCH_SIZE)]
            else:
                contexts = [passages[id][:top_k] for id in ids]

            if "bart" in model_name:
                contexts = ["<P> " + " <P> ".join(c) for c in contexts]
            else:
                contexts = [" | ".join(c) for c in contexts]

            questions = [f"{q} context: {c}" for q, c in zip(questions, contexts)]

        inputs = tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            no_repeat_ngram_size=3,
            max_length=100,
            num_beams=5,
            early_stopping=True,
        )

        answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.update({str(id): ans for id, ans in zip(ids, answers)})

print("Saving to json file...")
with open(SAVE_PATH, "w") as f:
    json.dump(predictions, f)
