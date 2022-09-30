from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import json
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

JSON_FILENAME = "t5_ft_ob_dpr_predictions.json"
BATCH_SIZE = 64
pred_dict = {}

dataset = load_dataset("din0s/asqa")
tokenizer = T5TokenizerFast.from_pretrained("google/t5-small-ssm-nq")
model = AutoModelForSeq2SeqLM.from_pretrained("din0s/t5-small-asqa-ob")
model = model.to(device)

# Load passages (only for open book)
dpr_path = "./passages/dpr_passages.json"

with open(dpr_path, "r") as f:
    passages = json.load(f)

print("Extracting T5 predictions..")
for i in range(0, len(dataset["dev"]), BATCH_SIZE):
    
    print("Batch: ", i)
    batch = dataset["dev"][i:i+BATCH_SIZE]
    questions = batch['ambiguous_question']
    context = [passages[i][0] for i in batch['sample_id']]

    q_and_a = [f"question: {q} context: {c}" for q, c in zip(questions, context)]
    
    inputs = tokenizer(q_and_a, return_tensors="pt", padding=True).to(device)
    
    gen_outputs = model.generate(
        inputs["input_ids"], 
        attention_mask=inputs["attention_mask"], 
        no_repeat_ngram_size=3,
        max_length=100, 
        num_beams=5, 
        early_stopping=True,)

    answers = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)

    id_ans_list = list(zip(batch['sample_id'], answers))
    pred_dict.update({str(sample_id) : ans for sample_id, ans in id_ans_list})

print("Saving predictions to json file..")
with open(JSON_FILENAME, 'w') as fp:
    json.dump(pred_dict, fp)