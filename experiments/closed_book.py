from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, T5TokenizerFast
import json

JSON_FILENAME = "t5_ft_closed_book_predictions_ngram_3.json"
BATCH_SIZE = 64
pred_dict = {}

dataset = load_dataset("din0s/asqa")
tokenizer = T5TokenizerFast.from_pretrained("google/t5-large-ssm-nq")
model = AutoModelForSeq2SeqLM.from_pretrained("irenepap/t5-small-asqa-cb")

print("Extracting T5 predictions..")
for i in range(0, len(dataset["dev"]), BATCH_SIZE):
    
    print("Batch: ", i)
    batch = dataset["dev"][i:i+BATCH_SIZE]
    questions = batch['ambiguous_question']
    
    inputs = tokenizer(questions, return_tensors="pt", padding=True)
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