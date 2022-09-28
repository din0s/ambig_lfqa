from datasets import load_dataset
import json

JSON_FILENAME = "naive_question.json"
pred_dict = {}

dataset = load_dataset("din0s/asqa")

for i in dataset["dev"]:

    sample_id = i['sample_id']
    question = i['ambiguous_question']
    answer = 8 * (" "+ question)
    pred_dict.update({str(sample_id) : answer})

print("Saving predictions to json file..")
with open(JSON_FILENAME, 'w') as fp:
    json.dump(pred_dict, fp)