from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("din0s/asqa")

# at K
K = [3, 5]
PASSAGE_PATHS = {"bm25":"./passages/bm25_passages.json", "dpr":"./passages/dpr_passages.json"}

for (retriever, path) in PASSAGE_PATHS.items():
    # Load passages
    with open(path, "r") as f:
        passages = json.load(f)

    for k in K:
        JSON_FILENAME = f"{retriever}@{k}.json"
        pred_dict = {}

        for i in dataset["dev"]:
            sample_id = i['sample_id']
            question = i['ambiguous_question']     
            answer = ' '.join(passages[sample_id][:k])
            pred_dict.update({str(sample_id) : answer})

        print("Saving predictions to json file..")
        with open(JSON_FILENAME, 'w') as fp:
            json.dump(pred_dict, fp)