from datasets import load_dataset

import pandas as pd
import itertools
import glob
import json
import nltk

if __name__ == '__main__':
    nltk.download('stopwords')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    asqa = load_dataset('din0s/asqa', split='dev')

    for file_name in glob.glob("*@*.json"):
        with open(file_name, 'r') as f:
            preds = json.load(f)

        stats = {}
        for sample in asqa:
            # lowercase model output
            model_output = preds[sample['sample_id']].lower()
            # remove stopwords
            filtered_output = [w for w in model_output.split(" ") if w not in stopwords]
            # calculate unique words percentage
            pct_unique = len(set(filtered_output)) / len(filtered_output)

            # flatten gold answers into a single set
            gold_answers = set(itertools.chain(*[p['short_answers'] for p in sample['qa_pairs']]))
            # identify gold tokens that exist in the model output
            gold_in_output = itertools.chain(*[g.split(" ") for g in gold_answers if g.lower() in model_output])
            # count gold answers in output
            count_gold = sum([1 for g in gold_answers if g.lower() in model_output])
            # remove stopwords
            filtered_gold_in_output = [w for w in gold_in_output if w not in stopwords]
            # calculate golden tokens percentage
            pct_gold = len(filtered_gold_in_output) / len(filtered_output)

            stats[sample['sample_id']] = {
                'pct_unique': pct_unique,
                'pct_gold': pct_gold,
                'pct_gold_na': pct_gold if pct_gold > 0 else None,
                'avg_num_found': count_gold,
                'avg_num_found_na': count_gold if count_gold > 0 else None,
                'avg_num_gold': len(gold_answers),
                'pct_gold_found': count_gold / len(gold_answers),
                'pct_gold_found_na': count_gold / len(gold_answers) if count_gold > 0 else None,
            }

        df = pd.DataFrame.from_dict(stats, orient='index').mean(axis=0)
        df[[idx for idx in df.index if 'pct' in idx]] *= 100
        print(file_name)
        print(df.map(lambda e: float(f"{e:.2f}")).to_json())
