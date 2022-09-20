from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import json


def prepare_question(q: str, k: int = 3) -> str:
    """
    Prepare a question for the model by appending the top-k passages from the BM25 index.

    Args:
        q: The ambiguous question.
        k: The number of passages to append.
    """
    hits = index.search(q, k=k)
    return f"question: {q} context: {' '.join([h.raw for h in hits])}"


def answer_batch(batch: dict[str, list]) -> dict[str]:
    """
    Answer a batch of questions using the seq2seq model.
    This function is used by the `map` method of the `datasets.Dataset` class.

    Args:
        batch: A dict containing the questions to answer under `ambiguous_question`.
    """
    questions = [prepare_question(q) for q in batch["ambiguous_question"]]
    inputs = tokenizer(questions, return_tensors="pt", padding=True, truncation=True)
    gen_outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=100, num_beams=5,
        early_stopping=True,
    )

    batch["answer"] = tokenizer.batch_decode(gen_outputs, skip_special_tokens=True)
    return batch


# Load the dataset, tokenizer, model and index.
dataset = load_dataset("din0s/asqa")
tokenizer = AutoTokenizer.from_pretrained("t5-large", model_max_length=512)
model = AutoModelForSeq2SeqLM.from_pretrained("t5-large")
index = LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')

# Remove unnecessary columns.
col_keep = ["ambiguous_question", "sample_id"]
col_remove = list(set(dataset["dev"].column_names).difference(col_keep))
data = dataset["dev"].remove_columns(col_remove)

# Answer the questions in the dataset.
answers = data.map(answer_batch, batched=True, batch_size=16)

# Save the answers to a JSON file.
with open("t5_bm25_predictions.json", "w") as fp:
    answers_dict = {r["sample_id"]: r["answer"] for r in answers}
    json.dump(answers_dict, fp)
