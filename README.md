# Tackling Ambiguous Long-Form Question Answering
Konstantinos Papakostas, Irene Papadopoulou

_This project was developed for the MSc Artificial Intelligence course "Information Retrieval 2" at the University of Amsterdam._

## Abstract
Long-Form Question Answering (LFQA) is a challenging field where models are tasked to generate fluent and concise answers. In order to perform well, the proposed approaches need to effectively summarize relevant evidence and produce an answer in natural language that satisfies the users' needs. We aim to explore some of the common approaches in this field, with a focus on ambiguous questions, which pose an added difficulty as the answers need to cover multiple aspects from a single query. We find that larger generative models produce better answers overall, as their capability to combine relevant passages increases. Additionally, we perform a human evaluation study that confirms the general trends observed when using established LFQA metrics.

## Results
<img src="images/main_results.png" width='80%'/>
<img src="images/random_retrieval.png" width='42%'>
<img src="images/human_head2head.png" width='36%'>

## Training
You can finetune T5-base on the ASQA dataset by running the following command:
```sh
BASE_MODEL=t5-base OPEN_BOOK=true python finetune.py
```

Similarly, you can finetune BART-large with:
```sh
BASE_MODEL=bart-large MODEL_NAME=facebook/bart-large OPEN_BOOK=true python finetune.py
```

All available environment variables are:
- `BASE_MODEL`: the base model being used (defaults to "t5-base")
- `MODEL_NAME`: the HuggingFace name of the model (defaults to **BASE_MODEL**)
- `TOKENIZER_NAME`: the HuggingFace name of the tokenizer (defaults to **MODEL_NAME**)
- `DATASET_HF_USER`: the HuggingFace user that hosts the dataset to train on (defaults to "din0s")
- `DATASET_NAME`: the HuggingFace dataset trian on (defaults to "asqa")
- `OPEN_BOOK`: whether to finetune for the open-book scenario or not (defaults to "false")

## Human Evaluation
To replicate our human evaluation study, you can use the notebook [create_pairwise_comparisons.ipynb](human_annotation/create_pairwise_comparisons.ipynb).

## ASQA Dataset
This project is built on top of the ASQA dataset. For more information, please refer to the [ASQA repository](https://github.com/google-research/language/tree/master/language/asqa). The following setup instructions come from the original codebase.

### Download
To download the ASQA dataset, run:

```
mkdir dataset
gsutil cp -R gs://gresearch/ASQA/data/ASQA.json dataset
```

Note: this requires [gsutil](https://cloud.google.com/storage/docs/gsutil).

### Setup
1. You might want to setup a virtual environment before installation.

2. Install PyTorch by following the instructions [here](https://pytorch.org/get-started/locally/#start-locally).

3. Install python packages and download the Roberta checkpoint by running:

```
sh install.sh
```

### Evaluation in one bash script
```
chmod +x ./eval.sh
./eval.sh ${RESULTS_PATH} ${EXP_NAME}
```
The final results will show on the screen and will also be generated in ./results/${EXP_NAME}/final_eval_results.json.
