# Hummingbird
Hummingbird dataset and code for EMNLP 2021 paper "Does BERT Learn as Humans Perceive? Understanding Linguistic Styles through Lexica"


<img src="human_vs_bert.png" width="400">

## Hummingbird Dataset
Dataset is under ``data/hummingbird`` folder. It contains annotated texts for all the eight styles (politeness, sentiment, anger, disgust, fear, joy, and sadness).

Below is the explanation of each column
* ``human_label``	= annotator's style label for the text
  * 0 if the text is polite, positive, anger, disgust, fear, joy, or sad, 
  * 1 if the text is impolite, negative, not anger, etc, 
  * 0.5 if neutral for "politeness" and "sentiment"
* ``orig_text``	= original text
* ``processed_text`` = text after preprocessing (lower case and removal of some punctuations)
* ``perception_scores``	= human's perception label for the tokens in processed_text

Example aggregation of the whole dataset can be seen in ``hummingbird_data.html``. Blocked text means 3 people agree, blue/red text means 2 people agree, gray text means only 1 person annotate the word as important stylistic cue. 

### token_avg
This directory contains a list of words for each style with their corresponding count (``count``), average perception scores (``avg_attr``), and their
standard deviation (``std_attr``). Ignore the last column (``avg_pred``).

## A Subset of Existing Datasets
A subset of benchmarking dataset is under `data/orig` folder. It has word importance scores from Captum. 
* ``pred_class`` = predicted label, 
  * 0 if the text is polite, positive, anger, disgust, fear, joy, or sad, 
  * 1 if the text is impolite, negative, not anger, etc.,
* ``pred_prob``	= prediction probability
* ``raw_input``	= tokenized text by BERT
* ``attribution_scores`` = word importance scores by integrated gradients from Captum

These existing datasets are extracted from the following previous works:
| Style | Name | Link |
| :---: | :---: | :---: |
| Politeness | StanfordPoliteness | [link](https://www.cs.cornell.edu/~cristian/Politeness.html) |
| Sentiment  |  SentiTreeBank | [link](https://nlp.stanford.edu/sentiment/treebank.html) |
| Offensiveness |  Tweet Datasets for Hate Speech and Offensiveness (HateOffensive)| [link](https://github.com/t-davidson/hate-speech-and-offensive-language) |
|  Emotion |  SemEval 2018| [link](https://competitions.codalab.org/competitions/17751#learn_the_details-datasets) |

### token_avg
This directory contains a list of words for each style with their corresponding count (``count``), average attribution scores (``avg_attr``), their
standard deviation (``std_attr``), and average prediction probability - if it's closer to one then the label is more positive/polite/higher emotion, etc. (``avg_pred``).

## Code
* ``extract_tokens_from_bert_data.py``: code for aggregating BERT's tokenized tokens and create the ``[style]_features.tsv`` file for analysis. 

### model
* ``training.py``: code for training the model
  * Example command for running the code for joy emotion: 
  
  ``python training.py --input_dir ../dataset/emotion_semeval/joy --task_name emotion --output_dir ../model/joy  --temp_dir tmp_out/new_tmp_joy``
  
* ``captum_label.py``: code for testing the model and obtaining word importance scores (attribution scores). 
   * Example command for running the code for joy emotion: 
  
  ``python captum_label.py --data_dir ../dataset/emotion_semeval/joy --model_type bert --emotion joy --do_eval --do_interpret --do_lower_case --model_name_or_path ../model/joy --output_dir ../model/joy --eval_dataset ../dataset/emotion_semeval/joy/dev.tsv``

  * Example command for running the code for politeness: 
  
  ``python captum_label.py --data_dir ../dataset/StanfordPoliteness/ --model_type bert --task StanfordPoliteness --do_eval --do_interpret --do_lower_case --model_name_or_path ../model/politeness --output_dir ../model/politeness --eval_dataset ../dataset/StanfordPoliteness/dev.tsv``


## BibTex
```
# coding=utf-8

@InProceedings{hayati-etal-2021hummingbird,
  author = 	"Hayati, Shirley Anugrah
		and Kang, Dongyeop
		and Ungar, Lyle",
  title = 	"Does BERT Learn as Humans Perceive? Understanding Linguistic Styles through Lexica",
  booktitle = 	"Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
  year = 	"2021",
  publisher = 	"Association for Computational Linguistics",
  location = 	"Punta Cana, Dominican Republic",
  url = 	"https://arxiv.org/pdf/2109.02738.pdf"
}
```

