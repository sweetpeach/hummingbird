import csv
import logging
import os
import sys
from colorama import Fore,Style
from io import open

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

logger = logging.getLogger(__name__)

import torch
from typing import Iterable
from IPython.core.display import display, HTML, Image

from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import InterpretableEmbeddingBase, TokenReferenceBase
from captum.attr import visualization
from captum.attr import configure_interpretable_embedding_layer, remove_interpretable_embedding_layer
from captum.attr._utils.visualization import *

#https://gist.github.com/davidefiocco/3e1a0ed030792230a33c726c61f6b3a5
def summarize_attributions(attributions):
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    return attributions.detach().cpu().numpy()

def add_attributions_to_visualizer(attributions, tokens, pred, pred_ind, label, delta, label_list):
    # storing couple samples in an array for visualization purposes
    vis = VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            label,
                            label_list[label],
                            attributions.sum(),
                            tokens[:len(attributions)],
                            delta)
    return vis

def add_attributions_to_visualizer_pred(attributions, tokens, pred, pred_ind, style_type, delta, label_list):
    # storing couple samples in an array for visualization purposes

    vis = VisualizationDataRecord(
                            attributions,
                            pred,
                            pred_ind,
                            100,
                            style_type,
                            attributions.sum(),
                            tokens[:len(attributions)],
                            delta)
    return vis

def visualize_text_pred(datarecords, filename, eval_task) -> None:
    print("visualize prediction text to ", filename)
    dom = []
    rows = [
        "<table width: 100%>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]

    preds_ids,label_ids = [], []
    tsv_file = filename+".tsv"
    html_file = filename+".html"
    tsv_rows = []
    for datarecord in datarecords:
        color = ''
        # if int(datarecord.pred_class) == int(datarecord.true_class):
        #     color = "bgcolor=#ccccff"
        # else:
        #     color = "bgcolor=#ffb3b3"
        color = "bgcolor=#ccccff"


        rows.append(
            "".join(
                [
                    "<tr {}>".format(color),
                    # format_classname(datarecord.true_class),
                    # format_classname(datarecord.target_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            int(datarecord.pred_class), float(datarecord.pred_prob)
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(float(datarecord.attr_score))),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )
        prediction = str(int(datarecord.pred_class))
        prediction_prob = "{:.2f}".format(float(datarecord.pred_prob))
        the_raw_input = datarecord.raw_input
        word_attr_scores = datarecord.word_attributions
        temp = []
        importance_list = []
        for word, importance in zip(the_raw_input, word_attr_scores[:len(the_raw_input)]):
            word = format_special_tokens(word)
            temp_str = word #+ " ({:.2f})".format(importance)
            temp.append(temp_str)
            importance_list.append("{:.2f}".format(importance))
        text = " ".join(temp)
        attr_list = " ".join(importance_list)
        tsv_row = [prediction, prediction_prob, text, attr_list]
        tsv_rows.append(tsv_row)

        preds_ids.append(int(datarecord.pred_class))
        label_ids.append(int(datarecord.true_class))

    
    with open(tsv_file, 'w') as writer:
        header = "pred_class\tpred_prob\traw_input\tattribution_scores\n"
        writer.write(header)
        for row in tsv_rows:
            to_be_written = "\t".join(row)
            writer.write(to_be_written+"\n")

    result = compute_metrics(eval_task, np.array(preds_ids), np.array(label_ids))
    dom.append("<p>Samples: {}, {}</p>".format(len(preds_ids), result))

    dom.append("".join(rows))
    dom.append("</table>")

    html = HTML("".join(dom))
    print("done")
    # print(html.data)
    with open(html_file, 'w') as f:
        f.write(html.data)
    print("finish writing to ", filename)

def visualize_text(datarecords, filename, eval_task) -> None:
    dom = []
    rows = [
        "<table width: 100%>"
        "<tr><th>True Label</th>"
        "<th>Predicted Label</th>"
        "<th>Attribution Label</th>"
        "<th>Attribution Score</th>"
        "<th>Word Importance</th>"
    ]

    preds_ids,label_ids = [], []
    for datarecord in datarecords:
        color = ''
        if int(datarecord.pred_class) == int(datarecord.true_class):
            color = "bgcolor=#ccccff"
        else:
            color = "bgcolor=#ffb3b3"


        rows.append(
            "".join(
                [
                    "<tr {}>".format(color),
                    format_classname(datarecord.true_class),
                    # format_classname(datarecord.target_class),
                    format_classname(
                        "{0} ({1:.2f})".format(
                            int(datarecord.pred_class), float(datarecord.pred_prob)
                        )
                    ),
                    format_classname(datarecord.attr_class),
                    format_classname("{0:.2f}".format(float(datarecord.attr_score))),
                    format_word_importances(
                        datarecord.raw_input, datarecord.word_attributions
                    ),
                    "<tr>",
                ]
            )
        )
        preds_ids.append(int(datarecord.pred_class))
        label_ids.append(int(datarecord.true_class))

    result = compute_metrics(eval_task, np.array(preds_ids), np.array(label_ids))
    dom.append("<p>Samples: {}, {}</p>".format(len(preds_ids), result))

    dom.append("".join(rows))
    dom.append("</table>")

    html = HTML("".join(dom))
    with open(filename, 'w') as f:
        f.write(html.data)


def highlight(input):
    return Fore.YELLOW+str(input)+Style.RESET_ALL



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id, ref_ids=False):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id
        if ref_ids:
            self.ref_ids = ref_ids




class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()


    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

    @classmethod
    def _read_text(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

def convert_examples_to_features(examples,
                                 tokenizer,
                                 max_length=512,
                                 task=None,
                                 label_list=None,
                                 output_mode=None,
                                 pad_on_left=False,
                                 pad_token=0,
                                 pad_token_segment_id=0,
                                 mask_padding_with_zero=True,
                                 verbose=True):
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    # print("num of examples: ", len(examples))
    # quit()
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        inputs = tokenizer.encode_plus(
            example.text_a, example.text_b, add_special_tokens=True, max_length=max_length, return_token_type_ids=True,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # for interpret functions
        ref_ids = [input_ids[0]] + [pad_token] * len(input_ids[1:-1]) + [input_ids[-1]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            ref_ids = ([pad_token] * padding_length) + ref_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            ref_ids = ref_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(
            len(attention_mask), max_length
        )
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(
            len(token_type_ids), max_length
        )

        if example.label is None:
            print("example label is NONE")
            label_id = None
        else:
            if output_mode == "classification":
                # print(label_map)
                # print(example.label)
                # quit()
                label_id = label_map[example.label]
            elif output_mode == "regression":
                label_id = float(example.label)
            else:
                raise KeyError(output_mode)

        if verbose and ex_index < 2:
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            logger.info("*** Example ***")
            logger.info("guid: %s" % (highlight(example.guid)))
            logger.info("tokens: %s" % highlight(" ".join([str(x) for x in tokens])))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            #logger.info("ref_ids: %s" % " ".join([str(x) for x in ref_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: {} ({})".format(label_id, example.label))

        features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              label_id=label_id,
                              ref_ids=ref_ids))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)

    label_set = list(set(labels))
    new_labels, new_preds =  [], []
    unmatched_label_prediction_cnt = 0
    for l,p in zip(labels,preds):
        if p not in label_set:
            unmatched_label_prediction_cnt += 1
        else:
            new_preds.append(p)
            new_labels.append(l)
    if unmatched_label_prediction_cnt > 0:
        # from pdb import set_trace; set_trace()
        f1 = f1_score(y_true=new_labels, y_pred=new_preds, average='macro')
    else:
        f1 = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1": f1,
        "acc_and_f1": (acc + f1) / 2,
        "unmatched_label_prediction_cnt": unmatched_label_prediction_cnt,
        "cnt": len(preds)
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }

class Emotion(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    """Processor for the GYAFC data set (SLUE version)."""
    def get_labels(self, emotion_label):
        """See base class."""
        return [emotion_label, "not_"+emotion_label]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        # print(lines)
        for (i, line) in enumerate(lines):
            # if i == 0:
            #     continue
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                text_a = line[1]
                if len(line) == 3:
                    label = line[2]
                    # print(line)
                    # print("label inside: ", label)
                else:
                    label = None
            # print(label)
            # quit()
            # print(text_a)
            # quit()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        # print(examples[0].label)
        # quit()
        return examples

class StanfordPolitenessProcessor(DataProcessor):
    """Processor for the StanfordPoliteness data set (SLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        return ["polite", "impolite"]        

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            # if len(line) != 4:
            #     continue
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:
                    text_a = line[2]
                    if len(line) == 4:
                        label = 'polite' if float(line[3]) > 0 else 'impolite'
                    else:
                        label = None
                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class HateOffensive(DataProcessor):
    """Processor for the HateOffensive data set (SLUE version)."""
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    def get_labels(self):
        """See base class."""
        # return ["hate", "offensive", "neither"]
        return ["offensive", "neither"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                try:

                    if line[1] != '0':
                        text_a = line[2]
                    if len(line) >= 2:
                        label = 'offensive' if line[1]=='1' else 'neither'                    
                    else:
                        label = None

                except Exception as e:
                    print(i,e,line)
                    from pdb import set_trace; set_trace()
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class SentiTreeBankProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_test_input_examples(self, test_input):
        """See base class."""
        return self._create_examples(
            self._read_text(os.path.join(test_input)), "test_input")

    """Processor for the Sentiment TreeBank data set (SLUE version)."""
    """We make coarse-grained version (e.g., positive, negative), which can be extended to fine-grained one easily"""
    def get_labels(self):
        """See base class."""
        # return ["very positive", "positive", "neutral", "negative", "very negative"]
        return ["positive", "negative"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        labels = self.get_labels()
        for (i, line) in enumerate(lines):
            if i == 0 and len(line) != 1:
                continue
            guid = "%s-%s" % (set_type, i)
            if len(line) == 1:
                text_a = line[0]
                label = None
            else:
                if line[4] not in labels:
                    continue
                text_a = line[1]
                if len(line) == 5:
                    label = line[4]
                else:
                    label = None
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



processors = {    
    "emotion": Emotion,
    "stanfordpoliteness": StanfordPolitenessProcessor,
    "hateoffensive": HateOffensive,
    "sentitreebank": SentiTreeBankProcessor,
}

output_modes = {
    "emotion":"classification",
    "stanfordpoliteness": "classification",
    "hateoffensive": "classification",
    "sentitreebank": "classification",
}

def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)

    if task_name == "emotion":
        return acc_and_f1(preds, labels)
    elif task_name == "stanfordpoliteness":
        return acc_and_f1(preds, labels)
    elif task_name == "hateoffensive":
        return acc_and_f1(preds, labels)
    elif task_name == "sentitreebank":
        return acc_and_f1(preds, labels)
    else:
        raise KeyError(task_name)
