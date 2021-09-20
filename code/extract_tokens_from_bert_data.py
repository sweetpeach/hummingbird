import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def remove_cls_and_bert(sentence, attr_score_list):
	tokens = sentence.split(" ")
	attr_scores = attr_score_list.split(" ")
	new_tokens = tokens
	new_attrs = attr_scores
	if tokens[0].lower() == "[cls]":
		if tokens[len(tokens)-1].lower() == "[sep]":
			new_tokens = tokens[1:len(tokens)-1]
			new_attrs = attr_scores[1:len(tokens)-1]
	if len(new_tokens) != len(new_attrs):
		print("error here")
	return new_tokens, new_attrs

def match_bert_token_to_original(bert_tokens, orig_text):
	raw_orig = orig_text.lower().split(" ")
	orig = []
	for word in raw_orig:
		if word != "":
			orig.append(word.strip())
	bert_idx = 0
	last_bert_idx = 1
	orig_idx = 0
	#an art ##ful --> An artful
	orig_to_bert_mapping = {}
	orig_idx2token = {}
	if "babies" in orig_text:
		print(orig_text)
		print(bert_tokens)

	if "#strategy" in orig:
		print(orig_text)
		print(orig)
		print(bert_tokens)

	while bert_idx < len(bert_tokens) and orig_idx < len(orig):

		current_orig = orig[orig_idx]
		current_bert = bert_tokens[bert_idx]
		orig_to_bert_mapping[orig_idx] = [bert_idx]
		
		bert_idx += 1
		orig_idx2token[orig_idx] = current_orig
		if current_bert != current_orig:			
			combined = current_bert
			last_bert_idx = bert_idx
			while last_bert_idx < len(bert_tokens):
				next_part = bert_tokens[last_bert_idx].replace("##","").strip()				
				combined += next_part
				orig_to_bert_mapping[orig_idx].append(last_bert_idx)
				# if current_orig == "well-established":
				# 	print("combined: ", combined)
				if combined == current_orig:					
					bert_idx = last_bert_idx + 1
					break
				else:
					last_bert_idx += 1

		orig_idx += 1
	return orig_to_bert_mapping, orig_idx2token

def make_new_attr_score(orig2bert, bert_attribution_scores, do_debug=False):
	if do_debug:
		print(len(bert_attribution_scores) == len(orig2bert))
		quit()
	new_score = {}
	for key, value in orig2bert.items():
		the_list = []
		for bert_idx in value:
			the_list.append(bert_attribution_scores[bert_idx])
		np_list = np.array(the_list).astype(np.float)
		
		attr_avg = np.mean(np_list)
		new_score[key] = attr_avg
	return new_score

def process_bert(bert_data, orig_data):
	
	bert_sentences = bert_data["raw_input"].values
	bert_attribution_scores = bert_data["attribution_scores"].values
	pred_probs = bert_data["pred_prob"].values
	word2attributions = defaultdict(list)
	pred_token_dic = defaultdict(list)

	counter = 0
	for sentence, score, orig_text, prob in zip(bert_sentences, bert_attribution_scores, orig_data, pred_probs):
		bert_tokens, attr_list = remove_cls_and_bert(sentence, score)
		orig2bert, idx2token = match_bert_token_to_original(bert_tokens, orig_text)
		do_debug = False
		new_attr_dict = make_new_attr_score(orig2bert, attr_list, do_debug)

		for idx, score in new_attr_dict.items():
			real_word = idx2token[idx].replace(",", "").replace("?", "").replace(".", "").replace("!", "").replace("?", "").replace(";", "").replace('"', "").strip()
			word2attributions[real_word].append(score)
			pred_token_dic[real_word].append(prob)

		counter += 1

	return word2attributions, pred_token_dic

def write_to_file(output_file, attr_token_dic, pred_token_dic):
	attr_avg = ((np.average(values), tks) for tks, values in attr_token_dic.items())

	print("Writing file to ", output_file)
	with open(output_file, "w") as writer:
		header = "idx\tword\tcount\tavg_attr\tstd_attr\tavg_pred\n"
		writer.write(header)
		for idx, (avg, tks)  in enumerate(sorted(attr_avg, reverse=True)):
			token_edited = tks.replace(",", "").replace("?", "").replace(".", "").replace("!", "").replace("?", "").replace(";", "").replace('"', "").strip()
			if str(token_edited) != "":
				writer.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
				idx, token_edited, len(attr_token_dic[tks]),
				np.average(attr_token_dic[tks]),
				np.std(attr_token_dic[tks]),
				np.average(pred_token_dic[tks])
				))
	print("Finish writing...")

def main():
	bert_dir = "../data/orig/"
	the_style = "politeness"
	bert_file = bert_dir+the_style+".tsv"
	orig_file = "../data/text_only_new.txt"

	bert_data = pd.read_csv(bert_file, sep="\t")
	orig_data = pd.read_csv(orig_file, sep="\t", header=None)[0].values
	word2attr, pred_dict = process_bert(bert_data, orig_data)
	output_file = bert_dir+the_style+"_features.tsv"
	write_to_file(output_file, word2attr, pred_dict)
main()