from pytorch_transformers import BertForSequenceClassification, BertConfig, BertModel, BertTokenizer
import pandas as pd
import numpy as np
import json, re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from pytorch_transformers import AdamW, WarmupLinearSchedule
import os

def set_seed(n_gpu, the_seed):
	random.seed(the_seed)
	np.random.seed(the_seed)
	torch.manual_seed(the_seed)
	if n_gpu > 0:
		torch.cuda.manual_seed_all(the_seed)

config = BertConfig.from_pretrained('bert-base-uncased')
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

def load_dataset(input_path, dataset_name, lower, label_to_idx=None):
	utterances = []
	labels = []
	if "train" in input_path:
		label_to_idx = {}

	if dataset_name == "politeness":
		if "train" in input_path:
			data_file = pd.read_csv(input_path, sep="\t")		
			counter_data = 0
			for idx, row in data_file.iterrows():
				utterance = row[3]
				if lower:
					utterance = utterance.lower()
				label = "polite" if float(row[0]) == 0 else "impolite"
				if "train" in input_path and label not in label_to_idx:
					label_to_idx['polite'] = 0
					label_to_idx['impolite'] = 1
				label = label_to_idx[label]
				utterances.append(utterance)
				labels.append(label)
				counter_data += 1				
			print("Num of data: ", counter_data)
		else:

			data_file = pd.read_csv(input_path, sep="\t", header=None)		
			counter_data = 0
			for idx, row in data_file.iterrows():				
				utterance = row[2]
				if lower:
					utterance = utterance.lower()
				label = "polite" if float(row[3]) > 0 else "impolite"
				if "train" in input_path and label not in label_to_idx:
					label_to_idx['polite'] = 0
					label_to_idx['impolite'] = 1
				label = label_to_idx[label]
				utterances.append(utterance)
				labels.append(label)
				counter_data += 1
			print("Num of data: ", counter_data)

	if dataset_name == "offensive":
		data_file = pd.read_csv(input_path, sep="\t", header=None)
		for idx, row in data_file.iterrows():
			utterance = row[2]
			if int(row[1]) > 0:
				if int(row[1]) == 1:
					label = "offensive"
				if int(row[1]) == 2:
					label = "neither"
				if "train" in input_path and label not in label_to_idx:
					label_to_idx['offensive'] = 0						
					label_to_idx['neither'] = 1
				label = label_to_idx[label]
				utterances.append(utterance)
				labels.append(label)
		

	if dataset_name == "sentiment":
		data_file = pd.read_csv(input_path, sep="\t")
		for idx, row in data_file.iterrows():
			utterance = row['phrase']
			label = row['coarse']
			if label != 'neutral':
				if "train" in input_path and label not in label_to_idx:
					label_to_idx[label] = len(label_to_idx)
				label = label_to_idx[label]
				if lower:
					utterance = utterance.lower()
				utterances.append(utterance)
				labels.append(label)

	if dataset_name == "emotion":
		data_file = pd.read_csv(input_path, sep="\t")
		for idx, row in data_file.iterrows():
			utterance = row['tweet']
			label = row['emotion']
			if "train" in input_path and label not in label_to_idx:
				if 'not' in label:
					label_to_idx[label] = 1
				else:
					label_to_idx[label] = 0
			if lower:
				utterance = utterance.lower()
			label = label_to_idx[label]
			utterances.append(utterance)
			labels.append(label)

	print(label_to_idx)
	return utterances, labels, label_to_idx


def process_data(sentences):

	MAX_LEN = 128
	input_ids = []
	counter_nan = 0
	for sent in sentences:
		sent = str(sent)
		tokenized = sent.split(" ")
		counter_nan += 1
		
		input_id = tokenizer.encode(sent, add_special_tokens=True)
		input_ids.append(input_id)

	input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

	# Create attention masks
	attention_masks = []
	# Create a mask of 1s for each token followed by 0s for padding
	for seq in input_ids:
		seq_mask = [float(i>0) for i in seq]
		attention_masks.append(seq_mask)

	return input_ids, attention_masks

def main():
	parser = argparse.ArgumentParser()

	## Required parameters
	parser.add_argument("--input_dir", default=None, type=str,
	help="The input data dir for training.")
	parser.add_argument("--temp_dir", default=None, type=str, help="The input data dir for development or testing.")
	parser.add_argument("--task_name", default=None, type=str, required=True)
	parser.add_argument("--output_dir", default=None, type=str, help="The output directory where the model predictions and checkpoints will be written.")

	args = parser.parse_args()
	input_dir = args.input_dir
	temp_dir = args.temp_dir
	task_name = args.task_name
	output_dir = args.output_dir
	
	print("[INFO] Input Dir: ", input_dir)
	print("[INFO] Task: ", task_name)
	print("[INFO] Temp Dir: ", temp_dir)
	print("[INFO] Output dir: ", output_dir)

	train_dataset = input_dir + "/train.tsv"
	dev_dataset = input_dir + "/dev.tsv"
	train_utts, train_labels, label2idx = load_dataset(train_dataset, task_name, lower)
	if task_name != "emotion":
		
		label_file = task_name+".labels"
	else:
		emotion_label = "joy"
		label_file = task_name + "_"+emotion_label+".labels"
		print("Emotion: " + label_file)
	print("task name: ", task_name)
	torch.save(label2idx, label_file)
	print("Saving label to idx to :", label_file)
	dev_utts, dev_labels, _ = load_dataset(dev_dataset, task_name, lower, label2idx)

	num_labels = len(label2idx)
	model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=num_labels)
	train_inputs, train_masks = process_data(train_utts)
	
	print("Finish processing data")

	train_inputs = torch.tensor(train_inputs)
	dev_inputs = torch.tensor(dev_inputs)
	train_labels = torch.tensor(train_labels)
	dev_labels = torch.tensor(dev_labels)
	train_masks = torch.tensor(train_masks)
	dev_masks = torch.tensor(dev_masks)

	batch_size = 8
	set_seed(1, 23)
	
	train_data = TensorDataset(train_inputs, train_masks, train_labels)
	train_sampler = RandomSampler(train_data)
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
	dev_data = TensorDataset(dev_inputs, dev_masks, dev_labels)
	dev_sampler = SequentialSampler(dev_data)
	dev_dataloader = DataLoader(dev_data, sampler=dev_sampler, batch_size=batch_size)

	# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = model.cuda()
	# BERT fine-tuning parameters
	param_optimizer = list(model.named_parameters())
	no_decay = ['bias', 'gamma', 'beta']
	optimizer_grouped_parameters = [
		{'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
		 'weight_decay_rate': 0.01},
		{'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
		 'weight_decay_rate': 0.0}
	]

	optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
	
	def flat_accuracy(preds, labels):
		pred_flat = np.argmax(preds, axis=1).flatten()
		labels_flat = labels.flatten()
		return np.sum(pred_flat == labels_flat) / len(labels_flat)

	# Store our loss and accuracy for plotting
	train_loss_set = []
	# Number of training epochs 
	epochs = 3
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	n_gpu = torch.cuda.device_count()
	torch.cuda.get_device_name(0)
	counter = 0
	for _ in trange(epochs, desc="Epoch"):  

		## TRAINING
		model.train()  		
		tr_loss = 0
		nb_tr_examples, nb_tr_steps = 0, 0		
		epoch_iterator = tqdm(train_dataloader, desc="Iteration")
		for step, batch in enumerate(epoch_iterator):
		
			batch = tuple(t.to(device) for t in batch)			
			b_input_ids, b_input_mask, b_labels = batch			
			optimizer.zero_grad()			
			outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
			loss = outputs[0]
			train_loss_set.append(loss.item())
			# Backward pass
			loss.backward()
			# Update parameters and take a step using the computed gradient
			optimizer.step()
			# Update tracking variables
			tr_loss += loss.item()
			nb_tr_examples += b_input_ids.size(0)
			nb_tr_steps += 1

		print("Train loss: {}".format(tr_loss/nb_tr_steps))
		
		## VALIDATION
		
		model.eval()		
		eval_loss, eval_accuracy = 0, 0
		nb_eval_steps, nb_eval_examples = 0, 0
				
		for batch in dev_dataloader:			
			batch = tuple(t.to(device) for t in batch)
			b_input_ids, b_input_mask, b_labels = batch

			with torch.no_grad():				
				outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)    
				logits = outputs[0]
			
			preds = logits.detach().cpu().numpy()
			label_ids = b_labels.to('cpu').numpy()			
			tmp_eval_accuracy = flat_accuracy(preds, label_ids)
			print(label_ids)
			eval_accuracy += tmp_eval_accuracy
			nb_eval_steps += 1
		dev_acc = eval_accuracy/nb_eval_steps
		print("Dev Accuracy: {}".format(dev_acc))
		if dev_acc == 1.0 or dev_acc > 0.999:
			break
				
		temp_dir_model = temp_dir +"/"+ str(counter)
		if not os.path.exists(temp_dir_model):
			os.makedirs(temp_dir_model)

		model.save_pretrained(temp_dir_model)
		tokenizer.save_pretrained(temp_dir_model)

		print("[EPOCH {}] Saving checkpoint to {}".format(counter, temp_dir_model))
		counter += 1

	output_dir_model = output_dir
	if not os.path.exists(output_dir_model):
		os.makedirs(output_dir_model)

	print("Saving model to: " , output_dir_model)
	model.save_pretrained(output_dir_model)
	tokenizer.save_pretrained(output_dir_model)


main()