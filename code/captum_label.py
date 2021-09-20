import argparse
import glob
import os
import json
import random
import coloredlogs, logging
from collections import defaultdict
from pprint import pprint
import numpy as np
from typing import Iterable
from IPython.core.display import display, HTML, Image
from tqdm import tqdm, trange

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
from captum.attr import LayerIntegratedGradients
from transformers import *

from utils import (compute_metrics, convert_examples_to_features, output_modes, processors, highlight)
from configuration_new_captum import _parse_args, _set_barrier,_set_barrier_v2, _set_fp16, _set_seed, logger
from utils import summarize_attributions, add_attributions_to_visualizer, visualize_text, visualize_text_pred, add_attributions_to_visualizer_pred

import os

def evaluate(args, model, tokenizer, prefix=""):

    eval_task_names = args.task_name
    eval_outputs_dirs =args.output_dir

    results = {}

    model.eval()
    model.zero_grad()

    # functions gradient-based interpretation
    if args.do_interpret:
        def predict(inputs):
            return model(inputs)[0]

        def custom_forward(inputs):
            preds = predict(inputs)
            return torch.softmax(preds, dim = 1)[0][0].unsqueeze(-1)

        if args.model_type in ['bert']:
            lig = LayerIntegratedGradients(custom_forward, model.bert.embeddings)
        attr_token_dic = defaultdict(list)
        pred_token_dic = defaultdict(list)
        vis_list = []


    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)
        processor = processors[eval_task]()
        if args.task_name == "emotion":
            emotion_label = args.emotion
            label_list = processor.get_labels(emotion_label)
        else:
            label_list = processor.get_labels()

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        for batch in tqdm(eval_dataloader, desc="Evaluating", ncols=8):
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[2],
                      'token_type_ids': batch[3] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use token_type_ids
                      }
            if args.eval_dataset is None:
                inputs['labels'] = batch[4]


            with torch.no_grad():
                outputs = model(**inputs)

            if len(outputs) >= 2:
                tmp_eval_loss, logits = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            elif len(outputs) == 1:
                logits = outputs[0]

            nb_eval_steps += 1
            if args.output_mode == "classification":
                logits = torch.nn.functional.softmax(logits, dim=1)

            if preds is None:
                preds = logits.detach().cpu().numpy()
                if 'labels' in inputs:
                    out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                if 'labels' in inputs:
                    out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)



            # captum doesn't support batching well, so interpreting per instance
            if args.do_interpret:
                nz = (batch[0]!=0).sum(dim=1)
                # print("batch: ", batch)
                for logit, nz_one, input_ids, ref_ids in zip(logits, nz, batch[0], batch[1]):
                    input_ids = input_ids[:nz_one].unsqueeze(0)
                    ref_ids = ref_ids[:nz_one].unsqueeze(0)
                    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].detach().cpu().numpy().tolist())


                    attributions, delta = lig.attribute(inputs=input_ids,
                                                    baselines=ref_ids,
                                                    n_steps=30,
                                                    internal_batch_size=8,
                                                    return_convergence_delta=True)

                    attributions = summarize_attributions(attributions)
                    pred_idx = torch.argmax(logit)
                    logit_for_vis = logit[0]
                    

                    if args.task_name.lower() == "stanfordpoliteness":
                        style_type = "polite"
                    elif args.task_name == "emotion":
                        style_type = args.emotion
                    elif args.task_name.lower() == "sentitreebank":
                        style_type = "sentiment"
                    elif args.task_name.lower() == "hateoffense":
                        style_type = "offensive"
                    else:
                        style_type = "nothing"

                    vis = add_attributions_to_visualizer_pred(
                                        attributions,
                                        tokens,
                                        logit_for_vis,
                                        pred_idx,
                                        style_type,
                                        delta, label_list)
                    vis_list.append(vis)                    
                    
                    for token, attr in zip(tokens[1:-1], attributions[1:-1]):
                        attr_token_dic[token].append(attr)                    

        eval_loss = eval_loss / nb_eval_steps        

        if out_label_ids is not None:
            result = compute_metrics(eval_task, preds_ids, out_label_ids)
            results.update(result)

        if results:
            if args.eval_dataset is not None:
                basename_eval_dataset = os.path.basename(args.eval_dataset).replace('.txt','')
                result_file = "eval_results_{}.txt".format(basename_eval_dataset)
                output_eval_file = os.path.join(eval_output_dir, result_file)

            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results {} *****".format(prefix))
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(highlight(result[key])))
                    writer.write("%s = %s\n" % (key, str(result[key])))


        if preds is not None:
            if args.eval_dataset is not None:
                basename_eval_dataset = os.path.basename(args.eval_dataset).replace('.txt','')
                output_pred_file = os.path.join(eval_output_dir, "eval_preds_{}.txt".format(basename_eval_dataset))
            else:
                output_pred_file = os.path.join(eval_output_dir, "eval_preds_test.txt")
            with open(output_pred_file, "w") as writer:
                logger.info("***** Prediction results {} *****".format(prefix))
                logger.info("***** Output: {} ******".format(output_pred_file))
                for idx, pred in enumerate(preds):                    
                    writer.write("{}\n".format(
                       '\t'.join([str(p) for p in pred])
                    ))
                    if idx % 100000 == 0:
                        writer.flush()


        # visualize interpret results
        if args.do_interpret:
            if args.eval_dataset is not None:
                basename_eval_dataset = os.path.basename(args.eval_dataset).replace('.txt','')

                txt_file = "eval_interpret_{}.txt".format(basename_eval_dataset)
                output_interpret_file = os.path.join(eval_output_dir, txt_file)
                csv_file = "eval_interpret_feature_{}.csv".format(basename_eval_dataset)
                output_interpret_feature_file = os.path.join(eval_output_dir, csv_file)

                html_file = "eval_attribution_{}".format(basename_eval_dataset)
                output_interpret_file2 = os.path.join(eval_output_dir, html_file)
                print("do visualize")
                visualize_text_pred(vis_list, output_interpret_file2, eval_task)
            else:
                html_file = "eval_interpret_test.html"
                csv_file = "eval_interpret_feature_test.csv"
                
                output_interpret_file = os.path.join(eval_output_dir, html_file)
                output_interpret_feature_file = os.path.join(eval_output_dir, csv_file)


            logger.info("***** Interpret results saved: {} *****".format(output_interpret_file))
            # visualize_text(vis_list, output_interpret_file, eval_task)


            attr_avg = ((np.average(values), tks) for tks, values in attr_token_dic.items())
            logger.info("***** Interpret feature file saved: {} *****".format(output_interpret_feature_file))
            with open(output_interpret_feature_file, "w") as writer:
                for idx, (avg, tks)  in enumerate(sorted(attr_avg, reverse=True)):
                    writer.write("{}\t{}\t{}\t{}\t{}\t{}\n".format(
                        idx, tks, len(attr_token_dic[tks]),
                        np.average(attr_token_dic[tks]),
                        np.std(attr_token_dic[tks]),
                        np.average(pred_token_dic[tks])
                    ))

    return results



def load_and_cache_examples(args, task, tokenizer, evaluate=False, show_stats=True):
    _set_barrier(args, evaluate=evaluate)

    processor = processors[task]()
    output_mode = output_modes[task]
    if args.task_name == "emotion":
        emotion_label = args.emotion
        label_list = processor.get_labels(emotion_label)
        
    else:
        label_list = processor.get_labels()


    if args.eval_dataset is not None:
        basename_eval_dataset = os.path.basename(args.eval_dataset).replace('.txt','')
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.data_dir, '../cached_{}={}_{}_{}'.format(
            'test_input' if evaluate else 'train',
            basename_eval_dataset,
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            ))
    else:
        # Load data features from cache or dataset file
        cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
            'test',
            list(filter(None, args.model_name_or_path.split('/'))).pop(),
            str(args.max_seq_length),
            str(task)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        if args.eval_dataset is not None:
            logger.info("[NOT NONE] Creating features from dataset file at %s", args.eval_dataset)
            examples = processor.get_test_input_examples(args.eval_dataset) if evaluate else processor.get_train_examples(args.data_dir)
        else:
            logger.info("Creating features from dataset file at %s", args.data_dir)
            
            examples = processor.get_test_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)

        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list = label_list,
            max_length = args.max_seq_length,
            output_mode = output_mode,
            pad_on_left=bool(args.model_type in ['xlnet']),
            pad_token = tokenizer.pad_token_id,
            pad_token_segment_id=tokenizer.pad_token_type_id)

        if args.my_data:
            cached_features_file = "../output_data/" + args.data_dir +"/cached"
            
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    
    if show_stats and args.eval_dataset is None:
        nonzero_cnt = 0
        for feature in features:
            nonzero_cnt += (np.count_nonzero(feature.input_ids) - 2)

        # Vocab size
        vocab_dict = defaultdict(int)
        for feature in features:
            for f in feature.input_ids:
                vocab_dict[f] += 1
        

        # feature distribution
        feature_dict = defaultdict(int)
        if output_mode == 'classification':
            for feature in features:
                feature_dict[label_list[feature.label_id]] += 1
            total_c = sum([c for _,c in feature_dict.items()])

        stats = {
            'num_features': len(features),
            'avg_sent_len': nonzero_cnt / len(features),
            'vocab_size': len(vocab_dict),
            'feature_dist': feature_dict}

        for key in stats.keys():
            logger.info(" {} = {}".format(key, highlight(stats[key])))


    _set_barrier_v2(args, evaluate=evaluate)


    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_ref_ids = torch.tensor([f.ref_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)


    if args.eval_dataset is None:
        if output_mode == "classification":
            all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    else:
        all_label_ids = None

    if all_label_ids is None:
        dataset = TensorDataset(all_input_ids, all_ref_ids, all_attention_mask, all_token_type_ids)

    else:        
        dataset = TensorDataset(all_input_ids, all_ref_ids, all_attention_mask, all_token_type_ids, all_label_ids)
    return dataset


def main(args):
    
    if args.task_name != "emotion":
        args.data_dir = args.data_dir.split('_')[0]
    # e.g., bert-base-uncased -> bert
    args.model_type = args.model_type.lower()

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    if args.task_name == "emotion":
        emotion_label = args.emotion
        label_list = processor.get_labels(emotion_label)
        
    else:
        label_list = processor.get_labels()
    num_labels = len(label_list)
    logger.info("Processor: {}, label: {} ({})".format(processor,label_list,num_labels))


    # config, tokenizer, model
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    _set_barrier(args)
    model.to(args.device)
    _set_fp16(args)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)

        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            # model = model_class.from_pretrained(checkpoint)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer,  prefix=global_step)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    args = _parse_args()
    pprint(args.__dict__)
    main(args)
