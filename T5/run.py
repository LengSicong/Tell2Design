# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# Copyright 2020 The HuggingFace Team. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Uses some code from
# https://github.com/huggingface/transformers/blob/master/examples/seq2seq/finetune_trainer.py


import argparse
import configparser
import itertools
import json
import logging
import os
from collections import defaultdict
import torch
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, EncoderDecoderModel, BertConfig, EncoderDecoderConfig, Trainer
import transformers

from arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from datasets import load_dataset
from evaluate import evaluate, get_avg_results, print_results
from utils import get_episode_indices
# from transformers_src import T5ForConditionalGeneration,Trainer, T5Config
from transformers_src import T5ForConditionalGeneration,T5Config, default_data_collator


os.environ["TOKENIZERS_PARALLELISM"] = "false"
def main():
    assert torch.cuda.is_available(), 'CUDA not available'
    
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('job')
    parser.add_argument('-c', '--config_file', type=str, default='config.ini', help='configuration file')
    parser.add_argument('-e', '--eval', action='store_true', default=False, help='run evaluation only')
    parser.add_argument('--evaluate_checkpoints', action='store_true', default=False,
                        help='evaluate intermediate checkpoints instead of the final model')
    parser.add_argument('--evaluate_last_checkpoint', action='store_true', default=False,
                        help='evaluate the last intermediate checkpoint instead of the final model')
    parser.add_argument('--evaluate_checkpoint_in_dir', type=str, default=None,
                        help='evaluate the checkpoint in the given directory')
    parser.add_argument('-a', '--evaluate_all', action='store_true', default=False,
                        help='evaluate intermediate checkpoints together with the final model')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='which GPU to use for evaluation')
    parser.add_argument('-v', '--verbose_results', action='store_true', default=False,
                        help='print results for each evaluation run')
    args, remaining_args = parser.parse_known_args()

    # read config file
    config = configparser.ConfigParser(allow_no_value=False)
    config.read(args.config_file)
    job = args.job
    assert job in config

    # set defaults for other arguments
    defaults = {
        'overwrite_output_dir': True,
        'overwrite_cache': True,
        'per_device_eval_batch_size': 4,
        'learning_rate': 5e-4,
        'logging_steps': 1,     # do not log by default = 'logging_steps': 0
        'save_steps': 0,        # do not save checkpoints by default
    }

    # the config file gives default values for the command line arguments
    defaults.update(dict(config.items(job)))
    for key in defaults:
        if defaults[key] in ['True', 'False']:
            # interpret True/False as boolean
            defaults[key] = config.getboolean(job, key)
        if defaults[key] == 'None':
            # interpret as None
            defaults[key] = None

    if args.eval:
        # run evaluation only
        defaults['do_train'] = False

    # parse remaining arguments and divide them into three categories
    second_parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    second_parser.set_defaults(**defaults)
    model_args, data_args, training_args = second_parser.parse_args_into_dataclasses(remaining_args)

    try:
        os.mkdir(training_args.output_dir)
    except FileExistsError:
        pass

    # process arguments related to max length
    if data_args.max_output_seq_length_eval is None:
        # defaults first to max_output_seq_length, then max_seq_length_eval, then max_seq_length
        data_args.max_output_seq_length_eval = data_args.max_output_seq_length \
                                               or data_args.max_seq_length_eval \
                                               or data_args.max_seq_length

    if data_args.max_output_seq_length is None:
        # defaults to max_seq_length
        data_args.max_output_seq_length = data_args.max_seq_length

    if data_args.max_seq_length_eval is None:
        # defaults to max_seq_length
        data_args.max_seq_length_eval = data_args.max_seq_length

    if data_args.chunk_size_eval is None:
        # defaults to chunk_size
        data_args.chunk_size_eval = data_args.chunk_size

    if data_args.chunk_overlap_eval is None:
        # defaults to chunk overlap
        data_args.chunk_overlap_eval = data_args.chunk_overlap

    # construct name for the output directory
    # for example: conll04-t5-base-ep200-len256-ratio0-b4-train
    if data_args.exp != None:
        output_dir = os.path.join(
            training_args.output_dir,
            f'{args.job}'
            f'-{model_args.model_name_or_path.split("/")[-1]}'
            f'-{data_args.exp}'
            f'-ep{round(training_args.num_train_epochs)}'
            f'-len{data_args.max_seq_length}'
        )
    else:
        output_dir = os.path.join(
            training_args.output_dir,
            f'{args.job}'
            f'-{model_args.model_name_or_path.split("/")[-1]}'
            f'-ep{round(training_args.num_train_epochs)}'
            f'-len{data_args.max_seq_length}'
        )

    if data_args.max_output_seq_length != data_args.max_seq_length:
        output_dir += f'-{data_args.max_output_seq_length}'

    if training_args.learning_rate != 5e-4:
        output_dir += f'-lr{training_args.learning_rate}'

    output_dir += f'-b{training_args.per_device_train_batch_size}' \
                  f'-{data_args.train_split}'

    if data_args.chunk_size != 128:
        output_dir += f'-chunk{data_args.chunk_size}'
    if data_args.chunk_overlap != 64:
        output_dir += f'-overlap{data_args.chunk_overlap}'

    if data_args.output_format is not None:
        output_dir += f'-{data_args.output_format}'
    if data_args.input_format is not None:
        output_dir += f'-{data_args.input_format}'
    if data_args.train_subset < 1:
        output_dir += f'-size{data_args.train_subset:.2f}'
    if data_args.output_format_type is not None:
        output_dir += f'-{data_args.output_format_type}'
    if data_args.comment is not None:
        output_dir += f'-{data_args.comment}'

    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass

    # setup logging
    logging.basicConfig(
      filename=os.path.join(output_dir, 'logs.log'),
      format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
      datefmt='%Y-%m-%d %H:%M:%S',
      level=logging.INFO,
    )
    logging.getLogger().addHandler(logging.StreamHandler())

    # construct file name for the evaluation results
    evaluation_output_filename = f'results'
    if data_args.num_beams is not None:
        evaluation_output_filename += f'-{data_args.num_beams}beams'
    if data_args.max_seq_length_eval is not None:
        evaluation_output_filename += f'-len{data_args.max_seq_length_eval}'

    # create model config
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    # create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    )

    # get list of dataset names
    dataset_names = data_args.datasets.split(',')

    # construct list of episode indices
    episode_indices = get_episode_indices(data_args.episodes)

    # episode loop
    # (note that the episode index is used as the random seed, so that each episode is reproducible)
    evaluation_results = defaultdict(list)
    for ep_idx in episode_indices:
        print()
        logging.info(f'Episode {ep_idx} ({len(episode_indices)} episodes total)')
        episode_output_dir = os.path.join(output_dir, f'episode{ep_idx}')

        try:
            os.mkdir(episode_output_dir)
        except FileExistsError:
            pass

        logging.info(f'Output directory: {episode_output_dir}')

        training_args.output_dir = episode_output_dir   # checkpoints are saved in episode-specific directory

        # load pretrained model
        model = None
        if training_args.zero_shot or training_args.do_train:
            logging.info(f"Using model {model_args.model_name_or_path}")
            if data_args.exp == 'scratch': # train T5 from scratch
                if model_args.model_name_or_path.startswith('t5'):
                    model = transformers.T5ForConditionalGeneration(config)
                elif model_args.model_name_or_path.startswith('bert'):
                    config_encoder = BertConfig()
                    config_decoder = BertConfig()
                    config = EncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
                    # Initializing a Bert2Bert model from the bert-base-uncased style configurations
                    model = EncoderDecoderModel(config=config)
                    model.config.decoder_start_token_id = tokenizer.cls_token_id
                    model.config.pad_token_id = tokenizer.pad_token_id
                    model.config.vocab_size = model.config.decoder.vocab_size
            else: # load pretrained weights
                if data_args.boundary_in_where == "Encoder":
                    # if pretrained model is bert2bert encoder-decoder structure
                    if model_args.model_name_or_path.startswith('bert'):
                        model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
                        model.config.decoder_start_token_id = tokenizer.cls_token_id
                        model.config.pad_token_id = tokenizer.pad_token_id
                        model.config.vocab_size = model.config.decoder.vocab_size
                    if data_args.exp.endswith('finetune'):
                        config = T5Config.from_pretrained("t5-base")
                        model = T5ForConditionalGeneration.from_pretrained("/home/sicong/min_max_Floorplan_Generation_Baseline/experiments/floorplan-t5-base-no_boundary-ep20-len512-b12-train-original-baseline/episode0/pytorch_model.bin",config=config)
                        pass
                    else:
                        model = T5ForConditionalGeneration.from_pretrained("t5-base")
                        # model = AutoModelForSeq2SeqLM.from_pretrained(
                        #     model_args.model_name_or_path,
                        #     config=config,
                        #     cache_dir=model_args.cache_dir,
                        # )
                    
                # model = AutoModelForSeq2SeqLM.from_config(
                #     config=config
                elif data_args.boundary_in_where == "Decoder":
                    model = T5ForConditionalGeneration.from_pretrained("t5-base")
                else:
                    raise Exception("pleae indicate where to add boundary information in the argument: boundary_in_where (Encoder or Decoder)")

        # fine-tune the model
        if training_args.do_train:
            # load train dataset
            datasets = []
            for dataset_name in dataset_names:
                logging.info(f'Process dataset {dataset_name} (train)')
                dataset = load_dataset(
                    dataset_name, data_args, split=data_args.train_split,
                    max_input_length=data_args.max_seq_length, max_output_length=data_args.max_output_seq_length,
                    tokenizer=tokenizer, seed=ep_idx, train_subset=data_args.train_subset,
                )
                datasets.append(dataset)

            train_dataset = torch.utils.data.ConcatDataset(datasets) if training_args.do_train else None

            # construct trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                data_collator = default_data_collator
            )

            # start trainer
            logging.info('Start training')
            # trainer.train(
            #     model_path=model_args.model_name_or_path
            # )
            trainer.train()

            # save model parameters
            trainer.save_model(episode_output_dir)
        
        # run evaluation
        if training_args.local_rank in [-1, 0] and (training_args.do_eval or training_args.do_predict):
            # should we evaluate on dev, test, or both?
            evaluation_splits = []
            if training_args.do_eval:
                evaluation_splits.append('dev')
            if training_args.do_predict:
                evaluation_splits.append('test')

            # should we evaluate on the final model and/or on all intermediate checkpoints?
            evaluation_dirs = []

            if args.evaluate_checkpoints or args.evaluate_last_checkpoint or \
                    args.evaluate_checkpoint_in_dir or args.evaluate_all:
                # all intermediate checkpoints
                evaluation_dirs = list(sorted([
                    checkpoint_dir
                    for checkpoint_dir in os.listdir(episode_output_dir)
                    if checkpoint_dir.startswith('checkpoint-')
                ], key=lambda x: int(x[len('checkpoint-'):])))
                if args.evaluate_last_checkpoint:
                    # only evaluate on the last checkpoint
                    evaluation_dirs = [evaluation_dirs[-1]]
                elif args.evaluate_checkpoint_in_dir:
                    assert args.evaluate_checkpoint_in_dir in evaluation_dirs, \
                        "checkpoint {} does not exist".format(args.evaluate_checkpoint_in_dir)
                    evaluation_dirs = [args.evaluate_checkpoint_in_dir]

            if args.evaluate_all or (not args.evaluate_checkpoints and not args.evaluate_last_checkpoint):
                # evaluate on the final model
                evaluation_dirs += ['']

            # datasets to evaluate on
            if data_args.eval_datasets is None:
                eval_dataset_names = dataset_names
            else:
                eval_dataset_names = data_args.eval_datasets.split(',')

            # evaluate all possible combinations of dev/test, model, and datasets
            for comb in itertools.product(evaluation_splits, evaluation_dirs, eval_dataset_names):
                split, evaluation_dir, dataset_name = comb
                model_dir = os.path.join(episode_output_dir, evaluation_dir)

                if args.evaluate_checkpoints or args.evaluate_last_checkpoint or args.evaluate_all or model is None:
                    # we need to load the model
                    if data_args.boundary_in_where == "Encoder":
                        if model_args.model_name_or_path.startswith('bert'):
                            model = EncoderDecoderModel.from_pretrained(model_dir)
                            model.config.decoder_start_token_id = tokenizer.cls_token_id
                            model.config.pad_token_id = tokenizer.pad_token_id
                            model.config.vocab_size = model.config.decoder.vocab_size
                        else:
                        #     model = AutoModelForSeq2SeqLM.from_pretrained(
                        #         't5-base',
                        #         config=config,
                        # )   
                            # model = T5ForConditionalGeneration.from_pretrained(
                            #         't5-base',
                            #         config=config,
                            # )
                            t5_config = T5Config.from_pretrained("t5-base")
                            model = T5ForConditionalGeneration(t5_config)
                            # model.load_state_dict(model_dir)
                            checkpoint = torch.load(os.path.join(model_dir,'pytorch_model.bin'))
                            model.load_state_dict(checkpoint)
                            # model = model.from_pretrained(pretrained_model_name_or_path=model_dir, config='t5-base')
                    elif data_args.boundary_in_where == "Decoder":
                        # model = T5ForConditionalGeneration.from_pretrained(
                        #     model_dir,
                        #     config=config,
                        # )
                        model = T5ForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=model_dir, config='t5-base')
                    else:
                        raise Exception("pleae indicate where to add boundary information in the argument: boundary_in_where (Encoder or Decoder)")

                if len(evaluation_dir) > 0:
                    logging.info(f'Evaluate {evaluation_dir} on {dataset_name} {split}')
                else:
                    logging.info(f'Evaluate on {dataset_name} {split}')

                res = evaluate(
                    model=model, dataset_name=dataset_name, data_args=data_args, tokenizer=tokenizer, split=split,
                    seed=ep_idx, batch_size=training_args.per_device_eval_batch_size, gpu=args.gpu, output_dir=model_dir
                )
                # store results
                evaluation_results[comb].append(res)

                # print results
                if args.verbose_results:
                    print_results(res)

                # save results to file
                with open(
                        os.path.join(model_dir, evaluation_output_filename + f'-{dataset_name}-{split}.json'), 'w'
                ) as f:
                    json.dump(res, f, indent=0)

    # # print average results and save them to file
    # for comb, results in evaluation_results.items():
    #     split, evaluation_dir, dataset_name = comb

    #     print()
    #     logging.info(
    #         f'Average of {split} results over {len(results)} episodes ({dataset_name} {evaluation_dir}):'
    #     )
    #     res = get_avg_results(results)

    #     # print average results
    #     print_results(res)

    #     # save average results to file
    #     filename = evaluation_output_filename + f'-{dataset_name}-{split}'
    #     if len(evaluation_dir) > 0:
    #         filename += '-'
    #     filename += f'{evaluation_dir}.json'

    #     with open(os.path.join(output_dir, filename), 'w') as f:
    #         json.dump(res, f, indent=0)

    print()
    logging.info(f'Model weights and intermediate checkpoints saved in {output_dir}')


if __name__ == "__main__":
    main()
