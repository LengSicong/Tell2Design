# Tell2Design: A Dataset for Language-Guided Floor Plan Generation

Code for the paper "[Tell2Design: A Dataset for Language-Guided Floor Plan Generation](http)" (ACL 2023)

If you use this code or dataset, please cite the paper using the bibtex reference below.
```
@inproceedings{Tell2Design,
    title={Tell2Design: A Dataset for Language-Guided Floor Plan Generation},
    author={Sicong Leng, Yang Zhou, Mohammed Haroon Dupty, Wee Sun Lee, Sam Conrad Joyce, and Wei Lu},
    booktitle={The 61st Annual Meeting of the Association for Computational Linguistics, {ACL} 2023},
    year={2023},
}
```


## Requirements

- Python 3.7.6
- PyTorch (tested with version 1.10.0)
- Transformers (tested with version 4.14.1)

You can install all required Python packages with `pip install -r requirements.txt`


## Datasets

The dataset collected in this paper is stored in [Google Drive](https://drive.google.com/file/d/1ebPneckeDR88YMjGb2t1iguCCKHl8kGB/view?usp=sharing). Please download the zip file from the cloud and extract to your local machine.

### Data Structure
- General Data/
    - floorplan_image/ *(all floor plan RGB images)*
    - human_annotated_tags/ *(all human-annotated floor plan descriptions)*
    - Tell2Design_artificial_all.pkl *(all artificial-generated floor plan descriptions)*
- Separated Data/ *(data separated w.r.t. different training stages for all baselines)*
    - 1st_finetune_data/
    - 2nd_finetune_data/
    - eval_data/


## Running the Seq2Seq Baseline

Use the following command:
`python run.py floorplan`

The `floorplan` argument refers to a section of the config file, which by default is `config.ini`. A [sample config file](config.ini) is provided.

For example, to replicate the paper's results, have the following section in the config file:
```
[floorplan]
datasets = floorplan
model_name_or_path = t5-base
num_train_epochs = 20
max_seq_length = 512
max_seq_length_eval = 512
per_device_train_batch_size = 4
per_device_eval_batch_size = 4
do_train = True
do_eval = True
do_predict = False
boundary_in_where = Encoder
exp = full
editing_data = False
output_format_type = original
comment = baseline
adapter = None
```
Then run `python run.py floorplan`.

Config arguments can be overwritten by command line arguments.
For example: `python run.py floorplan --num_train_epochs 50`.


### Additional details

If `do_train = True`, the model is trained on the given train split (e.g., `'train'`) of the given datasets.
The final weights and intermediate checkpoints are written in a directory such as `experiments/floorplan-t5-base-full-ep20-len512-b12-train-original-baseline`, with one subdirectory per episode.
Results in JSON format are also going to be saved there.

In every episode, the model is trained on a different (random) permutation of the training set.
The random seed is given by the episode number, so that every episode always produces the same exact model.

Once a model is trained, it is possible to evaluate it without training again.
For this, set `do_train = False` or (more easily) provide the `-e` command-line argument: `python run.py floorplan -e`.

If `do_eval = True`, the model is evaluated on the `'dev'` split.
If `do_predict = True`, the model is evaluated on the `'test'` split.


### Arguments

The following are the most important command-line arguments for the `run.py` script.
Run `python run.py -h` for the full list.

- `-c CONFIG_FILE`: specify config file to use (default is `config.ini`)
- `-e`: only run evaluation (overwrites the setting `do_train` in the config file)
- `-a`: evaluate also intermediate checkpoints, in addition to the final model
- `-v` : print results for each evaluation run
- `-g GPU`: specify which GPU to use for evaluation

The following are the most important arguments for the config file. 
See the [sample config file](config.ini) to understand the format.

- `model_name_or_path` (str): path to pretrained model or model identifier from [huggingface.co/models](https://huggingface.co/models) (e.g. `t5-base`)
- `do_train` (bool): whether to run training (default is False)
- `do_eval` (bool): whether to run evaluation on the `dev` set (default is False)
- `do_predict` (bool): whether to run evaluation on the `test` set (default is False)
- `num_train_epochs` (int): number of train epochs
- `learning_rate` (float): initial learning rate (default is 5e-4)
- `per_device_train_batch_size` (int): batch size per GPU during training (default is 8)
- `per_device_eval_batch_size` (int): batch size during evaluation (default is 8; only one GPU is used for evaluation)
- `max_seq_length` (int): maximum input sequence length after tokenization; longer sequences are truncated
- `max_output_seq_length` (int): maximum output sequence length (default is `max_seq_length`)
- `max_seq_length_eval` (int): maximum input sequence length for evaluation (default is `max_seq_length`)
- `max_output_seq_length_eval` (int): maximum output sequence length for evaluation (default is `max_output_seq_length` or `max_seq_length_eval` or `max_seq_length`)
- `episodes` (str): episodes to run (default is `0`; an interval can be specified, such as `1-4`; the episode number is used as the random seed)

See [arguments.py](./T5/arguments.py) and [transformers.TrainingArguments](https://github.com/huggingface/transformers/blob/master/src/transformers/training_args.py) for additional config arguments.


## Licenses

The code of this repository is released under the [Apache 2.0 license](LICENSE).
The dataset of this repository is released uder the [CC BY-NC 4.0 license](https://creativecommons.org/licenses/by-nc/4.0/deed.en_GB).
