# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import os
import logging
import random
from typing import Dict, Generator, Tuple, List
from abc import ABC, abstractmethod
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer, torch_distributed_zero_first
from transformers_src import default_data_collator

from arguments import DataTrainingArguments
from input_example import InputFeatures, InputExample
from input_formats import INPUT_FORMATS
from output_formats import OUTPUT_FORMATS, key2ind, ind2key

key2ind = {'living room': 1, 'master room': 2, 'kitchen': 3, 'bathroom 1': 4, 'bathroom 2': 5, 'bathroom 3': 6, 'dining room': 7, 'common room 1': 8, 'common room 2': 9, 'common room 3': 10, 'common room 4': 11, 'balcony 1': 12, 'balcony 2': 13, 'balcony 3': 14, 'entrance': 15, 'storage': 16, '0': 17, '1': 18, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26, '10': 27, '11': 28, '12': 29, '13': 30, '14': 31, '15': 32, '16': 33, '17': 34, '18': 35, '19': 36, '20': 37, '21': 38, '22': 39, '23': 40, '24': 41, '25': 42, '26': 43, '27': 44, '28': 45, '29': 46, '30': 47, '31': 48, '32': 49, '33': 50, '34': 51, '35': 52, '36': 53, '37': 54, '38': 55, '39': 56, '40': 57, '41': 58, '42': 59, '43': 60, '44': 61, '45': 62, '46': 63, '47': 64, '48': 65, '49': 66, '50': 67, '51': 68, '52': 69, '53': 70, '54': 71, '55': 72, '56': 73, '57': 74, '58': 75, '59': 76, '60': 77, '61': 78, '62': 79, '63': 80, '64': 81, '65': 82, '66': 83, '67': 84, '68': 85, '69': 86, '70': 87, '71': 88, '72': 89, '73': 90, '74': 91, '75': 92, '76': 93, '77': 94, '78': 95, '79': 96, '80': 97, '81': 98, '82': 99, '83': 100, '84': 101, '85': 102, '86': 103, '87': 104, '88': 105, '89': 106, '90': 107, '91': 108, '92': 109, '93': 110, '94': 111, '95': 112, '96': 113, '97': 114, '98': 115, '99': 116, '100': 117, '101': 118, '102': 119, '103': 120, '104': 121, '105': 122, '106': 123, '107': 124, '108': 125, '109': 126, '110': 127, '111': 128, '112': 129, '113': 130, '114': 131, '115': 132, '116': 133, '117': 134, '118': 135, '119': 136, '120': 137, '121': 138, '122': 139, '123': 140, '124': 141, '125': 142, '126': 143, '127': 144, '128': 145, '129': 146, '130': 147, '131': 148, '132': 149, '133': 150, '134': 151, '135': 152, '136': 153, '137': 154, '138': 155, '139': 156, '140': 157, '141': 158, '142': 159, '143': 160, '144': 161, '145': 162, '146': 163, '147': 164, '148': 165, '149': 166, '150': 167, '151': 168, '152': 169, '153': 170, '154': 171, '155': 172, '156': 173, '157': 174, '158': 175, '159': 176, '160': 177, '161': 178, '162': 179, '163': 180, '164': 181, '165': 182, '166': 183, '167': 184, '168': 185, '169': 186, '170': 187, '171': 188, '172': 189, '173': 190, '174': 191, '175': 192, '176': 193, '177': 194, '178': 195, '179': 196, '180': 197, '181': 198, '182': 199, '183': 200, '184': 201, '185': 202, '186': 203, '187': 204, '188': 205, '189': 206, '190': 207, '191': 208, '192': 209, '193': 210, '194': 211, '195': 212, '196': 213, '197': 214, '198': 215, '199': 216, '200': 217, '201': 218, '202': 219, '203': 220, '204': 221, '205': 222, '206': 223, '207': 224, '208': 225, '209': 226, '210': 227, '211': 228, '212': 229, '213': 230, '214': 231, '215': 232, '216': 233, '217': 234, '218': 235, '219': 236, '220': 237, '221': 238, '222': 239, '223': 240, '224': 241, '225': 242, '226': 243, '227': 244, '228': 245, '229': 246, '230': 247, '231': 248, '232': 249, '233': 250, '234': 251, '235': 252, '236': 253, '237': 254, '238': 255, '239': 256, '240': 257, '241': 258, '242': 259, '243': 260, '244': 261, '245': 262, '246': 263, '247': 264, '248': 265, '249': 266, '250': 267, '251': 268, '252': 269, '253': 270, '254': 271, '255': 272}
ind2key = {1: 'living room', 2: 'master room', 3: 'kitchen', 4: 'bathroom 1', 5: 'bathroom 2', 6: 'bathroom 3', 7: 'dining room', 8: 'common room 1', 9: 'common room 2', 10: 'common room 3', 11: 'common room 4', 12: 'balcony 1', 13: 'balcony 2', 14: 'balcony 3', 15: 'entrance', 16: 'storage', 17: '0', 18: '1', 19: '2', 20: '3', 21: '4', 22: '5', 23: '6', 24: '7', 25: '8', 26: '9', 27: '10', 28: '11', 29: '12', 30: '13', 31: '14', 32: '15', 33: '16', 34: '17', 35: '18', 36: '19', 37: '20', 38: '21', 39: '22', 40: '23', 41: '24', 42: '25', 43: '26', 44: '27', 45: '28', 46: '29', 47: '30', 48: '31', 49: '32', 50: '33', 51: '34', 52: '35', 53: '36', 54: '37', 55: '38', 56: '39', 57: '40', 58: '41', 59: '42', 60: '43', 61: '44', 62: '45', 63: '46', 64: '47', 65: '48', 66: '49', 67: '50', 68: '51', 69: '52', 70: '53', 71: '54', 72: '55', 73: '56', 74: '57', 75: '58', 76: '59', 77: '60', 78: '61', 79: '62', 80: '63', 81: '64', 82: '65', 83: '66', 84: '67', 85: '68', 86: '69', 87: '70', 88: '71', 89: '72', 90: '73', 91: '74', 92: '75', 93: '76', 94: '77', 95: '78', 96: '79', 97: '80', 98: '81', 99: '82', 100: '83', 101: '84', 102: '85', 103: '86', 104: '87', 105: '88', 106: '89', 107: '90', 108: '91', 109: '92', 110: '93', 111: '94', 112: '95', 113: '96', 114: '97', 115: '98', 116: '99', 117: '100', 118: '101', 119: '102', 120: '103', 121: '104', 122: '105', 123: '106', 124: '107', 125: '108', 126: '109', 127: '110', 128: '111', 129: '112', 130: '113', 131: '114', 132: '115', 133: '116', 134: '117', 135: '118', 136: '119', 137: '120', 138: '121', 139: '122', 140: '123', 141: '124', 142: '125', 143: '126', 144: '127', 145: '128', 146: '129', 147: '130', 148: '131', 149: '132', 150: '133', 151: '134', 152: '135', 153: '136', 154: '137', 155: '138', 156: '139', 157: '140', 158: '141', 159: '142', 160: '143', 161: '144', 162: '145', 163: '146', 164: '147', 165: '148', 166: '149', 167: '150', 168: '151', 169: '152', 170: '153', 171: '154', 172: '155', 173: '156', 174: '157', 175: '158', 176: '159', 177: '160', 178: '161', 179: '162', 180: '163', 181: '164', 182: '165', 183: '166', 184: '167', 185: '168', 186: '169', 187: '170', 188: '171', 189: '172', 190: '173', 191: '174', 192: '175', 193: '176', 194: '177', 195: '178', 196: '179', 197: '180', 198: '181', 199: '182', 200: '183', 201: '184', 202: '185', 203: '186', 204: '187', 205: '188', 206: '189', 207: '190', 208: '191', 209: '192', 210: '193', 211: '194', 212: '195', 213: '196', 214: '197', 215: '198', 216: '199', 217: '200', 218: '201', 219: '202', 220: '203', 221: '204', 222: '205', 223: '206', 224: '207', 225: '208', 226: '209', 227: '210', 228: '211', 229: '212', 230: '213', 231: '214', 232: '215', 233: '216', 234: '217', 235: '218', 236: '219', 237: '220', 238: '221', 239: '222', 240: '223', 241: '224', 242: '225', 243: '226', 244: '227', 245: '228', 246: '229', 247: '230', 248: '231', 249: '232', 250: '233', 251: '234', 252: '235', 253: '236', 254: '237', 255: '238', 256: '239', 257: '240', 258: '241', 259: '242', 260: '243', 261: '244', 262: '245', 263: '246', 264: '247', 265: '248', 266: '249', 267: '250', 268: '251', 269: '252', 270: '253', 271: '254', 272: '255'}

class BaseDataset(Dataset, ABC):
    """
    Base class for all datasets.
    """
    name = None         # name of the dataset
    data_name = None    # name of the directory, if different from the name of the dataset
    task_descriptor = None  # string to prepend to every input sentence if multitask=True (default is self.name)

    default_input_format = 'plain'
    default_output_format = None
    default_data_dir = 'data'

    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            max_input_length: int,
            max_output_length: int,
            overwrite_cache: bool = False,
            mode: str = 'train',
            local_rank: int = -1,
            train_subset: float = 1,  # a number < 1 is to use only a subset of training data (random)
            seed: int = None,
            shuffle: bool = True,
            data_args: DataTrainingArguments = None,
            is_eval: bool = False,
    ):
        if seed is not None:
            # set random seed for repeatability
            random.seed(seed)

        self.data_args = data_args
        self.tokenizer = tokenizer

        self.max_input_length = max_input_length
        self.max_output_length = max_output_length

        self.input_format = INPUT_FORMATS[
            data_args.input_format if data_args.input_format is not None else self.default_input_format
        ]()
        self.output_format = OUTPUT_FORMATS[
            data_args.output_format if data_args.output_format is not None else self.default_output_format
        ]()

        self.data_path = data_args.data_dir if data_args.data_dir is not None else self.default_data_dir

        self.is_eval = is_eval
        self.eval_nll = data_args.eval_nll

        cached_data_file = os.path.join(
            self.data_dir(),
            f"cached_{self.name}_{mode}_{tokenizer.__class__.__name__}_{max_input_length}_{max_output_length}_{self.data_args.exp}"
            f"{'_multitask' if data_args.multitask else ''}.pth"
        )

        with torch_distributed_zero_first(local_rank):
            # make sure only the first process in distributed training processes the dataset,
            # and the others can use the cached version

            if os.path.exists(cached_data_file) and not overwrite_cache:
                self.load_cached_data(cached_data_file)

            else:
                self.load_schema()   # here the dataset can load information such as entity/relation types
                self.examples = self.load_data(mode=mode, seed=seed)

                # assign examples to this dataset
                for example in self.examples:
                    example.dataset = self

                self.features = self.compute_features(
                    max_input_length=max_input_length,
                    max_output_length=max_output_length,
                    multitask=data_args.multitask,
                )

                if local_rank in [-1, 0]:
                    # save data
                    self.save_data(cached_data_file)

            # shuffle indices
            self.indices = list(range(len(self.examples)))
            if seed is not None and shuffle:
                random.shuffle(self.indices)

            # compute effective size of the dataset
            self.effective_size = round(train_subset * len(self.examples))
            if train_subset != 1:
                logging.info(f"Effective dataset size reduced to {self.effective_size} ({train_subset * 100:.0f}%)")

    def __repr__(self):
        return f'Dataset {self.name}'

    def __len__(self):
        return self.effective_size

    def __getitem__(self, i: int) -> InputFeatures:
        return self.features[self.indices[i]]

    def get_example(self, i: int) -> InputExample:
        return self.examples[self.indices[i]]

    def data_dir(self):
        if self.data_name is not None:
            return os.path.join(self.data_path, self.data_name)
        else:
            return os.path.join(self.data_path, self.name)

    def load_cached_data(self, cached_data_file: str):
        d = torch.load(cached_data_file)
        self.examples, self.features = d['examples'], d['features']

    def save_data(self, cached_data_file: str):
        torch.save({
            'examples': self.examples,
            'features': self.features,
        }, cached_data_file)

    def load_schema(self):
        """
        Load extra dataset information, such as entity/relation types.
        """
        pass

    @abstractmethod
    def load_data_single_split(self, split: str, seed: int = None) -> List[InputExample]:
        """
        Load data for a single split (train, dev, or test).
        """
        pass

    def load_data(self, mode: str, seed: int = None) -> List[InputExample]:
        """
        Load all data, where 'mode' is a list of comma-separated splits to use.
        """
        examples = []

        if isinstance(mode, str):
            splits = mode.split(',')
        else:
            assert isinstance(mode, (list, tuple))
            splits = mode

        for split in splits:
            examples += self.load_data_single_split(split, seed=seed)

        return examples

    def _warn_max_sequence_length(self, max_sequence_length: int, sentences: List[str], name: str):
        max_length_needed = max(len(self.tokenizer.tokenize(x)) for x in sentences)
        if max_length_needed > max_sequence_length:
            logging.warning(
                f'Max sequence length is {max_sequence_length} but the longest {name} sequence is '
                f'{max_length_needed} long'
            )

    def batch_encode_output_(self, output_index, max_output_length):
        input_ids = []
        attention_mask = []
        for ins in output_index:
            input_ids.append(ins+(max_output_length-len(ins))*[0])
            attention_mask.append(len(ins)*[1]+(max_output_length-len(ins))*[0])
        input_ids = torch.tensor(input_ids, dtype=torch.int64)
        attention_mask = torch.tensor(attention_mask, dtype=torch.int64)
        output_tok = {'input_ids':input_ids,'attention_mask':attention_mask}
        return output_tok

    def compute_features(self, max_input_length: int, max_output_length: int, multitask: bool = False):
        input_sentences = [self.input_format.format_input(example, multitask=multitask) for example in self.examples]
        if self.data_args.output_format_type == 'short-relation':
            output_sentences = [self.output_format.format_short_output_with_relation(example) for example in self.examples]
        elif self.data_args.output_format_type == 'short':
            output_sentences = [self.output_format.format_short_output(example) for example in self.examples]
        elif self.data_args.output_format_type == 'long':
            output_sentences = [self.output_format.format_long_output(example) for example in self.examples]
        elif self.data_args.output_format_type == 'original':
            output_sentences = [self.output_format.format_short_output_(example) for example in self.examples]
        boundary_sentences = [' '.join(example.boundary_tokens) for example in self.examples]
        # TODO: Sicong if wanna add boundary sequence in encoding process, can directly add boundary sentence to input sentence here
        if self.data_args.boundary_in_where == 'Encoder':
            if self.data_args.exp.startswith('no_boundary'):
                pass
            else:
                # print("Boundary information is added to the end of the input sequence and used in Encoder!")
                logging.info("Boundary information is added to the end of the input sequence and used in Encoder.")
                input_sentences = [( (self.input_format.format_input(example, multitask=multitask))+' '.join(example.boundary_tokens) ) for example in self.examples]
                # input_sentences = [( ' '.join(example.boundary_tokens) ) + (self.input_format.format_input(example, multitask=multitask)) for example in self.examples] # reverse description and boundary token orders

        logging.info(f'Example input sententece: {input_sentences[0]}')
        logging.info(f'Example output sententece: {output_sentences[0]}')

        num_rooms = [len(example.rooms) for example in self.examples]
        regr_labels = []
        for example in self.examples:
            regr_label = []
            for room in example.rooms:
                regr_label.extend([room.x, room.y, room.h, room.w])
            regr_labels.append(regr_label)
        # check sanity of regression labels
        for i in range(len(num_rooms)):
            assert num_rooms[i] == len(regr_labels[i])/4

        input_tok = self.tokenizer.batch_encode_plus(
            input_sentences,
            max_length=max_input_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_input_length, input_sentences, "input")

        # output_index = [self.output_format.format_output_index(example) for example in self.examples]
        # output_tok = self.batch_encode_output_(output_index, max_output_length)

        output_tok = self.tokenizer.batch_encode_plus(
            output_sentences,
            max_length=max_output_length,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )
        self._warn_max_sequence_length(max_output_length, output_sentences, "output")

        boundary_tok = self.tokenizer.batch_encode_plus(
            boundary_sentences,
            max_length=50,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
        )

        assert input_tok.input_ids.size(0) == output_tok['input_ids'].size(0)
        assert input_tok.input_ids.size(0) == boundary_tok.input_ids.size(0)
    
        features = []
        if self.data_args.boundary_in_where == 'Encoder':
            for sentence_input_ids, att_mask, label_input_ids, num_room, regr_l in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                    output_tok['input_ids'], num_rooms, regr_labels):
                features.append(InputFeatures(
                    input_ids=sentence_input_ids.tolist(),
                    attention_mask=att_mask.tolist(),
                    label_ids=label_input_ids.tolist(),
                    num_rooms=num_room,
                    regr_labels=regr_l
                ))
            # for sentence_input_ids, att_mask, label_input_ids, num_room, regr_l, decoder_attention_mask in zip(input_tok.input_ids, input_tok.attention_mask,
            #                                                         output_tok['input_ids'], num_rooms, regr_labels, output_tok['attention_mask']):
            #     features.append(InputFeatures(
            #         input_ids=sentence_input_ids.tolist(),
            #         attention_mask=att_mask.tolist(),
            #         label_ids=label_input_ids.tolist(),
            #         num_rooms=num_room,
            #         regr_labels=regr_l,
            #         decoder_attention_mask = decoder_attention_mask.tolist()
            #     ))
        
        else: # boundary_in_where == 'Decoder"
            for sentence_input_ids, att_mask, label_input_ids, boundary_input_ids, boundary_tok_mask in zip(input_tok.input_ids, input_tok.attention_mask,
                                                                    output_tok.input_ids, boundary_tok.input_ids, boundary_tok.attention_mask):
                features.append(InputFeatures(
                    input_ids=sentence_input_ids.tolist(),
                    attention_mask=att_mask.tolist(),
                    boundary_ids=boundary_input_ids.tolist(),
                    boundary_mask=boundary_tok_mask.tolist(),
                    label_ids=label_input_ids.tolist()
                ))
    
        return features

    def decode_new(self, prediction):
        prediction = prediction.tolist()
        string = ""
        for pre in prediction:
            if pre == 0:
                pass
            else:
                string += f'{ind2key[pre]} '
        return string

    def generate_output_sentences(self, data_args: DataTrainingArguments, model, device, batch_size: int, features) \
            -> Generator[Tuple[InputExample, str], None, None]:
        """
        Generate pairs (example, output_sentence) for evaluation.
        """
        test_data_loader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=default_data_collator,
        )
        total = len(test_data_loader)
        for i, inputs in tqdm(enumerate(test_data_loader), total=len(test_data_loader)):
            if data_args.boundary_in_where == 'Encoder':
                predictions = model.generate(
                    inputs['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams
                )
            elif data_args.boundary_in_where == 'Decoder':
                predictions = model.generate(
                    inputs['input_ids'].to(device),
                    max_length=data_args.max_output_seq_length_eval,
                    num_beams=data_args.num_beams, 
                    features=inputs['boundary_ids']
                )

            for j, (input_ids, label_ids, prediction) in enumerate(
                    zip(inputs['input_ids'], inputs['labels'], predictions)):
                if data_args.boundary_in_where == 'Encoder':
                    current_id = i * batch_size + j
                    example = self.get_example(current_id)
                    # output_sentence = self.decode_new(prediction)
                    # pre_list = prediction.tolist()
                    output_sentence = self.tokenizer.decode(prediction, skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)
                elif data_args.boundary_in_where == 'Decoder':
                    current_id = i * batch_size + j
                    example = self.get_example(current_id)
                    output_sentence = self.tokenizer.decode(prediction[51:], skip_special_tokens=True,
                                                            clean_up_tokenization_spaces=False)

                yield example, output_sentence, None

    @abstractmethod
    def evaluate_dataset(self, data_args: DataTrainingArguments, model, device, batch_size: int, macro: bool = False) \
            -> Dict[str, float]:
        """
        Evaluate model on this dataset, returning the task-relevant metrics.
        """
        pass
