# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Tuple, List, Dict
import numpy as np

from input_example import InputFeatures, EntityType, RelationType, Entity, Relation, Intent, InputExample, CorefDocument, Room
from utils import augment_sentence, get_span


OUTPUT_FORMATS = {}


def register_output_format(format_class):
    OUTPUT_FORMATS[format_class.name] = format_class
    return format_class


class BaseOutputFormat(ABC):
    name = None

    BEGIN_ENTITY_TOKEN = '['
    END_ENTITY_TOKEN = ']'
    SEPARATOR_TOKEN = '|'
    RELATION_SEPARATOR_TOKEN = '='

    @abstractmethod
    def format_output(self, example: InputExample) -> str:
        """
        Format output in augmented natural language.
        """
        raise NotImplementedError

    @abstractmethod
    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract whatever information the task asks for.
        """
        raise NotImplementedError

    def parse_output_sentence(self, example: InputExample, output_sentence: str) -> Tuple[list, bool]:
        """
        Parse an output sentence in augmented language and extract inferred entities and tags.
        Return a pair (predicted_entities, wrong_reconstruction), where:
        - each element of predicted_entities is a tuple (entity_name, tags, start, end)
            - entity_name (str) is the name as extracted from the output sentence
            - tags is a list of tuples, obtained by |-splitting the part of the entity after the entity name
            - this entity corresponds to the tokens example.tokens[start:end]
            - note that the entity_name could differ from ' '.join(example.tokens[start:end]), if the model was not
              able to exactly reproduce the entity name, or if alignment failed
        - wrong_reconstruction (bool) says whether the output_sentence does not match example.tokens exactly

        An example follows.

        example.tokens:
        ['Tolkien', 'wrote', 'The', 'Lord', 'of', 'the', 'Rings']

        output_sentence:
        [ Tolkien | person ] wrote [ The Lord of the Rings | book | author = Tolkien ]

        output predicted entities:
        [
            ('Tolkien', [('person',)], 0, 1),
            ('The Lord of the Rings', [('book',), ('author', 'Tolkien')], 2, 7)
        ]
        """
        output_tokens = []
        unmatched_predicted_entities = []

        # add spaces around special tokens, so that they are alone when we split
        padded_output_sentence = output_sentence
        for special_token in [
            self.BEGIN_ENTITY_TOKEN, self.END_ENTITY_TOKEN,
            self.SEPARATOR_TOKEN, self.RELATION_SEPARATOR_TOKEN,
        ]:
            padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')

        entity_stack = []   # stack of the entities we are extracting from the output sentence
        # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
        # where state is "name" (before the first | separator) or "other" (after the first | separator)

        for token in padded_output_sentence.split():
            if len(token) == 0:
                continue

            elif token == self.BEGIN_ENTITY_TOKEN:
                # begin entity
                start = len(output_tokens)
                entity_stack.append([start, "name", [], []])

            elif token == self.END_ENTITY_TOKEN and len(entity_stack) > 0:
                # end entity
                start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()

                entity_name = ' '.join(entity_name_tokens).strip()
                end = len(output_tokens)

                tags = []

                # split entity_other_tokens by |
                splits = [
                    list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == self.SEPARATOR_TOKEN)
                    if not x
                ]

                if state == "other" and len(splits) > 0:
                    for x in splits:
                        tags.append(tuple(' '.join(x).split(' ' + self.RELATION_SEPARATOR_TOKEN + ' ')))

                unmatched_predicted_entities.append((entity_name, tags, start, end))

            else:
                # a normal token
                if len(entity_stack) > 0:
                    # inside some entities
                    if token == self.SEPARATOR_TOKEN:
                        x = entity_stack[-1]

                        if x[1] == "name":
                            # this token marks the end of name tokens for the current entity
                            x[1] = "other"
                        else:
                            # simply add this token to entity_other_tokens
                            x[3].append(token)

                    else:
                        is_name_token = True

                        for x in reversed(entity_stack):
                            # check state
                            if x[1] == "name":
                                # add this token to entity_name_tokens
                                x[2].append(token)

                            else:
                                # add this token to entity_other tokens and then stop going up in the tree
                                x[3].append(token)
                                is_name_token = False
                                break

                        if is_name_token:
                            output_tokens.append(token)

                else:
                    # outside
                    output_tokens.append(token)

        # check if we reconstructed the original sentence correctly, after removing all spaces
        wrong_reconstruction = (''.join(output_tokens) != ''.join(example.tokens))

        # now we align self.tokens with output_tokens (with dynamic programming)
        cost = np.zeros((len(example.tokens) + 1, len(output_tokens) + 1))  # cost of alignment between tokens[:i]
        # and output_tokens[:j]
        best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

        for i in range(len(example.tokens) + 1):
            for j in range(len(output_tokens) + 1):
                if i == 0 and j == 0:
                    continue

                candidates = []

                # match
                if i > 0 and j > 0:
                    candidates.append(
                        ((0 if example.tokens[i - 1] == output_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

                # skip in the first sequence
                if i > 0:
                    candidates.append((1 + cost[i - 1, j], 2))

                # skip in the second sequence
                if j > 0:
                    candidates.append((1 + cost[i, j - 1], 3))

                chosen_cost, chosen_option = min(candidates)
                cost[i, j] = chosen_cost
                best[i, j] = chosen_option

        # reconstruct best alignment
        matching = {}

        i = len(example.tokens) - 1
        j = len(output_tokens) - 1

        while i >= 0 and j >= 0:
            chosen_option = best[i + 1, j + 1]

            if chosen_option == 1:
                # match
                matching[j] = i
                i, j = i - 1, j - 1

            elif chosen_option == 2:
                # skip in the first sequence
                i -= 1

            else:
                # skip in the second sequence
                j -= 1

        # update predicted entities with the positions in the original sentence
        predicted_entities = []

        for entity_name, entity_tags, start, end in unmatched_predicted_entities:
            new_start = None  # start in the original sequence
            new_end = None  # end in the original sequence

            for j in range(start, end):
                if j in matching:
                    if new_start is None:
                        new_start = matching[j]

                    new_end = matching[j]

            if new_start is not None:
                # predict entity
                entity_tuple = (entity_name, entity_tags, new_start, new_end + 1)
                predicted_entities.append(entity_tuple)

        return predicted_entities, wrong_reconstruction

    def zy_parse_output_sentence(self, example: InputExample, output_sentence: str) -> Tuple[list, bool]:
        """
        Parse an output sentence in augmented language and extract inferred entities and tags.
        Return a pair (predicted_entities, wrong_reconstruction), where:
        - each element of predicted_entities is a tuple (entity_name, tags, start, end)
            - entity_name (str) is the name as extracted from the output sentence
            - tags is a list of tuples, obtained by |-splitting the part of the entity after the entity name
            - this entity corresponds to the tokens example.tokens[start:end]
            - note that the entity_name could differ from ' '.join(example.tokens[start:end]), if the model was not
              able to exactly reproduce the entity name, or if alignment failed
        - wrong_reconstruction (bool) says whether the output_sentence does not match example.tokens exactly

        An example follows.

        example.tokens:
        ['Tolkien', 'wrote', 'The', 'Lord', 'of', 'the', 'Rings']

        output_sentence:
        [ Tolkien | person ] wrote [ The Lord of the Rings | book | author = Tolkien ]

        output predicted entities:
        [
            ('Tolkien', [('person',)], 0, 1),
            ('The Lord of the Rings', [('book',), ('author', 'Tolkien')], 2, 7)
        ]
        """
        output_tokens = []
        unmatched_predicted_entities = []

        # add spaces around special tokens, so that they are alone when we split
        padded_output_sentence = output_sentence
        for special_token in [
            self.BEGIN_ENTITY_TOKEN, self.END_ENTITY_TOKEN,
            self.SEPARATOR_TOKEN, self.RELATION_SEPARATOR_TOKEN,
        ]:
            padded_output_sentence = padded_output_sentence.replace(special_token, ' ' + special_token + ' ')

        entity_stack = []   # stack of the entities we are extracting from the output sentence
        # this is a list of lists [start, state, entity_name_tokens, entity_other_tokens]
        # where state is "name" (before the first | separator) or "other" (after the first | separator)

        for token in padded_output_sentence.split():
            if len(token) == 0:
                continue

            elif token == self.BEGIN_ENTITY_TOKEN:
                # begin entity
                start = len(output_tokens)
                entity_stack.append([start, "other", [], []])

            elif token == self.END_ENTITY_TOKEN and len(entity_stack) > 0:
                # end entity
                start, state, entity_name_tokens, entity_other_tokens = entity_stack.pop()

                entity_name = ' '.join(entity_name_tokens).strip()
                end = len(output_tokens)

                tags = []

                # split entity_other_tokens by |
                splits = [
                    list(y) for x, y in itertools.groupby(entity_other_tokens, lambda z: z == self.SEPARATOR_TOKEN)
                    if not x
                ]

                if state == "other" and len(splits) > 0:
                    for x in splits:
                        tags.append(tuple(' '.join(x).split(' ' + self.RELATION_SEPARATOR_TOKEN + ' ')))

                unmatched_predicted_entities.append((entity_name, tags, start, end))

            else:
                # a normal token
                if len(entity_stack) > 0:
                    # inside some entities
                    if token == self.SEPARATOR_TOKEN:
                        x = entity_stack[-1]

                        if x[1] == "name":
                            # this token marks the end of name tokens for the current entity
                            x[1] = "other"
                        else:
                            # simply add this token to entity_other_tokens
                            x[3].append(token)

                    else:
                        is_name_token = True

                        for x in reversed(entity_stack):
                            # check state
                            if x[1] == "name":
                                # add this token to entity_name_tokens
                                x[2].append(token)

                            else:
                                # add this token to entity_other tokens and then stop going up in the tree
                                x[3].append(token)
                                is_name_token = False
                                break

                        if is_name_token:
                            output_tokens.append(token)

                else:
                    # outside
                    output_tokens.append(token)

        # check if we reconstructed the original sentence correctly, after removing all spaces
        wrong_reconstruction = (''.join(output_tokens) != ''.join(example.tokens))

        # now we align self.tokens with output_tokens (with dynamic programming)
        cost = np.zeros((len(example.tokens) + 1, len(output_tokens) + 1))  # cost of alignment between tokens[:i]
        # and output_tokens[:j]
        best = np.zeros_like(cost, dtype=int)  # best choice when aligning tokens[:i] and output_tokens[:j]

        for i in range(len(example.tokens) + 1):
            for j in range(len(output_tokens) + 1):
                if i == 0 and j == 0:
                    continue

                candidates = []

                # match
                if i > 0 and j > 0:
                    candidates.append(
                        ((0 if example.tokens[i - 1] == output_tokens[j - 1] else 1) + cost[i - 1, j - 1], 1))

                # skip in the first sequence
                if i > 0:
                    candidates.append((1 + cost[i - 1, j], 2))

                # skip in the second sequence
                if j > 0:
                    candidates.append((1 + cost[i, j - 1], 3))

                chosen_cost, chosen_option = min(candidates)
                cost[i, j] = chosen_cost
                best[i, j] = chosen_option

        # reconstruct best alignment
        matching = {}

        i = len(example.tokens) - 1
        j = len(output_tokens) - 1

        while i >= 0 and j >= 0:
            chosen_option = best[i + 1, j + 1]

            if chosen_option == 1:
                # match
                matching[j] = i
                i, j = i - 1, j - 1

            elif chosen_option == 2:
                # skip in the first sequence
                i -= 1

            else:
                # skip in the second sequence
                j -= 1

        # update predicted entities with the positions in the original sentence
        predicted_entities = []

        for entity_name, entity_tags, start, end in unmatched_predicted_entities:
            new_start = None  # start in the original sequence
            new_end = None  # end in the original sequence

            entity_tuple = (entity_name, entity_tags, new_start, new_end)
            predicted_entities.append(entity_tuple)

        return predicted_entities, wrong_reconstruction

special_token_id = 32000
room_type_list = {'living room':0, 'master room':1, 'kitchen':2, 
    'bathroom 1':3,'bathroom 2':4,'bathroom 3':5, 'dining room':6, 'common room 1':7,'common room 2':8,'common room 3':9,
    'common room 4':10, 'balcony 1':11,'balcony 2':12,
    'balcony 3':13, 'entrance':14, 'storage':15}
key2ind = {'living room': 1, 'master room': 2, 'kitchen': 3, 'bathroom 1': 4, 'bathroom 2': 5, 'bathroom 3': 6, 'dining room': 7, 'common room 1': 8, 'common room 2': 9, 'common room 3': 10, 'common room 4': 11, 'balcony 1': 12, 'balcony 2': 13, 'balcony 3': 14, 'entrance': 15, 'storage': 16, '0': 17, '1': 18, '2': 19, '3': 20, '4': 21, '5': 22, '6': 23, '7': 24, '8': 25, '9': 26, '10': 27, '11': 28, '12': 29, '13': 30, '14': 31, '15': 32, '16': 33, '17': 34, '18': 35, '19': 36, '20': 37, '21': 38, '22': 39, '23': 40, '24': 41, '25': 42, '26': 43, '27': 44, '28': 45, '29': 46, '30': 47, '31': 48, '32': 49, '33': 50, '34': 51, '35': 52, '36': 53, '37': 54, '38': 55, '39': 56, '40': 57, '41': 58, '42': 59, '43': 60, '44': 61, '45': 62, '46': 63, '47': 64, '48': 65, '49': 66, '50': 67, '51': 68, '52': 69, '53': 70, '54': 71, '55': 72, '56': 73, '57': 74, '58': 75, '59': 76, '60': 77, '61': 78, '62': 79, '63': 80, '64': 81, '65': 82, '66': 83, '67': 84, '68': 85, '69': 86, '70': 87, '71': 88, '72': 89, '73': 90, '74': 91, '75': 92, '76': 93, '77': 94, '78': 95, '79': 96, '80': 97, '81': 98, '82': 99, '83': 100, '84': 101, '85': 102, '86': 103, '87': 104, '88': 105, '89': 106, '90': 107, '91': 108, '92': 109, '93': 110, '94': 111, '95': 112, '96': 113, '97': 114, '98': 115, '99': 116, '100': 117, '101': 118, '102': 119, '103': 120, '104': 121, '105': 122, '106': 123, '107': 124, '108': 125, '109': 126, '110': 127, '111': 128, '112': 129, '113': 130, '114': 131, '115': 132, '116': 133, '117': 134, '118': 135, '119': 136, '120': 137, '121': 138, '122': 139, '123': 140, '124': 141, '125': 142, '126': 143, '127': 144, '128': 145, '129': 146, '130': 147, '131': 148, '132': 149, '133': 150, '134': 151, '135': 152, '136': 153, '137': 154, '138': 155, '139': 156, '140': 157, '141': 158, '142': 159, '143': 160, '144': 161, '145': 162, '146': 163, '147': 164, '148': 165, '149': 166, '150': 167, '151': 168, '152': 169, '153': 170, '154': 171, '155': 172, '156': 173, '157': 174, '158': 175, '159': 176, '160': 177, '161': 178, '162': 179, '163': 180, '164': 181, '165': 182, '166': 183, '167': 184, '168': 185, '169': 186, '170': 187, '171': 188, '172': 189, '173': 190, '174': 191, '175': 192, '176': 193, '177': 194, '178': 195, '179': 196, '180': 197, '181': 198, '182': 199, '183': 200, '184': 201, '185': 202, '186': 203, '187': 204, '188': 205, '189': 206, '190': 207, '191': 208, '192': 209, '193': 210, '194': 211, '195': 212, '196': 213, '197': 214, '198': 215, '199': 216, '200': 217, '201': 218, '202': 219, '203': 220, '204': 221, '205': 222, '206': 223, '207': 224, '208': 225, '209': 226, '210': 227, '211': 228, '212': 229, '213': 230, '214': 231, '215': 232, '216': 233, '217': 234, '218': 235, '219': 236, '220': 237, '221': 238, '222': 239, '223': 240, '224': 241, '225': 242, '226': 243, '227': 244, '228': 245, '229': 246, '230': 247, '231': 248, '232': 249, '233': 250, '234': 251, '235': 252, '236': 253, '237': 254, '238': 255, '239': 256, '240': 257, '241': 258, '242': 259, '243': 260, '244': 261, '245': 262, '246': 263, '247': 264, '248': 265, '249': 266, '250': 267, '251': 268, '252': 269, '253': 270, '254': 271, '255': 272}
ind2key = {1: 'living room', 2: 'master room', 3: 'kitchen', 4: 'bathroom 1', 5: 'bathroom 2', 6: 'bathroom 3', 7: 'dining room', 8: 'common room 1', 9: 'common room 2', 10: 'common room 3', 11: 'common room 4', 12: 'balcony 1', 13: 'balcony 2', 14: 'balcony 3', 15: 'entrance', 16: 'storage', 17: '0', 18: '1', 19: '2', 20: '3', 21: '4', 22: '5', 23: '6', 24: '7', 25: '8', 26: '9', 27: '10', 28: '11', 29: '12', 30: '13', 31: '14', 32: '15', 33: '16', 34: '17', 35: '18', 36: '19', 37: '20', 38: '21', 39: '22', 40: '23', 41: '24', 42: '25', 43: '26', 44: '27', 45: '28', 46: '29', 47: '30', 48: '31', 49: '32', 50: '33', 51: '34', 52: '35', 53: '36', 54: '37', 55: '38', 56: '39', 57: '40', 58: '41', 59: '42', 60: '43', 61: '44', 62: '45', 63: '46', 64: '47', 65: '48', 66: '49', 67: '50', 68: '51', 69: '52', 70: '53', 71: '54', 72: '55', 73: '56', 74: '57', 75: '58', 76: '59', 77: '60', 78: '61', 79: '62', 80: '63', 81: '64', 82: '65', 83: '66', 84: '67', 85: '68', 86: '69', 87: '70', 88: '71', 89: '72', 90: '73', 91: '74', 92: '75', 93: '76', 94: '77', 95: '78', 96: '79', 97: '80', 98: '81', 99: '82', 100: '83', 101: '84', 102: '85', 103: '86', 104: '87', 105: '88', 106: '89', 107: '90', 108: '91', 109: '92', 110: '93', 111: '94', 112: '95', 113: '96', 114: '97', 115: '98', 116: '99', 117: '100', 118: '101', 119: '102', 120: '103', 121: '104', 122: '105', 123: '106', 124: '107', 125: '108', 126: '109', 127: '110', 128: '111', 129: '112', 130: '113', 131: '114', 132: '115', 133: '116', 134: '117', 135: '118', 136: '119', 137: '120', 138: '121', 139: '122', 140: '123', 141: '124', 142: '125', 143: '126', 144: '127', 145: '128', 146: '129', 147: '130', 148: '131', 149: '132', 150: '133', 151: '134', 152: '135', 153: '136', 154: '137', 155: '138', 156: '139', 157: '140', 158: '141', 159: '142', 160: '143', 161: '144', 162: '145', 163: '146', 164: '147', 165: '148', 166: '149', 167: '150', 168: '151', 169: '152', 170: '153', 171: '154', 172: '155', 173: '156', 174: '157', 175: '158', 176: '159', 177: '160', 178: '161', 179: '162', 180: '163', 181: '164', 182: '165', 183: '166', 184: '167', 185: '168', 186: '169', 187: '170', 188: '171', 189: '172', 190: '173', 191: '174', 192: '175', 193: '176', 194: '177', 195: '178', 196: '179', 197: '180', 198: '181', 199: '182', 200: '183', 201: '184', 202: '185', 203: '186', 204: '187', 205: '188', 206: '189', 207: '190', 208: '191', 209: '192', 210: '193', 211: '194', 212: '195', 213: '196', 214: '197', 215: '198', 216: '199', 217: '200', 218: '201', 219: '202', 220: '203', 221: '204', 222: '205', 223: '206', 224: '207', 225: '208', 226: '209', 227: '210', 228: '211', 229: '212', 230: '213', 231: '214', 232: '215', 233: '216', 234: '217', 235: '218', 236: '219', 237: '220', 238: '221', 239: '222', 240: '223', 241: '224', 242: '225', 243: '226', 244: '227', 245: '228', 246: '229', 247: '230', 248: '231', 249: '232', 250: '233', 251: '234', 252: '235', 253: '236', 254: '237', 255: '238', 256: '239', 257: '240', 258: '241', 259: '242', 260: '243', 261: '244', 262: '245', 263: '246', 264: '247', 265: '248', 266: '249', 267: '250', 268: '251', 269: '252', 270: '253', 271: '254', 272: '255'}

@register_output_format
class FloorPlanOutputFormat(BaseOutputFormat):
    """
    Output formate for Floor plan generation (Sicong)
    """
    name = 'floorplan'

    def format_short_output(self, example: InputExample) -> str:
        string = ''
        start_token = '['
        end_token = ']'
        sep = '|'
        for room in example.rooms:
            # string += f'{start_token} {room.type} {sep} x coordinate = {str(room.x)} {sep} y coordinate = {str(room.y)} {sep} height = {str(room.h)} {sep} width = {str(room.w)} {end_token} '
            string += f'{room.type} {str(room.x_min)} {str(room.y_min)} {str(room.x_max)} {str(room.y_max)} '
        return string

    def format_short_output_with_relation(self, example: InputExample) -> str:
        string = ''
        start_token = '['
        end_token = ']'
        sep = '|'
        for room in example.rooms:
            # string += f'{start_token} {room.type} {sep} x coordinate = {str(room.x)} {sep} y coordinate = {str(room.y)} {sep} height = {str(room.h)} {sep} width = {str(room.w)} {end_token} '
            string += f'{room.type} {str(room.x_min)} {str(room.y_min)} {str(room.x_max)} {str(room.y_max)} {str(room.near_x_min)} {str(room.near_y_min)} {str(room.near_x_max)} {str(room.near_y_max)} '
        return string
    
    def format_long_output(self, example: InputExample) -> str:
        string = ''
        start_token = '['
        end_token = ']'
        sep = '|'
        for room in example.rooms:
            string += f'{start_token} {room.type} {sep} x min = {str(room.x_min)} {sep} y min = {str(room.y_min)} {sep} x_max = {str(room.x_max)} {sep} y_max = {str(room.y_max)} {end_token} '
            # string += f'{room.type} {str(room.x)} {str(room.y)} {str(room.h)} {str(room.w)} '
        return string
    
    def format_short_output_(self, example: InputExample) -> str:
        string = ''
        start_token = '['
        end_token = ']'
        sep = '|'
        for room in example.rooms:
            string += f'{start_token} {room.type} {sep} x coordinate = {str(room.x)} {sep} y coordinate = {str(room.y)} {sep} height = {str(room.h)} {sep} width = {str(room.w)} {end_token} '
            # string += f'{room.type} {str(room.x_min)} {str(room.y_min)} {str(room.x_max)} {str(room.y_max)} {str(room.near_x_min)} {str(room.near_y_min)} {str(room.near_x_max)} {str(room.near_y_max)} '
        return string
    
    def format_output_index(self, example: InputExample) -> str:
        output_index = []
        for room in example.rooms:
            # string += f'{start_token} {room.type} {sep} x coordinate = {str(room.x)} {sep} y coordinate = {str(room.y)} {sep} height = {str(room.h)} {sep} width = {str(room.w)} {end_token} '
            # string += f'{room.type} {str(room.x)} {str(room.y)} {str(room.h)} {str(room.w)} '
            output_index.extend([key2ind[room.type], key2ind[str(room.x)], key2ind[str(room.y)],key2ind[str(room.h)],key2ind[str(room.w)]])
        return output_index

    def format_output(self, example: InputExample) -> str:
        augmentations = []
        for room in example.rooms:
            tags = []
            tags.append((room.type,))
            tags.append(('x coordinate',str(room.x)))
            tags.append(('y coordinate',str(room.y)))
            tags.append(('height', str(room.h)))
            tags.append(('width', str(room.w)))
            augmentations.append((
                tags,
                room.start,
                room.end,
            ))

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str, prediction_index: list):
        # Sicong: format output sentence to parse
        new_rooms = []
        output_tokens = output_sentence.split()
        if '[' in output_tokens and '|' in output_tokens and 'x_max' in output_tokens: # long output format
            output_tokens = output_sentence.split('[')
            output_tokens.pop(0)
            tokens = "".join(output_tokens).split(']')
            tokens.pop(-1)
            for token in tokens:
                tok = token.split('|')
                room_type = tok[0].strip()
                for t in tok[1:]:
                    if 'x min' in t:
                        x_min = t.split('=')[1].strip()
                    elif 'y min' in t:
                        y_min = t.split('=')[1].strip()
                    elif 'x_max' in t:
                        x_max = t.split('=')[1].strip()
                    elif 'y_max' in t:
                        y_max = t.split('=')[1].strip()
                x = int((int(x_min) + int(x_max)) / 2)
                y = int((int(y_min) + int(y_max)) / 2)
                w = int(y_max) - int(y_min)
                h = int(x_max) - int(x_min)
                room = Room(type=room_type, x=x, y=y, h=h, w=w)
                new_rooms.append(room)
            predicted_examples = InputExample(rooms=new_rooms)
            output_sentence_ = self.format_short_output_(predicted_examples)
            pass
        elif '[' in output_tokens and '|' in output_tokens: # original output format
            output_sentence_ = output_sentence
            pass
        elif '-1' in output_tokens: # short-relation output format
            rooms = []
            room_attributes = []
            index = []
            flag = False
            for i in range(len(output_tokens)):
                if flag == False:
                    try:
                        token = int(output_tokens[i])
                        if token > 4:
                            flag = True
                            index.append(i)
                    except:
                        pass
                else:
                    try:
                        token = int(output_tokens[i])
                    except:
                        flag = False
            prev_idx = 0
            for idx in index:
                room_attributes.append(output_tokens[idx:idx+4])
                rooms.append(" ".join(output_tokens[prev_idx:idx]))
                prev_idx = idx+8
            for i in range(len(rooms)):
                # xmin ymin xmax ymax
                x=int(( int(room_attributes[i][0])+int(room_attributes[i][2]) )/2)
                y=int(( int(room_attributes[i][1])+int(room_attributes[i][3]) )/2)
                w = int(room_attributes[i][3])-int(room_attributes[i][1])
                h = int(room_attributes[i][2])-int(room_attributes[i][0])
                new_rooms.append(Room(type=rooms[i], x=x, y=y, h=h, w=w))
            predicted_examples = InputExample(rooms=new_rooms)
            output_sentence_ = self.format_short_output_(predicted_examples)
            pass
        else: # short output format
            rooms = []
            room_attributes = []
            index = []
            flag = False
            for i in range(len(output_tokens)):
                if flag == False:
                    try:
                        token = int(output_tokens[i])
                        if token > 4:
                            flag = True
                            index.append(i)
                    except:
                        pass
                else:
                    try:
                        token = int(output_tokens[i])
                    except:
                        flag = False
            prev_idx = 0
            for idx in index:
                room_attributes.append(output_tokens[idx:idx+4])
                rooms.append(" ".join(output_tokens[prev_idx:idx]))
                prev_idx = idx+4
            for i in range(len(rooms)):
                # xmin ymin xmax ymax
                x=int(( int(room_attributes[i][0])+int(room_attributes[i][2]) )/2)
                y=int(( int(room_attributes[i][1])+int(room_attributes[i][3]) )/2)
                w = int(room_attributes[i][3])-int(room_attributes[i][1])
                h = int(room_attributes[i][2])-int(room_attributes[i][0])
                new_rooms.append(Room(type=rooms[i], x=x, y=y, h=h, w=w))
            predicted_examples = InputExample(rooms=new_rooms)
            output_sentence_ = self.format_short_output_(predicted_examples)

        # pre_index_ = []
        # for idx in prediction_index:
        #     if idx == 0:
        #         pass
        #     else:
        #         pre_index_.append(idx)
        # assert pre_index_[-1] == 1
        # pre_index_.pop(-1)  # remove eos token
        # assert type(len(pre_index_)//5) == int
        # num_rooms_ = len(pre_index_)//5
        # for i in range(num_rooms_):
        #     new_rooms.append(Room(type=ind2key[pre_index_[5*i+0]],x=ind2key[pre_index_[5*i+1]],y=ind2key[pre_index_[5*i+2]],h=ind2key[pre_index_[5*i+3]],w=ind2key[pre_index_[5*i+4]]))
        # predicted_examples = InputExample(rooms=new_rooms)
        # output_sentence_ = self.format_short_output_(predicted_examples)

        room_types = ['living room', 'master room', 'kitchen', 'bathroom', 'dining room', 'common room 2', 'common room 3', 'common room 1', 'common room 4', 'balcony'
            , 'entrance', 'storage', 'common room']
        attribute_types = ['x coordinate', 'y coordinate', 'height', 'width']
        format_error = False    # whether the augmented language format is invalid
        label_error = False

        if output_sentence_.count(self.BEGIN_ENTITY_TOKEN) != output_sentence_.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        # parse output sentence
        # raw_predictions, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)
        raw_predictions, wrong_reconstruction = self.zy_parse_output_sentence(example, output_sentence_)

        # update predicted entities with the positions in the original sentence
        predicted_rooms_by_name = defaultdict(list)
        predicted_rooms = set()
        raw_predicted_relations = []

        # process and filter entities
        for entity_name, tags, start, end in raw_predictions:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the room type
                format_error = True
                continue

            room_type = tags[0][0]

            if room_type in room_types or room_type[:-2] in room_types:
                room_tuple = (room_type, start, end)
                predicted_rooms.add(room_tuple)
                predicted_rooms_by_name[room_type].append(room_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        attribute_type, value = tag
                        if attribute_type in attribute_types:
                            raw_predicted_relations.append((attribute_type, value, room_tuple, room_type))
                        else:
                            label_error = True

                    else:
                        # the relation tag has the wrong length
                        format_error = True

            else:
                # the predicted entity type does not exist
                label_error = True
        
            #error = format_error or label_error or wrong_reconstruction  # whether there is syntax error

        return predicted_rooms_by_name, predicted_rooms, raw_predicted_relations, wrong_reconstruction, format_error, label_error

@register_output_format
class JointEROutputFormat(BaseOutputFormat):
    """
    Output format for joint entity and relation extraction.
    """
    name = 'joint_er'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [ Tolkien | person | born in = here ] was born [ here | location ]
        """
        # organize relations by head entity
        relations_by_entity = {entity: [] for entity in example.entities}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))

        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]
            for relation_type, tail in relations_by_entity[entity]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start:tail.end])))

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set, bool, bool, bool, bool]:
        """
        Process an output sentence to extract predicted entities and relations (among the given entity/relation types).

        Return the predicted entities, predicted relations, and four booleans which describe if certain kinds of errors
        occurred (wrong reconstruction of the sentence, label error, entity error, augmented language format error).
        """
        label_error = False     # whether the output sentence has at least one non-existing entity or relation type
        entity_error = False    # whether there is at least one relation pointing to a non-existing head entity
        format_error = False    # whether the augmented language format is invalid

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        entity_types = set(entity_type.natural for entity_type in entity_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()
        raw_predicted_relations = []

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                format_error = True
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
                predicted_entities_by_name[entity_name].append(entity_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_entity = tag
                        if relation_type in relation_types:
                            raw_predicted_relations.append((relation_type, entity_tuple, related_entity))
                        else:
                            label_error = True

                    else:
                        # the relation tag has the wrong length
                        format_error = True

            else:
                # the predicted entity type does not exist
                label_error = True

        predicted_relations = set()

        for relation_type, entity_tuple, related_entity in raw_predicted_relations:
            if related_entity in predicted_entities_by_name:
                # look for the closest instance of the related entity
                # (there could be many of them)
                _, head_start, head_end = entity_tuple
                candidates = sorted(
                    predicted_entities_by_name[related_entity],
                    key=lambda x:
                    min(abs(x[1] - head_end), abs(head_start - x[2]))
                )

                for candidate in candidates:
                    relation = (relation_type, entity_tuple, candidate)

                    if relation not in predicted_relations:
                        predicted_relations.add(relation)
                        break

            else:
                # cannot find the related entity in the sentence
                entity_error = True

        return predicted_entities, predicted_relations, wrong_reconstruction, label_error, entity_error, format_error



@register_output_format
class JointICSLFormat(JointEROutputFormat):
    """
    Output format for joint intent classification and slot labeling.
    """
    name = 'joint_icsl'
    BEGIN_INTENT_TOKEN = "(("
    END_INTENT_TOKEN = "))"
        
    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language.
        """
        augmentations = []
        for entity in example.entities:
            tags = [(entity.type.natural,)]

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        augmented_sentence = augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)
        
        return (f"(( {example.intent.natural} )) " + augmented_sentence)

    def run_inference(self, example: InputExample, output_sentence: str,
            entity_types: Dict[str, EntityType] = None) -> Tuple[str, set]:
        entity_types = set(entity_type.natural for entity_type in entity_types.values())

        # parse output sentence
        # get intent
        for special_token in [self.BEGIN_INTENT_TOKEN, self.END_INTENT_TOKEN]:
            output_sentence.replace(special_token, ' ' + special_token + ' ')

        output_sentence_tokens = output_sentence.split()

        if self.BEGIN_INTENT_TOKEN in output_sentence_tokens and \
                self.END_INTENT_TOKEN in output_sentence_tokens:
            intent = output_sentence.split(self.BEGIN_INTENT_TOKEN)[1].split(self.END_INTENT_TOKEN)[0].strip()
            output_sentence = output_sentence.split(self.END_INTENT_TOKEN)[1]   # remove intent from sentence

        label_error = False     # whether the output sentence has at least one non-existing entity or relation type
        format_error = False    # whether the augmented language format is invalid

        if output_sentence.count(self.BEGIN_ENTITY_TOKEN) != output_sentence.count(self.END_ENTITY_TOKEN):
            # the parentheses do not match
            format_error = True

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities_by_name = defaultdict(list)
        predicted_entities = set()

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                format_error = True
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)
            else:
                label_error = True

        return intent, predicted_entities, wrong_reconstruction, label_error, format_error
  

@register_output_format
class EventOutputFormat(JointEROutputFormat):
    """
    Output format for event extraction, where an input example contains exactly one trigger.
    """
    name = 'ace2005_event'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, similarly to JointEROutputFormat (but we also consider triggers).
        """
        # organize relations by head entity
        relations_by_entity = {entity: [] for entity in example.entities + example.triggers}
        for relation in example.relations:
            relations_by_entity[relation.head].append((relation.type, relation.tail))

        augmentations = []
        for entity in (example.entities + example.triggers):
            if not relations_by_entity[entity]:
                continue

            tags = [(entity.type.natural,)]
            for relation_type, tail in relations_by_entity[entity]:
                tags.append((relation_type.natural, ' '.join(example.tokens[tail.start:tail.end])))

            augmentations.append((
                tags,
                entity.start,
                entity.end,
            ))

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set, bool]:
        """
        Process an output sentence to extract arguments, given as entities and relations.
        """
        entity_types = set(entity_type.natural for entity_type in entity_types.values())
        relation_types = set(relation_type.natural for relation_type in relation_types.values()) \
            if relation_types is not None else {}

        triggers = example.triggers
        assert len(triggers) <= 1
        if len(triggers) == 0:
            # we do not have triggers
            return set(), set(), False

        trigger = triggers[0]

        # parse output sentence
        raw_predicted_entities, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        # update predicted entities with the positions in the original sentence
        predicted_entities = set()
        predicted_relations = set()

        # process and filter entities
        for entity_name, tags, start, end in raw_predicted_entities:
            if len(tags) == 0 or len(tags[0]) > 1:
                # we do not have a tag for the entity type
                continue

            entity_type = tags[0][0]

            if entity_type in entity_types:
                entity_tuple = (entity_type, start, end)
                predicted_entities.add(entity_tuple)

                # process tags to get relations
                for tag in tags[1:]:
                    if len(tag) == 2:
                        relation_type, related_entity = tag
                        if relation_type in relation_types:
                            predicted_relations.add(
                                (relation_type, entity_tuple, (trigger.type.natural, trigger.start, trigger.end))
                            )

        return predicted_entities, predicted_relations, wrong_reconstruction


@register_output_format
class CorefOutputFormat(BaseOutputFormat):
    """
    Output format for coreference resolution.
    """
    name = 'coref'

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        Tolkien's epic novel [ The Lord of the Rings ] was published in 1954-1955, years after the
        [ book | The Lord of the Rings ] was completed.
        """
        augmentations = []

        for group in example.groups:
            previous_entity = None
            for entity in group:
                augmentation = (
                    [(' '.join(example.tokens[previous_entity.start:previous_entity.end]),)]
                    if previous_entity is not None else [],
                    entity.start,
                    entity.end,
                )
                augmentations.append(augmentation)
                previous_entity = entity

        return augment_sentence(example.tokens, augmentations, self.BEGIN_ENTITY_TOKEN, self.SEPARATOR_TOKEN,
                                self.RELATION_SEPARATOR_TOKEN, self.END_ENTITY_TOKEN)

    def run_inference(self, example: InputExample, output_sentence: str) \
            -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Process an output sentence to extract coreference relations.

        Return a list of ((start, end), parent) where (start, end) denote an entity span, and parent is either None
        or another (previous) entity span.
        """
        raw_annotations, wrong_reconstruction = self.parse_output_sentence(example, output_sentence)

        res = []
        previous_entities = {}
        for entity, tags, start, end in raw_annotations:
            entity_span = (start, end)

            if len(tags) > 0 and tags[0][0] in previous_entities:
                previous_entity = tags[0][0]
                res.append((entity_span, previous_entities[previous_entity]))

            else:
                # no previous entity found
                res.append((entity_span, None))

            # record this entity
            previous_entities[entity] = entity_span

        return res


@register_output_format
class RelationClassificationOutputFormat(BaseOutputFormat):
    """
    Output format for relation classification.
    """
    name = 'rel_output'

    def format_output(self, example: InputExample) -> str:
        en1_span = [example.entities[0].start, example.entities[0].end]
        en2_span = [example.entities[1].start, example.entities[1].end]
        words = example.tokens
        s = f"relationship between {self.BEGIN_ENTITY_TOKEN} {get_span(words, en1_span)} {self.END_ENTITY_TOKEN} and " \
            f"{self.BEGIN_ENTITY_TOKEN} {get_span(words, en2_span)} {self.END_ENTITY_TOKEN} " \
            f"{self.RELATION_SEPARATOR_TOKEN} {example.relations[0].type.natural}"
        return s.strip()

    def run_inference(self, example: InputExample, output_sentence: str,
                      entity_types: Dict[str, EntityType] = None, relation_types: Dict[str, RelationType] = None) \
            -> Tuple[set, set]:
        """
        Process an output sentence to extract the predicted relation.

        Return an empty list of entities and a single relation, so that it is compatible with joint entity-relation
        extraction datasets.
        """
        predicted_relation = output_sentence.split(self.RELATION_SEPARATOR_TOKEN)[-1].strip()
        predicted_entities = set()  # leave this empty as we only predict the relation

        predicted_relations = {(
            predicted_relation,
            example.relations[0].head.to_tuple() if example.relations[0].head else None,
            example.relations[0].tail.to_tuple() if example.relations[0].tail else None,
        )}

        return predicted_entities, predicted_relations


@register_output_format
class MultiWozOutputFormat(BaseOutputFormat):
    """
    Output format for the MultiWoz DST dataset.
    """
    name = 'multi_woz'

    none_slot_value = 'not given'
    domain_ontology = {
        'hotel': [
            'price range',
            'type',
            'parking',
            'book stay',
            'book day',
            'book people',
            'area',
            'stars',
            'internet',
            'name'
        ],
        'train': [
            'destination',
            'day',
            'departure',
            'arrive by',
            'book people',
            'leave at'
        ],
        'attraction': ['type', 'area', 'name'],
        'restaurant': [
            'book people',
            'book day',
            'book time',
            'food',
            'price range',
            'name',
            'area'
        ],
        'taxi': ['leave at', 'destination', 'departure', 'arrive by'],
        'bus': ['people', 'leave at', 'destination', 'day', 'arrive by', 'departure'],
        'hospital': ['department']
    }

    def format_output(self, example: InputExample) -> str:
        """
        Get output in augmented natural language, for example:
        [belief] hotel price range cheap , hotel type hotel , duration two [belief]
        """
        turn_belief = example.belief_state
        domain_to_slots = defaultdict(dict)
        for label in turn_belief:
            domain, slot, value = label.split("-")
            domain_to_slots[domain][slot] = value

        # add slots that are not given
        for domain, slot_dict in domain_to_slots.items():
            for slot in self.domain_ontology[domain]:
                if slot not in slot_dict:
                    slot_dict[slot] = self.none_slot_value

        output_list = []
        for domain, slot_dict in sorted(domain_to_slots.items(), key=lambda p: p[0]):
            output_list += [
                f"{domain} {slot} {value}" for slot, value in sorted(slot_dict.items(), key=lambda p: p[0])
            ]
        output = " , ".join(output_list)
        output = f"[belief] {output} [belief]"
        return output

    def run_inference(self, example: InputExample, output_sentence: str):
        """
        Process an output sentence to extract the predicted belief.
        """
        start = output_sentence.find("[belief]")
        end = output_sentence.rfind("[belief]")

        label_span = output_sentence[start+len("[belief]"):end]
        belief_set = set([
            slot_value.strip() for slot_value in label_span.split(",")
            if self.none_slot_value not in slot_value
        ])
        return belief_set
