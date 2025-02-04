import json
import os


def find_first_in_list(val, lst):
    idx = -1
    pos = -1

    for i, elem in enumerate(lst):
        if val == elem:
            idx = i

    if idx >= 0:
        # Calculate approximate character position of the matched value
        punct_cnt = lst[:idx].count('.') + lst[:idx].count(',')
        pos = len(' '.join(lst[:idx])) + 1 - punct_cnt

    return idx, pos


def find_all_in_list(val, lst):
    indexes = []
    positions = []

    for i, elem in enumerate(lst):
        if val == elem:
            indexes.append(i)

            # Calculate approximate character position of the matched value
            punct_cnt = lst[:i].count('.') + lst[:i].count(',')
            positions.append(len(' '.join(lst[:i])) + 1 - punct_cnt)

    return indexes, positions


def get_slot_value_alternatives(slot):
    # slot_alternatives_path = os.path.join('slot_aligner', 'alignment', 'alternatives.json')
    slot_alternatives_path = "D:\data2text-nlg\slot_aligner\\alignment\\alternatives.json"
    with open(slot_alternatives_path, 'r') as f_alternatives:
        alternatives_dict = json.load(f_alternatives)

    return alternatives_dict.get(slot, {})
