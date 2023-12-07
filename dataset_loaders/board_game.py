import os
import re

from data_loader import MRToTextDataset


class BoardGameDataset(MRToTextDataset):
    """An MR-to-text dataset in the board_game domain."""

    name = 'board_game'
    delimiters = {
        'da_beg': '(',
        'da_end': ')',
        'da_sep': None,
        'slot_sep': ',',
        'val_beg': '[',
        'val_end': ']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data', 'board_game')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir, 'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')

        return dataset_path

    @staticmethod
    def verbalize_da_name(da_name):
        return da_name.replace('_', ' ')

    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'average_rating': 'average rating',
            'min_playing_time': 'min playing time',
            'max_play_time': 'max play time',
            'complexity_rating': 'complexity rating',
            'alternate_names': 'alternate names',
            'instance_of':'instance of',
            'min_players':'min players',
            'max_players':'max players',
            'publication_year':'publication year',
            'based_on':'based on',
            'game_mechanics':'game mechanics',
            'country_of_origin':'country of origin'
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name.replace('_', ' ')
            # for tok in ['album']:
            #     slot_name_verbalized = re.sub(r'\b{}\b'.format(re.escape(tok)), tok.capitalize(), slot_name_verbalized)

        return slot_name_verbalized

    @staticmethod
    def get_single_word_slot_representation(slot_name):
        single_word_slot_repr = {
            'average_rating': 'average',
            'min_playing_time': 'mintime',
            'max_play_time': 'maxtime',
            'complexity_rating': 'complexity',
            'alternate_names': 'alternate',
            'instance_of': 'instance',
            'min_players': 'minplayers',
            'max_players': 'maxplayers',
            'publication_year': 'year',
            'based_on': 'basedon',
            'game_mechanics': 'mechanics',
            'country_of_origin': 'country'
        }

        return single_word_slot_repr.get(slot_name, slot_name)

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {'name', 'average_rating', 'min_playing_time', 'max_play_time', 'complexity_rating','instance_of','genres','publisher','min_players','max_players','publication_year','based_on','game_mechanics','developer','country_of_origin'},
            'list': {'alternate_names'}
        }