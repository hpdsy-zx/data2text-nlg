import os
import re

from data_loader import MRToTextDataset

class SongDataset(MRToTextDataset):
    """An MR-to-text dataset in the song domain."""

    name = 'song'
    delimiters = {
        'da_beg':'(',
        'da_end':')',
        'da_sep':None,
        'slot_sep':',',
        'val_beg':'[',
        'val_end':']'
    }

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data','song')
        if partition == 'valid':
            dataset_path = os.path.join(dataset_dir,'valid.csv')
        elif partition == 'test':
            dataset_path = os.path.join(dataset_dir, 'test.csv')
        else:
            dataset_path = os.path.join(dataset_dir, 'train.csv')
        
        return dataset_path
    
    @staticmethod
    def verbalize_da_name(da_name):
        return da_name.replace('_',' ')
    
    @staticmethod
    def verbalize_slot_name(slot_name):
        slots_to_override = {
            'publication_date':'publication date',
            'publication_year':'publication year',
            'record_label':'record label',
            'instance_of':'instance of',
            'is_from_album':'is from ablum'
        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name.replace('_', ' ')

        return slot_name_verbalized
    
    @staticmethod
    def get_single_word_slot_representation(slot_name):
        single_word_slot_repr = {
            'publication_date':'date',
            'publication_year':'year',
            'record_label':'record',
            'instance_of':'instance',
            'is_from_album':'album'
        }

        return single_word_slot_repr.get(slot_name,slot_name)
    
    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple':{'name','publication_date','publication_year','rating','is_from_album'},
            'list':{'performer','producer','genres','record_label','instance_of'}
        }
    
class Song20PercentDataset(SongDataset):
    """A 20% sample of the Song dataset"""
    name = 'song_20_percent'

    @staticmethod
    def get_data_file_path(partition):
        dataset_dir = os.path.join('data','song')
        if partition == 'train':
            dataset_path = os.path.join('data','song')
        