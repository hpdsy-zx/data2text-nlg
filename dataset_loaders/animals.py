import os
import re

from data_loader import MRToTextDataset


class AnimalsDataset(MRToTextDataset):
    """An MR-to-text dataset in the animals domain."""

    name = 'animals'
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
        dataset_dir = os.path.join('data', 'animals')
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
            'taxonomy.scientific_name': 'taxonomy scientific name',
            'taxonomy.kingdom': 'taxonomy kingdom',
            'taxonomy.phylum': 'taxonomy phylum',
            'taxonomy.class': 'taxonomy class',
            'taxonomy.order': 'taxonomy order',
            'taxonomy.genus': 'taxonomy genus',
            'characteristics.prey': 'characteristics prey',
            'characteristics.name_of_young': 'characteristics name of young',
            'characteristics.group_behavior': 'characteristics group behavior',
            'characteristics.estimated_population_size': 'characteristics estimated population size',
            'characteristics.biggest_threat': 'characteristics biggest threat',

            'characteristics.most_distinctive_feature': 'characteristics most distinctive feature',
            'characteristics.other_name(s)': 'characteristics other name(s)',
            'characteristics.gestation_period': 'characteristics gestation period',
            'characteristics.habitat': 'characteristics habitat',
            'characteristics.diet': 'characteristics diet',
            'characteristics.average_litter_size': 'characteristics average litter size',

            'characteristics.lifestyle': 'characteristics lifestyle',
            'characteristics.common_name': 'characteristics common name',
            'characteristics.number_of_species': 'characteristics number of species',
            'characteristics.location': 'characteristics location',
            'characteristics.slogan': 'characteristics slogan',
            'characteristics.group': 'characteristics group',
            'characteristics.color': 'characteristics color',
            'characteristics.skin_type': 'characteristics skin type',
            'characteristics.top_speed': 'characteristics top speed',
            'characteristics.lifespan': 'characteristics lifespan',
            'characteristics.weight': 'characteristics weight',
            'characteristics.length': 'characteristics length',
            'characteristics.age_of_sexual_maturity': 'characteristics age of sexual maturity',
            'characteristics.age_of_weaning': 'characteristics age of weaning',

            'characteristics.litter_size': 'characteristics litter size',
            'characteristics.predators': 'characteristics predators',
            'characteristics.type': 'characteristics type',
            'characteristics.height': 'characteristics height',
            'characteristics.distinctive_feature': 'characteristics distinctive feature',
            'characteristics.temperament': 'characteristics temperament',
            'characteristics.training': 'characteristics training',
            'characteristics.wingspan': 'characteristics wingspan',

            'characteristics.incubation_period': 'characteristics incubation period',
            'characteristics.age_of_fledgling': 'characteristics age of fledgling',
            'characteristics.average_clutch_size': 'characteristics average clutch size',
            'characteristics.venomous': 'characteristics venomous',
            'characteristics.aggression': 'characteristics aggression',
            'characteristics.main_prey': 'characteristics main prey',
            'characteristics.water_type': 'characteristics water type',
            'characteristics.age_of_independence': 'characteristics age of independence',
            'characteristics.average_spawn_size': 'characteristics average spawn size',
            'characteristics.nesting_location': 'characteristics nesting location',
            'characteristics.age_of_molting': 'characteristics age of molting',
            'characteristics.favorite_food': 'characteristics favorite food',
            'characteristics.origin': 'characteristics origin',
            'characteristics.migratory': 'characteristics migratory',
            'characteristics.optimum_ph_level': 'characteristics optimum ph level',
            'characteristics.special_features': 'characteristics special features',

        }

        if slot_name in slots_to_override:
            slot_name_verbalized = slots_to_override[slot_name]
        else:
            slot_name_verbalized = slot_name.replace('_', ' ')

        return slot_name_verbalized

    @staticmethod
    def get_single_word_slot_representation(slot_name):
        single_word_slot_repr = {
            'taxonomy.scientific_name': 'taxonomy scientific name',
            'taxonomy.kingdom': 'taxonomy kingdom',
            'taxonomy.phylum': 'taxonomy phylum',
            'taxonomy.class': 'taxonomy class',
            'taxonomy.order': 'taxonomy order',
            'taxonomy.genus': 'taxonomy genus',
            'characteristics.prey': 'characteristics prey',
            'characteristics.name_of_young': 'characteristics name of young',
            'characteristics.group_behavior': 'characteristics group behavior',
            'characteristics.estimated_population_size': 'characteristics estimated population size',
            'characteristics.biggest_threat': 'characteristics biggest threat',

            'characteristics.most_distinctive_feature': 'characteristics most distinctive feature',
            'characteristics.other_name(s)': 'characteristics other name(s)',
            'characteristics.gestation_period': 'characteristics gestation period',
            'characteristics.habitat': 'characteristics habitat',
            'characteristics.diet': 'characteristics diet',
            'characteristics.average_litter_size': 'characteristics average litter size',

            'characteristics.lifestyle': 'characteristics lifestyle',
            'characteristics.common_name': 'characteristics common name',
            'characteristics.number_of_species': 'characteristics number of species',
            'characteristics.location': 'characteristics location',
            'characteristics.slogan': 'characteristics slogan',
            'characteristics.group': 'characteristics group',
            'characteristics.color': 'characteristics color',
            'characteristics.skin_type': 'characteristics skin type',
            'characteristics.top_speed': 'characteristics top speed',
            'characteristics.lifespan': 'characteristics lifespan',
            'characteristics.weight': 'characteristics weight',
            'characteristics.length': 'characteristics length',
            'characteristics.age_of_sexual_maturity': 'characteristics age of sexual maturity',
            'characteristics.age_of_weaning': 'characteristics age of weaning',

            'characteristics.litter_size': 'characteristics litter size',
            'characteristics.predators': 'characteristics predators',
            'characteristics.type': 'characteristics type',
            'characteristics.height': 'characteristics height',
            'characteristics.distinctive_feature': 'characteristics distinctive feature',
            'characteristics.temperament': 'characteristics temperament',
            'characteristics.training': 'characteristics training',
            'characteristics.wingspan': 'characteristics wingspan',

            'characteristics.incubation_period': 'characteristics incubation period',
            'characteristics.age_of_fledgling': 'characteristics age of fledgling',
            'characteristics.average_clutch_size': 'characteristics average clutch size',
            'characteristics.venomous': 'characteristics venomous',
            'characteristics.aggression': 'characteristics aggression',
            'characteristics.main_prey': 'characteristics main prey',
            'characteristics.water_type': 'characteristics water type',
            'characteristics.age_of_independence': 'characteristics age of independence',
            'characteristics.average_spawn_size': 'characteristics average spawn size',
            'characteristics.nesting_location': 'characteristics nesting location',
            'characteristics.age_of_molting': 'characteristics age of molting',
            'characteristics.favorite_food': 'characteristics favorite food',
            'characteristics.origin': 'characteristics origin',
            'characteristics.migratory': 'characteristics migratory',
            'characteristics.optimum_ph_level': 'characteristics optimum ph level',
            'characteristics.special_features': 'characteristics special features',

        }

        return single_word_slot_repr.get(slot_name, slot_name)

    @staticmethod
    def get_slots_to_delexicalize():
        return {
            'simple': {'name', 'wikiID', 'taxonomy.kingdom', 'taxonomy.phylum', 'taxonomy.class','taxonomy.order','taxonomy.family','taxonomy.genus','characteristics.group_behavior',
                       'characteristics.gestation_period',
                       'characteristics.diet',
                       'characteristics.average_litter_size',
                       '',
                       '',
                       '',


                       },
            'list': {'taxonomy.scientific_name', 'locations', 'characteristics.prey', 'characteristics.name_of_young', 'characteristics.estimated_population_size',
                     'characteristics.biggest_threat','characteristics.most_distinctive_feature','characteristics.other_name(s)',
                     'characteristics.habitat',
                     'characteristics.lifestyle',
                     'characteristics.common_name',
                     '',
                     '',
                     '',

                     }
        }