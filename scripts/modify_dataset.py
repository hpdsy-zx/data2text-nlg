from collections import Counter
import json
import os
import pandas as pd
import re
import sys


def parse_da(mr, delimiters):
    # If no DA indication is expected in the data, return the MR unchanged
    if delimiters.get('da_beg') is None:
        return None

    # Verify if DA type is indicated at the beginning of the MR
    da_sep_idx = mr.find(delimiters['da_beg'])
    slot_sep_idx = mr.find(delimiters['slot_sep'])
    val_sep_idx = mr.find(delimiters['val_beg'])
    if da_sep_idx < 0 or 0 <= slot_sep_idx < da_sep_idx or 0 <= val_sep_idx < da_sep_idx:
        return None

    # Extract the DA type from the beginning of the MR
    return mr[:da_sep_idx]


def sample_rows(df_data, fraction):
    df_sampled = df_data.sample(frac=fraction)
    if len(df_sampled) < 1:
        df_sampled = df_data.sample(1)

    return df_sampled


def undersample_dataset(dataset_name, delimiters, fraction, trainset_only=False):
    if fraction <= 0.0 or fraction >= 1.0:
        print('Error: sample proportion must be greater than 0 and less than 1.')
        sys.exit()

    # Prepare the dataset file paths
    dataset_dir = os.path.join('..', 'data', dataset_name)
    if dataset_name == 'rest_e2e':
        data_files = ['trainset.csv']
        if not trainset_only:
            data_files.extend(['devset.csv', 'testset.csv'])
    elif dataset_name == 'video_game':
        data_files = ['train.csv']
        if not trainset_only:
            data_files.extend(['valid.csv', 'test.csv'])
    else:
        print(f'Error: dataset "{dataset_name}" not recognized')
        sys.exit()

    for file in data_files:
        # Load the data file
        df_data = pd.read_csv(os.path.join(dataset_dir, file), header=0)

        # Extract the DA types into a separate column
        df_data['da'] = df_data['mr'].apply(lambda x: parse_da(x, delimiters))

        # Sample the same fraction of rows from each DA type
        df_sampled = df_data.groupby('da', group_keys=False).apply(lambda x: sample_rows(x, fraction))
        df_sampled.drop('da', axis=1, inplace=True)

        # Compose the output file path
        out_file_name = os.path.splitext(file)[0] + f'_sampled_{fraction}.csv'
        out_file_path = os.path.join(dataset_dir, out_file_name)

        # Save to a CSV file (with UTF-8-BOM encoding)
        df_sampled.to_csv(out_file_path, index=False, encoding='utf-8-sig')


def convert_multiwoz_dataset_to_csv():
    data = {'train': [], 'valid': [], 'test': []}
    da_sep = ', '
    slot_sep = ', '

    dataset_dir = os.path.join('..', 'data', 'multiwoz')
    conversation_file = os.path.join(dataset_dir, 'data.json')
    validation_ids_file = os.path.join(dataset_dir, 'valListFile.txt')
    test_ids_file = os.path.join(dataset_dir, 'testListFile.txt')

    with open(conversation_file, 'r', encoding='utf-8') as f_conversations:
        conversations = json.load(f_conversations)
    with open(validation_ids_file, 'r', encoding='utf-8') as f_validation_ids:
        validation_ids = {line.strip() for line in f_validation_ids.readlines()}
    with open(test_ids_file, 'r', encoding='utf-8') as f_test_ids:
        test_ids = {line.strip() for line in f_test_ids.readlines()}

    invalid_mr_ctr = Counter()

    for conv_id, conv in conversations.items():
        conv_id_root = re.sub('\.json$', '', conv_id)

        for i, turn in enumerate(conv['log']):
            # Skip user turns
            # if not turn.get('metadata'):
            if i % 2 == 0:
                continue

            if turn.get('dialog_act'):
                if not isinstance(turn['dialog_act'], dict):
                    print(f'Warning: (system) turn #{i} of conversation {conv_id_root} not annotated.')

                    if conv_id in validation_ids:
                        partition = 'valid'
                    elif conv_id in test_ids:
                        partition = 'test'
                    else:
                        partition = 'train'
                    invalid_mr_ctr.update([partition])

                    continue

                mr = turn['dialog_act']
                utt = turn['text']
            else:
                print(f'Warning: MR for (system) turn #{i} of conversation {conv_id_root} not found.')

                if conv_id in validation_ids:
                    partition = 'valid'
                elif conv_id in test_ids:
                    partition = 'test'
                else:
                    partition = 'train'
                invalid_mr_ctr.update([partition])

                continue

            da_strings = []
            for da, slots in mr.items():
                slots_str = slot_sep.join(
                    ['{0}[{1}]'.format(re.sub(r'\s+', '_', slot[0].strip()), slot[1].strip()) for slot in slots
                     if slot[0] != 'none' or slot[1] != 'none'])
                da_strings.append('{0}({1})'.format(re.sub(r'\s+', '_', da.strip()), slots_str))

            mr_str = da_sep.join(da_strings)

            # Replace any sequence of whitespace characters with a single space in the utterance
            utt_processed = ' '.join(utt.split())

            if conv_id in validation_ids:
                partition = 'valid'
            elif conv_id in test_ids:
                partition = 'test'
            else:
                partition = 'train'
            data[partition].append((mr_str, utt_processed))

    print()
    print('>> Invalid MRs:')
    print(invalid_mr_ctr)

    for partition in ['train', 'valid', 'test']:
        df_data = pd.DataFrame(data[partition], columns=['mr', 'ref'])
        print(f'>> {partition} set size: {len(df_data)}')

        # Compose the output file path
        out_file_path = os.path.join(dataset_dir, partition + '.csv')

        # Save to a CSV file (with UTF-8-BOM encoding)
        df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')


def convert_rnnlg_dataset_to_csv(dataset_domain_name):
    dataset_dir = os.path.join('data', dataset_domain_name)

    for file_name in ['train.json', 'valid.json', 'test.json']:
        data_file = os.path.join(dataset_dir, file_name)
        with open(data_file, 'r', encoding='utf-8') as f_data:
            # Remove the comment at the beginning of the file
            for i in range(5):
                f_data.readline()

            # Read the rest of the file in JSON format into a DataFrame
            df_data = pd.read_json(f_data, encoding='utf-8')

        # Set the column names
        df_data.columns = ['mr', 'ref', 'baseline']

        # Compose the output file path and save to a CSV file (with UTF-8-BOM encoding)
        out_file_name = os.path.splitext(file_name)[0] + '.csv'
        out_file_path = os.path.join(dataset_dir, out_file_name)
        df_data.to_csv(out_file_path, index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    # dataset_name = 'rest_e2e'
    # delimiters = {
    #     'da_beg': None,
    #     'da_end': None,
    #     'slot_sep': ', ',
    #     'val_beg': '[',
    #     'val_end': ']'
    # }

    # dataset_name = 'video_game'
    # delimiters = {
    #     'da_beg': '(',
    #     'da_end': ')',
    #     'slot_sep': ', ',
    #     'val_beg': '[',
    #     'val_end': ']'
    # }
    #
    # undersample_dataset(dataset_name, delimiters, 0.5, trainset_only=True)

    # convert_multiwoz_dataset_to_csv()

    # rnnlg_domain = 'laptop_rnnlg'
    rnnlg_domain = 'tv_rnnlg'
    convert_rnnlg_dataset_to_csv(rnnlg_domain)
