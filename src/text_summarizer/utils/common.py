# import os
# from box.exceptions import BoxValueError
# import yaml
# from textSummarizer.logging import logger
# from ensure import ensure_annotations
# from box import ConfigBox
# from pathlib import Path
# from typing import Any

# @ensure_annotations
# def read_yaml(path_to_yaml: Path) -> ConfigBox:
#     """reads yaml file and returns
    
#     Args:
#         path_to_yaml (Path): path to the yaml file/input path

#     Raises:
#         ValueError: if yaml file is empty
#         e: empty file

#     Returns:
#         ConfigBox: ConfigBox type
#     """

#     try:
#         with open(path_to_yaml) as yaml_file:
#             content = yaml.safe_load(yaml_file)
#             logger.info(f"Successfully loaded yaml file: {path_to_yaml}")
#             return ConfigBox(content)
#     except BoxValueError:
#         raise ValueError(f"{path_to_yaml} is empty")
#     except Exception as e:
#         raise e

# @ensure_annotations
# def create_directories(path_to_directories: list, verbose=True):
#     '''create list of directories
    
#     Args:
#         path_to_directories (list): list of path of directories to be created
#         ignore_log (bool, optional): ignore if multiple directories is to be created. Defaults to False.
#     '''
#     for path in path_to_directories:
#         os.makedirs(path, exist_ok=True)
#         if verbose:
#             logger.info(f"Successfully created directory at: {path}")

# @ensure_annotations
# def get_size(path: Path) -> str:
#     '''get size in KB
    
#     Args:
#         path (Path): path to the file
        
#         Returns:
#             str: size of the file in KB
#     '''

#     size_in_kb = round(os.path.getsize(path) / 1024)
#     return f"~{size_in_kb} KB"
import os
from box.exceptions import BoxValueError
import yaml
from text_summarizer.logging.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
from datasets import load_dataset, concatenate_datasets
import pandas as pd

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns
    
    Args:
        path_to_yaml (Path): path to the yaml file/input path

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """

    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"Successfully loaded yaml file: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"{path_to_yaml} is empty")
    except Exception as e:
        raise e

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    '''create list of directories
    
    Args:
        path_to_directories (list): list of path of directories to be created
        ignore_log (bool, optional): ignore if multiple directories is to be created. Defaults to False.
    '''
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Successfully created directory at: {path}")

@ensure_annotations
def get_size(path: Path) -> str:
    '''get size in KB
    
    Args:
        path (Path): path to the file
        
        Returns:
            str: size of the file in KB
    '''

    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~{size_in_kb} KB"

def process_and_save_dataset(dataset, split, save_path):
    df = pd.DataFrame(dataset[split])
    df.to_csv(save_path, index=False)

def load_datasets(datasets_info, processed_data_path):
    datasets = []
    for info in datasets_info:
        name = info['name']
        config = info.get('config', None)
        trust_remote_code = info.get('trust_remote_code', False)
        dataset = load_dataset(name, config, trust_remote_code=trust_remote_code) if config else load_dataset(name, trust_remote_code=trust_remote_code)
        
        os.makedirs(processed_data_path, exist_ok=True)
        process_and_save_dataset(dataset, 'train', f'{processed_data_path}/{name}_train.csv')
        process_and_save_dataset(dataset, 'validation', f'{processed_data_path}/{name}_validation.csv')
        process_and_save_dataset(dataset, 'test', f'{processed_data_path}/{name}_test.csv')
        datasets.append(dataset)
    return datasets

def standardize_column_names(datasets, column_mappings):
    for i, dataset in enumerate(datasets):
        for split in dataset.keys():
            datasets[i][split] = dataset[split].rename_columns(column_mappings[i])
    return datasets

def concatenate_splits(datasets):
    train_datasets = [dataset['train'].select(range(1000)) for dataset in datasets if 'train' in dataset]
    val_datasets = [dataset['validation'].select(range(100)) for dataset in datasets if 'validation' in dataset]
    test_datasets = [dataset['test'].select(range(100)) for dataset in datasets if 'test' in dataset]
    return concatenate_datasets(train_datasets), concatenate_datasets(val_datasets), concatenate_datasets(test_datasets)

def convert_examples_to_features(example_batch, tokenizer):
    input_encodings = tokenizer(example_batch['text'], max_length=1024, truncation=True)
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['summary'], max_length=128, truncation=True)
    return {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }

def preprocess_data(dataset, tokenizer):
    return dataset.map(lambda x: convert_examples_to_features(x, tokenizer), batched=True)
