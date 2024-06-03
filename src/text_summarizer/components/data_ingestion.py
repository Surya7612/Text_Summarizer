import os
from text_summarizer.logging.logging import logger
from text_summarizer.utils.common import load_datasets, standardize_column_names, concatenate_splits
from text_summarizer.entity.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_datasets(self):
        datasets_info = []
        for name, config, trust_code in zip(self.config.dataset_names, self.config.configs, self.config.trust_remote_code):
            datasets_info.append({
                "name": name,
                "config": config,
                "trust_remote_code": trust_code
            })
        
        datasets = load_datasets(datasets_info)
        column_mappings = [
            {'article': 'text', 'highlights': 'summary'},
            {'document': 'text', 'summary': 'summary'},
            {'document': 'text', 'summary': 'summary'},
            {'document': 'text', 'summary': 'summary'},
            {'description': 'text', 'abstract': 'summary'},
            {'dialogue': 'text', 'summary': 'summary'},
        ]
        self.datasets = standardize_column_names(datasets, column_mappings)

    def concatenate_splits(self):
        train_dataset, val_dataset, test_dataset = concatenate_splits(self.datasets)
        os.makedirs(self.config.concatenated_data_dir, exist_ok=True)
        train_dataset.save_to_disk(os.path.join(self.config.concatenated_data_dir, "train"))
        val_dataset.save_to_disk(os.path.join(self.config.concatenated_data_dir, "validation"))
        test_dataset.save_to_disk(os.path.join(self.config.concatenated_data_dir, "test"))

