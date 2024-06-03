from text_summarizer.logging import logger

logger.info("This is an info message, Welcome to our custom logging")
from text_summarizer.config.configuration import load_config
from text_summarizer.utils.common import load_datasets, standardize_column_names, concatenate_splits, preprocess_data, create_directories
from text_summarizer.pipeline.train import train_model, evaluate_model, calculate_metric_on_test_ds
from text_summarizer.components.model import load_pegasus_model, load_bart_model
from text_summarizer.constants.constants import PEGASUS_MODEL_CKPT, BART_MODEL_CKPT
from transformers import AutoTokenizer, BartTokenizer
import torch
import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration
from datasets import load_metric

def main():
    # Load config
    config = load_config()

    # Create necessary directories
    create_directories([config.artifacts.data.processed_data_path, config.artifacts.models.pegasus_model_dir, config.artifacts.models.bart_model_dir])

    # Load datasets
    datasets_info = [
        {"name": "cnn_dailymail", "config": "3.0.0", "trust_remote_code": False},
        {"name": "xsum", "config": None, "trust_remote_code": True},
        {"name": "multi_news", "config": None, "trust_remote_code": True},
        {"name": "gigaword", "config": None, "trust_remote_code": True},
        {"name": "big_patent", "config": "a", "trust_remote_code": True},
        {"name": "samsum", "config": None, "trust_remote_code": True},
    ]

    datasets = load_datasets(datasets_info, config.artifacts.data.processed_data_path)
    column_mappings = [
        {'article': 'text', 'highlights': 'summary'},
        {'document': 'text', 'summary': 'summary'},
        {'document': 'text', 'summary': 'summary'},
        {'document': 'text', 'summary': 'summary'},
        {'description': 'text', 'abstract': 'summary'},
        {'dialogue': 'text', 'summary': 'summary'}
    ]
    datasets = standardize_column_names(datasets, column_mappings)
    train_dataset, val_dataset, test_dataset = concatenate_splits(datasets)

    # Train PEGASUS model
    pegasus_tokenizer = AutoTokenizer.from_pretrained(PEGASUS_MODEL_CKPT, model_max_length=1024)
    train_data_pegasus = preprocess_data(train_dataset, pegasus_tokenizer)
    val_data_pegasus = preprocess_data(val_dataset, pegasus_tokenizer)
    test_data_pegasus = preprocess_data(test_dataset, pegasus_tokenizer)
    train_model(PEGASUS_MODEL_CKPT, train_data_pegasus, val_data_pegasus, pegasus_tokenizer, config.artifacts.models.pegasus_model_dir)

    # Train BART model
    bart_tokenizer = BartTokenizer.from_pretrained(BART_MODEL_CKPT, model_max_length=1024)
    train_data_bart = preprocess_data(train_dataset, bart_tokenizer)
    val_data_bart = preprocess_data(val_dataset, bart_tokenizer)
    test_data_bart = preprocess_data(test_dataset, bart_tokenizer)
    train_model(BART_MODEL_CKPT, train_data_bart, val_data_bart, bart_tokenizer, config.artifacts.models.bart_model_dir)

    # Evaluate PEGASUS model
    pegasus_model = AutoModelForSeq2SeqLM.from_pretrained(config.artifacts.models.pegasus_model_dir)
    pegasus_score = calculate_metric_on_test_ds(test_data_pegasus, load_metric('rouge'), pegasus_model, pegasus_tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(pd.DataFrame(dict((rn, pegasus_score[rn].mid.fmeasure) for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"]), index=['PEGASUS']))

    # Evaluate BART model
    bart_model = BartForConditionalGeneration.from_pretrained(config.artifacts.models.bart_model_dir)
    bart_score = calculate_metric_on_test_ds(test_data_bart, load_metric('rouge'), bart_model, bart_tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu')
    print(pd.DataFrame(dict((rn, bart_score[rn].mid.fmeasure) for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"]), index=['BART']))

if __name__ == "__main__":
    main()
