from text_summarizer.config.configuration import load_config
from text_summarizer.utils.common import create_directories
from text_summarizer.pipeline.train import train_model, evaluate_model, calculate_metric_on_test_ds
from text_summarizer.components.model import load_pegasus_model, load_bart_model

def main():
    config = load_config()
    create_directories([config.artifacts.data.processed_data_path, config.artifacts.models.pegasus_model_dir, config.artifacts.models.bart_model_dir])
    # Add the necessary pipeline logic here

if __name__ == "__main__":
    main()
