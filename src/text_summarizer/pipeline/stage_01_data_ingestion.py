from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.components.data_ingestion import DataIngestion
from text_summarizer.logging.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_datasets()
        data_ingestion.concatenate_datasets()
