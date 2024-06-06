# SummarizeMaster

SummarizeMaster is a Python package for text summarization using state-of-the-art models like PEGASUS and BART. The package integrates multiple datasets from Hugging Face and provides functionalities for training, evaluation, and prediction.

## Features

- Utilizes PEGASUS and BART models for summarization.
- Integrates multiple datasets including CNN/DailyMail, XSum, Multi-News, Gigaword, BigPatent, and SAMSum.
- Provides functionalities for training, evaluation, and prediction.
- Supports cloud storage for datasets to handle large data efficiently.
- Implements a FastAPI application for serving the summarization model.

## Installation

1. Clone the repository:
    ```bash
    https://github.com/Surya7612/Text_Summarizer.git
    cd summarize-master
    ```

2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Training the Models

To train the models, run:
```bash
python main.py
```
This will execute the data ingestion, data validation, data transformation, model training, and model evaluation pipelines.

## Running the FastAPI Application
To start the FastAPI server, run:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080
```

Navigate to http://localhost:8080/docs to access the API documentation and test the endpoints.

## Making Predictions
You can use the /predict endpoint to make predictions. Send a POST request with the text to summarize:
```bash
curl -X POST "http://localhost:8080/predict" -H "Content-Type: application/json" -d '{"text": "Your text here"}'
```
## Version 0.1
This is version 0.1 of SummarizeMaster. In this version, we have:

- Used only 1 epoch for training and validation.
- Utilized a subset of the data due to computational restrictions.
Future versions will involve stronger models, more extensive data utilization, and improved training configurations.

## Project Structure
```bash
summarize-master/
├── .github/
│   └── workflows/
│       └── main.yaml
├── artifacts/
│   └── ...
├── config/
│   └── config.yaml
├── src/
│   └── text_summarizer/
│       ├── __init__.py
│       ├── components/
│       │   ├── __init__.py
│       │   ├── data_ingestion.py
│       │   ├── data_transformation.py
│       │   ├── data_validation.py
│       │   ├── model_evaluation.py
│       │   └── model_trainer.py
│       ├── entity/
│       │   ├── __init__.py
│       │   └── configuration.py
│       ├── logging/
│       │   └── __init__.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── stage_01_data_ingestion.py
│       │   ├── stage_02_data_validation.py
│       │   ├── stage_03_data_transformation.py
│       │   ├── stage_04_model_trainer.py
│       │   └── stage_05_model_evaluation.py
│       └── utils/
│           └── common.py
├── app.py
├── Dockerfile
├── main.py
├── README.md
├── requirements.txt
└── setup.py
```

## Configuration
The configuration file config/config.yaml contains all the necessary configurations for data ingestion, validation, transformation, model training, and evaluation.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for review.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
```bash
This README provides a comprehensive overview of the "SummarizeMaster" project, guiding users through installation, usage, and understanding the project structure.
```

# Work still in progress...
