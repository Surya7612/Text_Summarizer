from text_summarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        results = {}
        for model_path, tokenizer_path in zip(self.config.model_paths, self.config.tokenizer_paths):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model_name = model_path.split("/")[-1]
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
            summarization_pipeline = pipeline("summarization", model=model_path, tokenizer=tokenizer)
            summary = summarization_pipeline(text, **gen_kwargs)[0]["summary_text"]
            results[model_name] = summary
        return results
