from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk, load_metric
import torch
import os
import pandas as pd
from tqdm import tqdm
from text_summarizer.entity.configuration import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, batch_size=16, device="cuda" if torch.cuda.is_available() else "cpu", column_text="text", column_summary="summary"):
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))
        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
            summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), length_penalty=0.8, num_beams=8, max_length=128)
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
            decoded_summaries = [d.replace("", " ") for d in decoded_summaries]
            metric.add_batch(predictions=decoded_summaries, references=target_batches)
        score = metric.compute()
        return score

    def evaluate(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dataset = load_from_disk(self.config.data_path)
        rouge_metric = load_metric('rouge')
        scores = {}

        for model_path, tokenizer_path in zip(self.config.model_paths, self.config.tokenizer_paths):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
            score = self.calculate_metric_on_test_ds(dataset['test'], rouge_metric, model, tokenizer, batch_size=2, column_text='text', column_summary='summary')
            model_name = model_path.split("/")[-1]
            scores[model_name] = score

        for model_name, score in scores.items():
            rouge_dict = dict((rn, score[rn].mid.fmeasure) for rn in ["rouge1", "rouge2", "rougeL", "rougeLsum"])
            df = pd.DataFrame(rouge_dict, index=[model_name])
            df.to_csv(os.path.join(self.config.root_dir, f"{model_name}_metrics.csv"))
