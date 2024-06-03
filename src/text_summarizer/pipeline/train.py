import os
import pandas as pd
from datasets import load_metric, Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, BartForConditionalGeneration
import torch

def load_data(data_path):
    train_data = pd.read_csv(os.path.join(data_path, 'train_data.csv'))
    val_data = pd.read_csv(os.path.join(data_path, 'val_data.csv'))
    test_data = pd.read_csv(os.path.join(data_path, 'test_data.csv'))
    return train_data, val_data, test_data

def tokenize_data(data, tokenizer):
    return tokenizer(data, max_length=1024, truncation=True, padding='max_length', return_tensors='pt')

def create_datasets(tokenizer, train_data, val_data, test_data):
    train_dataset = Dataset.from_pandas(train_data)
    val_dataset = Dataset.from_pandas(val_data)
    test_dataset = Dataset.from_pandas(test_data)

    train_dataset = train_dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    test_dataset = test_dataset.map(lambda x: tokenize_data(x, tokenizer), batched=True)
    return train_dataset, val_dataset, test_dataset

def train_model(model_name, train_dataset, val_dataset, tokenizer, model_output_dir, epochs=1):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=model_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy='steps',
        eval_steps=100,
        save_steps=500,
        gradient_accumulation_steps=8
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )

    trainer.train()
    model.save_pretrained(model_output_dir)
    tokenizer.save_pretrained(model_output_dir)

def evaluate_model(model, tokenizer, test_dataset):
    metric = load_metric('rouge')

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    return compute_metrics

def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, batch_size=16, device='cpu', column_text="text", column_summary="summary"):
    def generate_batch_sized_chunks(list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i: i + batch_size]

    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in zip(article_batches, target_batches):
        inputs = tokenizer(article_batch, max_length=1024, truncation=True, padding="max_length", return_tensors="pt")
        summaries = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device), length_penalty=0.8, num_beams=8, max_length=128)
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True) for s in summaries]
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
    score = metric.compute()
    return score
