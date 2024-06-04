from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from text_summarizer.entity.configuration import ModelTrainerConfig
import torch
import os

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for model_ckpt in self.config.model_ckpts:
            tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
            seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
            
            dataset = load_from_disk(self.config.data_path)
            train_dataset = dataset["train"]
            val_dataset = dataset["validation"]

            trainer_args = TrainingArguments(
                output_dir=self.config.root_dir, num_train_epochs=self.config.num_train_epochs, warmup_steps=self.config.warmup_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size, per_device_eval_batch_size=self.config.per_device_train_batch_size,
                weight_decay=self.config.weight_decay, logging_steps=self.config.logging_steps,
                evaluation_strategy=self.config.evaluation_strategy, eval_steps=self.config.eval_steps, save_steps=self.config.save_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps
            )

            trainer = Trainer(model=model, args=trainer_args,
                              tokenizer=tokenizer, data_collator=seq2seq_data_collator,
                              train_dataset=train_dataset, 
                              eval_dataset=val_dataset)
            trainer.train()

            model_name = model_ckpt.split("/")[-1]
            model.save_pretrained(os.path.join(self.config.root_dir, f"{model_name}-model"))
            tokenizer.save_pretrained(os.path.join(self.config.root_dir, f"{model_name}-tokenizer"))
