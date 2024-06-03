import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, BartForConditionalGeneration, BartTokenizer

def load_pegasus_model(device):
    model_ckpt = "google/pegasus-cnn_dailymail"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt, model_max_length=1024)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)
    return model, tokenizer

def load_bart_model(device):
    model_ckpt = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_ckpt, model_max_length=1024)
    model = BartForConditionalGeneration.from_pretrained(model_ckpt).to(device)
    return model, tokenizer
