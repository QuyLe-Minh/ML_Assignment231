import os
from architecture import BERT
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import yaml
from util import *

config = Config()
test_X, test_y = load_spam_dataset("spam_test.tsv")

def BERT_eval():
    model = BERT(config.model_name).to(config.device)
    model.load_state_dict(torch.load("BERT.pt"))
    print("Load successfully")

    tokenizer = BertTokenizer.from_pretrained(config.model_name)
    
    test_dataset = CustomDataset(test_X, test_y, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False, num_workers = 2)
    
    def val(model, data_loader, device):
        model.eval()
        predictions = []
        actual_labels = []
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                outputs = model(input_ids, attention_mask)
                _, preds = torch.max(outputs, dim=1)
                predictions.extend(preds.cpu().tolist())
                actual_labels.extend(labels.cpu().tolist())
    
        
        return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)
    
    acc, report = val(model, test_loader, config.device)
    print("BERT performance:")
    print(report)

def ML_eval(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
        
    svm_report = classification_report(data["svm"], test_y)
    bayes_report = classification_report(data["naive_bayes"], test_y)

    print("SVM performance:")
    print(svm_report)

    print("Naive Bayes performance:")
    print(bayes_report)

def evaluation():
    BERT_eval()
    ML_eval("predictions.yaml")

evaluation()
