import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report
from architecture import BERT
from util import *
from torch.optim.lr_scheduler import ReduceLROnPlateau

config = Config()

train_X, train_y = load_spam_dataset("BERT/spam_train.tsv")
val_X, val_y = load_spam_dataset("BERT/spam_val.tsv")
test_X, test_y = load_spam_dataset("BERT/spam_test.tsv")

tokenizer = BertTokenizer.from_pretrained(config.model_name)

train_dataset = CustomDataset(train_X, train_y, tokenizer)
test_dataset = CustomDataset(test_X, test_y, tokenizer)

train_loader = DataLoader(train_dataset, batch_size = config.batch_size, shuffle = True, num_workers = 2)
test_loader = DataLoader(test_dataset, batch_size = config.batch_size, shuffle = False, num_workers = 2)

class BERT(nn.Module):
    def __init__(self, bert_model_name, num_classes = 2):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    
def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    size = len(data_loader.dataset)
    for (i, batch) in enumerate(data_loader):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids, attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i%100 == 0:
          loss, curr = loss.item(), i * config.batch_size
          print(f"loss: {loss:>7f}  [{curr:>5d}/{size:>5d}]")

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

model = BERT(config.model_name).to(config.device)
optimizer = AdamW(model.parameters(), lr=1e-5)
steps = len(train_loader) * config.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=steps)

best_one = 0.99821
count = 0
for t in range(config.epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_loader, optimizer, scheduler, config.device)
    acc, _ = val(model, test_loader, config.device)
    print(f"Accuracy: {(100 * acc):>0.3f}%")

    if acc > best_one:
      count = 0
      best_one = acc
      torch.save(model.state_dict(), "BERT/BERT.pt")
    else:
        count+=1
        if count == config.patience:
            break