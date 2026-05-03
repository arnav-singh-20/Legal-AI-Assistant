"""
Lawgorithm — BERT Legal Query Classifier
Trains two classification heads on top of BERT:
  1. Domain: criminal, civil, property, family, tax
  2. Intent: needs_lawyer, self_solvable, urgent, general_info

FIX: Separate weighted loss for both domain AND intent
     to handle class imbalance properly.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import pickle
import os

# ─── CONFIG ────────────────────────────────────────────────
BERT_MODEL  = "bert-base-uncased"
MAX_LEN     = 128
BATCH_SIZE  = 16
EPOCHS      = 5
LR          = 2e-5
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FIX 0: Use relative paths — works on any machine, not just your Mac
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR    = os.path.join(BASE_DIR, "saved_model")
DATA_PATH   = os.path.join(BASE_DIR, "..", "data", "legal_queries.csv")

print(f"Using device: {DEVICE}")
os.makedirs(SAVE_DIR, exist_ok=True)

# ─── DATASET ───────────────────────────────────────────────
class LegalDataset(Dataset):
    def __init__(self, texts, domain_labels, intent_labels, tokenizer, max_len):
        self.texts         = texts
        self.domain_labels = domain_labels
        self.intent_labels = intent_labels
        self.tokenizer     = tokenizer
        self.max_len       = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'domain_label':   torch.tensor(self.domain_labels[idx], dtype=torch.long),
            'intent_label':   torch.tensor(self.intent_labels[idx], dtype=torch.long),
        }

# ─── MODEL ─────────────────────────────────────────────────
class LawgorithmBERT(nn.Module):
    """
    BERT with two classification heads:
    - domain_head: 5 classes (criminal, civil, property, family, tax)
    - intent_head: 4 classes (needs_lawyer, self_solvable, urgent, general_info)
    """
    def __init__(self, n_domains, n_intents, dropout=0.3):
        super().__init__()
        self.bert    = BertModel.from_pretrained(BERT_MODEL)
        hidden       = self.bert.config.hidden_size  # 768
        self.dropout = nn.Dropout(dropout)

        self.domain_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_domains)
        )
        self.intent_head = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_intents)
        )

    def forward(self, input_ids, attention_mask):
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled     = outputs.pooler_output   # [CLS] token — shape: [batch, 768]
        pooled     = self.dropout(pooled)
        domain_out = self.domain_head(pooled)
        intent_out = self.intent_head(pooled)
        return domain_out, intent_out

# ─── LOAD DATA ─────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} samples")
print(f"Domain distribution: {dict(Counter(df['domain']))}")
print(f"Intent distribution: {dict(Counter(df['intent']))}")

domain_enc = LabelEncoder()
intent_enc = LabelEncoder()
df['domain_id'] = domain_enc.fit_transform(df['domain'])
df['intent_id'] = intent_enc.fit_transform(df['intent'])

with open(f"{SAVE_DIR}/domain_encoder.pkl", "wb") as f:
    pickle.dump(domain_enc, f)
with open(f"{SAVE_DIR}/intent_encoder.pkl", "wb") as f:
    pickle.dump(intent_enc, f)

print(f"Domain mapping: {dict(zip(domain_enc.classes_, range(len(domain_enc.classes_))))}")
print(f"Intent mapping: {dict(zip(intent_enc.classes_, range(len(intent_enc.classes_))))}")

X_train, X_test, y_dom_train, y_dom_test, y_int_train, y_int_test = train_test_split(
    df['query'].tolist(),
    df['domain_id'].tolist(),
    df['intent_id'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['domain_id']
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─── CLASS WEIGHTS ─────────────────────────────────────────
# FIX 1: Compute weights for DOMAIN (criminal 643 vs tax 55 = 11x imbalance)
domain_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_dom_train),
    y=y_dom_train
)
domain_weight_tensor = torch.FloatTensor(domain_weights).to(DEVICE)
print(f"Domain class weights: {dict(zip(domain_enc.classes_, domain_weights.round(2)))}")

# FIX 2: Compute weights for INTENT (urgent 41 vs general_info 665 = 16x imbalance)
intent_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_int_train),
    y=y_int_train
)
intent_weight_tensor = torch.FloatTensor(intent_weights).to(DEVICE)
print(f"Intent class weights: {dict(zip(intent_enc.classes_, intent_weights.round(2)))}")

# ─── DATALOADERS ───────────────────────────────────────────
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

train_dataset = LegalDataset(X_train, y_dom_train, y_int_train, tokenizer, MAX_LEN)
test_dataset  = LegalDataset(X_test,  y_dom_test,  y_int_test,  tokenizer, MAX_LEN)

train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader   = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

# ─── TRAIN ─────────────────────────────────────────────────
n_domains = len(domain_enc.classes_)
n_intents = len(intent_enc.classes_)

model     = LawgorithmBERT(n_domains, n_intents).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

# FIX 3: Separate criterion for each head with correct weights
# OLD BUG: one criterion with intent weights used for BOTH domain and intent loss
# This was applying intent class weights to domain predictions — completely wrong
domain_criterion = nn.CrossEntropyLoss(weight=domain_weight_tensor)
intent_criterion = nn.CrossEntropyLoss(weight=intent_weight_tensor)

total_steps = len(train_loader) * EPOCHS
scheduler   = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.1, total_iters=total_steps
)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    dom_correct = int_correct = total = 0

    for batch in train_loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        domain_labels  = batch['domain_label'].to(DEVICE)
        intent_labels  = batch['intent_label'].to(DEVICE)

        optimizer.zero_grad()

        domain_logits, intent_logits = model(input_ids, attention_mask)

        # FIX 4: Each head gets its own correctly weighted loss
        domain_loss = domain_criterion(domain_logits, domain_labels)
        intent_loss = intent_criterion(intent_logits, intent_labels)
        loss        = domain_loss + intent_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss  += loss.item()
        dom_correct += (domain_logits.argmax(1) == domain_labels).sum().item()
        int_correct += (intent_logits.argmax(1) == intent_labels).sum().item()
        total       += len(domain_labels)

    train_dom_acc = dom_correct / total * 100
    train_int_acc = int_correct / total * 100

    model.eval()
    val_dom_correct = val_int_correct = val_total = 0

    with torch.no_grad():
        for batch in test_loader:
            input_ids      = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            domain_labels  = batch['domain_label'].to(DEVICE)
            intent_labels  = batch['intent_label'].to(DEVICE)

            domain_logits, intent_logits = model(input_ids, attention_mask)

            val_dom_correct += (domain_logits.argmax(1) == domain_labels).sum().item()
            val_int_correct += (intent_logits.argmax(1) == intent_labels).sum().item()
            val_total       += len(domain_labels)

    val_dom_acc = val_dom_correct / val_total * 100
    val_int_acc = val_int_correct / val_total * 100
    avg_val_acc = (val_dom_acc + val_int_acc) / 2

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | "
          f"Train Dom: {train_dom_acc:.1f}% Int: {train_int_acc:.1f}% | "
          f"Val Dom: {val_dom_acc:.1f}% Int: {val_int_acc:.1f}%")

    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pt")
        print(f"  ✓ Saved best model (avg val acc: {avg_val_acc:.1f}%)")

# ─── FINAL EVALUATION ──────────────────────────────────────
print("\n─── Final Evaluation ───")
model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pt", map_location=DEVICE))
model.eval()

all_dom_true, all_dom_pred = [], []
all_int_true, all_int_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        input_ids      = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)

        domain_logits, intent_logits = model(input_ids, attention_mask)

        all_dom_pred.extend(domain_logits.argmax(1).cpu().numpy())
        all_dom_true.extend(batch['domain_label'].numpy())
        all_int_pred.extend(intent_logits.argmax(1).cpu().numpy())
        all_int_true.extend(batch['intent_label'].numpy())

print("\nDomain Classification Report:")
print(classification_report(all_dom_true, all_dom_pred,
      target_names=domain_enc.classes_))

print("\nIntent Classification Report:")
print(classification_report(all_int_true, all_int_pred,
      target_names=intent_enc.classes_))

tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel saved to {SAVE_DIR}/")
print("Training complete!")
