import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# Step 1: Load and Preprocess Data
file_path = '/mnt/data/ecommerce_reviews.csv'
data = pd.read_csv(file_path)

# Drop rows with missing reviews and map ratings to sentiment
data = data.dropna(subset=['Review Text'])
data['Sentiment'] = data['Rating'].apply(lambda x: "positive" if x >= 4 else "negative" if x <= 2 else "neutral")

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['Review Text'], data['Sentiment'], test_size=0.2, stratify=data['Sentiment'], random_state=42
)

# Step 2: Tokenization
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ReviewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(list(texts), truncation=True, padding=True, max_length=128, return_tensors="pt")
        self.labels = [0 if l == "negative" else 1 if l == "neutral" else 2 for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

train_dataset = ReviewsDataset(train_texts, train_labels, tokenizer)
val_dataset = ReviewsDataset(val_texts, val_labels, tokenizer)

# Step 3: Load BERT Model
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Step 4: Define Trainer and Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Step 5: Train and Evaluate
trainer.train()

# Save model for future use
model.save_pretrained("./sentiment_model")
tokenizer.save_pretrained("./sentiment_model")

# (Optional) Evaluate
eval_results = trainer.evaluate()
print("Evaluation Results:", eval_results)
