import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import json

df = pd.read_csv('imdb_top_500.csv')
print(f"Dataset: {len(df)} reviews, columns: {df.columns.tolist()}")

df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].str.replace('<br />', ' ', regex=False)
df['text'] = df['text'].str.replace(r'<[^>]+>', ' ', regex=True)
df['label'] = df['label'].astype(int)
print(f"Label distribution:\n{df['label'].value_counts()}")

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2), stop_words='english')
X = vectorizer.fit_transform(df['text']).toarray().astype(np.float32)
y = df['label'].astype(np.float32).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train_t = torch.tensor(X_train)
y_train_t = torch.tensor(y_train).unsqueeze(1)
X_test_t  = torch.tensor(X_test)
y_test_t  = torch.tensor(y_test).unsqueeze(1)

class SentimentNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 64),        nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),          nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

model = SentimentNN(X_train.shape[1])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()
loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

model.train()
for epoch in range(15):
    total_loss = 0
    for xb, yb in loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/15 - Loss: {total_loss/len(loader):.4f}")

model.eval()
with torch.no_grad():
    preds = (model(X_test_t) > 0.5).float()
    acc = (preds == y_test_t).float().mean().item()
print(f"Test accuracy: {acc:.4f}")

torch.save(model.state_dict(), 'sentiment_model.pth')
joblib.dump(vectorizer, 'vectorizer.pkl')

with open('model_info.json', 'w') as f:
    json.dump({
        'accuracy': round(acc, 4),
        'input_dim': int(X_train.shape[1]),
        'architecture': 'TF-IDF(5000) -> Dense(256) -> Dense(64) -> Sigmoid',
        'dataset': 'IMDB Top 500',
        'labels': {'0': 'negative', '1': 'positive'}
    }, f, indent=2)

print("Saved: sentiment_model.pth, vectorizer.pkl, model_info.json")
