import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Чтение данных из CSV файла
df = pd.read_csv('имена.csv')

class NameDataset(Dataset):
    def __init__(self, df):
        self.df = df  # Сохраняем DataFrame как атрибут класса
        self.words = list(set(df['Составляющее 1'].tolist() + df['Составляющее 2'].tolist()))
        self.names = list(set(df['Имя'].tolist()))

        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        self.name_to_idx = {name: idx for idx, name in enumerate(self.names)}
        self.idx_to_name = {idx: name for name, idx in self.name_to_idx.items()}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        word1 = self.word_to_idx[row['Составляющее 1']]
        word2 = self.word_to_idx[row['Составляющее 2']]
        name_idx = self.name_to_idx[row['Имя']]
        return torch.tensor([word1, word2]), torch.tensor(name_idx)

class NameGenerator(nn.Module):
    def __init__(self, vocab_size, name_size, embedding_dim, hidden_dim):
        super(NameGenerator, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, name_size)

    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        output = self.fc(lstm_out[:, -1, :])
        return output

# Параметры
embedding_dim = 128
hidden_dim = 128
num_epochs = 50
batch_size = 4
learning_rate = 0.001

# Создание набора данных и загрузчика
dataset = NameDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Создание модели
model = NameGenerator(len(dataset.words), len(dataset.names), embedding_dim, hidden_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Обучение модели
for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Функция генерации
def generate_name(word1, word2, model, dataset):
    word1_idx = dataset.word_to_idx.get(word1, None)
    word2_idx = dataset.word_to_idx.get(word2, None)

    if word1_idx is None or word2_idx is None:
        raise ValueError("Одно или оба слова не найдены в наборе данных.")

    input_tensor = torch.tensor([[word1_idx, word2_idx]])
    with torch.no_grad():
        output = model(input_tensor)
        output_idx = torch.argmax(output, dim=1).item()

    return dataset.idx_to_name[output_idx]

# Пример использования
word1 = "Любовь"
word2 = "Слава"
try:
    generated_name = generate_name(word1, word2, model, dataset)
    print(f'Сгенерированное имя: {generated_name}')
except ValueError as e:
    print(e)
