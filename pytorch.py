import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split

# Шаг 1: Загрузка данных из CSV
df = pd.read_csv('имена.csv')

names = df['Имя'].tolist()
part1 = df['Составляющее 1'].tolist()
part2 = df['Составляющее 2'].tolist()
genders = df['Пол'].tolist()

# Шаг 2: Предобработка данных
def tokenize(data):
    """Функция для токенизации строк"""
    tokens = []
    for item in data:
        if isinstance(item, str):
            tokens.extend(list(item))
        elif isinstance(item, float) or isinstance(item, int):
            tokens.extend(list(str(item)))  # Преобразуем число в строку и токенизируем
        else:
            raise ValueError(f"Неизвестный тип данных: {type(item)}")
    return tokens

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Токенизируем все данные
name_tokens = tokenize(names)
part1_tokens = tokenize(part1)
part2_tokens = tokenize(part2)
gender_labels = genders

# Создаем словарь для токенов
char_to_idx = {'<PAD>': 0, '<SOS>': 1}  # Добавляем специальные маркеры
char_to_idx.update({char: idx+2 for idx, char in enumerate(set(name_tokens))})
idx_to_char = {v: k for k, v in char_to_idx.items()}

# Преобразуем строки в индексы
def to_indices(tokens, char_to_idx):
    indices = [[char_to_idx.get(token, 0) for token in list(str(item))] for item in tokens]
    return indices

name_indices = to_indices(names, char_to_idx)
part1_indices = to_indices(part1, char_to_idx)
part2_indices = to_indices(part2, char_to_idx)

# Кодируем пол
gender_map = {'Мужской': 0, 'Женский': 1}
gender_indices = [gender_map[g] for g in gender_labels]

# Приведение всех списков к одинаковой длине
max_len = max(max(len(s) for s in name_indices),
              max(len(s) for s in part1_indices),
              max(len(s) for s in part2_indices))

name_indices_padded = np.array([seq + [0] * (max_len - len(seq)) for seq in name_indices])

# Выравнивание длин для part1 и part2
part1_indices_padded = []
part2_indices_padded = []
for seq in part1_indices:
    padded_seq = [char_to_idx['<SOS>']] + seq + [0] * (max_len - len(seq) - 1)
    part1_indices_padded.append(padded_seq)

for seq in part2_indices:
    padded_seq = [char_to_idx['<SOS>']] + seq + [0] * (max_len - len(seq) - 1)
    part2_indices_padded.append(padded_seq)

# Преобразование в массивы
part1_indices_padded = np.array(part1_indices_padded, dtype="object")
part2_indices_padded = np.array(part2_indices_padded, dtype=np.float16)

gender_indices_onehot = np.eye(2)[gender_indices]

# Шаг 3: Создание модели
class NameAnalyzer(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1):
        super(NameAnalyzer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        # Выходные слои для каждой целевой переменной
        self.part1_output = nn.Linear(hidden_dim, vocab_size)
        self.part2_output = nn.Linear(hidden_dim, vocab_size)
        self.gender_output = nn.Linear(hidden_dim, 2)  # 2 класса: мужской/женский

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_embedded)
        unpacked_lstm_out, unpacked_lengths = pad_packed_sequence(lstm_out, batch_first=True)

        part1_logits = self.part1_output(unpacked_lstm_out[:, -1, :])
        part2_logits = self.part2_output(unpacked_lstm_out[:, -1, :])
        gender_logits = self.gender_output(unpacked_lstm_out[:, -1, :])  # Используем последний временной шаг

        return part1_logits, part2_logits, gender_logits

# Параметры модели
vocab_size = len(char_to_idx) + 1  # Добавляем 1 для неизвестных символов
embedding_dim = 32
hidden_dim = 64
num_layers = 1

model = NameAnalyzer(vocab_size, embedding_dim, hidden_dim, num_layers)
criterion_part1 = nn.CrossEntropyLoss()
criterion_part2 = nn.CrossEntropyLoss()
criterion_gender = nn.BCEWithLogitsLoss()  # Для бинарной классификации пола

optimizer = torch.optim.Adam(model.parameters())

# Шаг 4: Обучение модели
num_epochs = 10

for epoch in range(num_epochs):
    for i in range(len(name_indices)):
        optimizer.zero_grad()

        input_seq = torch.LongTensor(name_indices_padded[i]).unsqueeze(0)
        target_part1 = torch.LongTensor(part1_indices_padded[i]).unsqueeze(0)
        target_part2 = torch.LongTensor(part2_indices_padded[i]).unsqueeze(0)
        target_gender = torch.FloatTensor(gender_indices_onehot[i]).unsqueeze(0)

        lengths = torch.LongTensor([input_seq.size(1)])

        outputs = model(input_seq, lengths)

        #loss_part1 = criterion_part1(outputs[0], target_part1.squeeze(0))
        #loss_part2 = criterion_part2(outputs[1], target_part2.squeeze().argmax(dim=-1))
        loss_gender = criterion_gender(outputs[2].squeeze(), target_gender.squeeze())

        total_loss =  loss_gender
        total_loss.backward()
        optimizer.step()

    print(f'Эпоха {epoch+1}/{num_epochs}, Потери: {total_loss.item():.4f}')

# Шаг 5: Тестирование модели
def predict(model, input_name):
    with torch.no_grad():
        input_indices = to_indices([input_name], char_to_idx)[0]
        input_seq = torch.LongTensor(input_indices).unsqueeze(0)
        lengths = torch.LongTensor([input_seq.size(1)])

        outputs = model(input_seq, lengths)
        part1_pred = idx_to_char[outputs[0].argmax(dim=-1).item()]
        part2_pred = idx_to_char[outputs[1].argmax(dim=-1).item()]
        gender_index = outputs[2].argmax(dim=-1)
        gender_pred = 'Мужской' if gender_index == 0 else 'Женский'
        predicted_parts = (part1_pred, part2_pred)
        return predicted_parts, gender_pred

# Пример использования
predicted_parts, predicted_gender = predict(model, 'Кецекуатль')
print(f"Предсказанные части имени: {predicted_parts}")
print(f"Пол: {predicted_gender}")
