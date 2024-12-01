import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Шаг 1: Загрузка данных
data = pd.read_csv('имена.csv')

# Шаг 2: Подготовка данных
data['Пол'] = data['Пол'].map({'Мужской': 0, 'Женский': 1})

X = data[['Имя', 'Составляющее 1', 'Составляющее 2']]
y = data['Пол']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Шаг 3: Токенизация и подготовка последовательностей
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['Имя'])
sequences = tokenizer.texts_to_sequences(X_train['Имя'])
word_index = tokenizer.word_index

max_len = max([len(s) for s in sequences])
padded_sequences = pad_sequences(sequences, maxlen=max_len, padding='post')

# Шаг 4: Построение модели
model = Sequential([
    Embedding(input_dim=len(word_index)+1, output_dim=128),
    LSTM(64),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Шаг 5: Обучение модели
history = model.fit(padded_sequences, y_train, epochs=10, validation_split=0.2)

# Шаг 6: Оценка модели
test_sequences = tokenizer.texts_to_sequences(X_test['Имя'])
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_len, padding='post')

loss, accuracy = model.evaluate(test_padded_sequences, y_test)
print(f'Тестовая точность: {accuracy:.2f}')
