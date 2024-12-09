import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Загружаем данные
data = pd.read_csv('имена.csv')

# Проверяем на NaN
print(data.isnull().sum())

# Убираем возможные NaN значения и приводим к строковому типу
data.iloc[:, 0] = data.iloc[:, 0].astype(str)
data.iloc[:, 2] = data.iloc[:, 2].astype(str)
data.iloc[:, 3] = data.iloc[:, 3].astype(str)

# Извлекаем имена и соответствующие слова
names = data.iloc[:, 0].values
words1 = data.iloc[:, 2].values
words2 = data.iloc[:, 3].values

# Создаем уникальные наборы слов и имен
unique_words = sorted(set(words1) | set(words2))
unique_names = sorted(set(names))

# Создаем индекс для слов и имён
word_to_index = {word: i + 1 for i, word in enumerate(unique_words)}
name_to_index = {name: i for i, name in enumerate(unique_names)}

# Подготовим входные и выходные данные
X = np.array([[word_to_index[w1], word_to_index[w2]] for w1, w2 in zip(words1, words2)])
y = np.array([name_to_index[name] for name in names[:X.shape[0]]])  # ограничиваем размер y

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Определим параметры модели
input_dim = len(unique_words) + 1  # Учитываем индекс 0 как "неизвестное слово"
output_dim = len(unique_names)

# Создаем модель Sequential
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=input_dim, output_dim=16, input_length=2),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(output_dim, activation='softmax')
])

# Компилируем модель
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Обучаем модель
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1)
def generate_name(word1, word2):
    # Преобразуем слова в индексы
    input_data = np.array([[word_to_index[word1], word_to_index[word2]]])

    # Получаем вероятности предсказаний
    predictions = model.predict(input_data)
    predicted_index = np.argmax(predictions, axis=-1)[0]

    # Получаем соответствующее имя
    for name, index in name_to_index.items():
        if index == predicted_index:
            return name

# Пример использования
word1 = "Любовь"
word2 = "Слава"
generated_name = generate_name(word1, word2)
print(f'Сгенерированное имя из слов "{word1}" и "{word2}": "{generated_name}"')
