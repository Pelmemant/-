import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Чтение данных из CSV файла
try:
    df = pd.read_csv('имена.csv')

    # Проверка данных на наличие пропусков
    print("Проверка данных на наличие пропусков:")
    print(df.isnull().sum())

    # Заполнение пропусков стабильно выбранными значениями
    df.fillna('', inplace=True)

    # Убедимся, что целевая переменная не содержит отсутствующих значений
    df = df[df['Имя'] != '']

    # Создание нового столбца с объединением слов
    df['combined'] = df['Составляющее 1'] + ' ' + df['Составляющее 2']

   

    # Подготовка данных для обучения
    X = df['combined'].astype(str)
    y = df['Имя'].astype(str)

    # Разделение данных на обучающий и тестовый наборы
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Преобразование текстовых данных в числовые признаки
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Создание и обучение модели
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_vec, y_train)

    # Оценка модели
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Точность: {accuracy:.2f}')

    # Функция для генерации нового имени
    def generate_name(word1, word2, model, vectorizer):
        combined_input = word1 + ' ' + word2
        input_vec = vectorizer.transform([combined_input])
        predicted_name = model.predict(input_vec)
        return predicted_name[0]

    # Пример использования
    word1 = "Владеть"
    word2 = "Мир"
    generated_name = generate_name(word1, word2, model, vectorizer)
    print(f'Сгенерированное имя: {generated_name}')

except Exception as e:
    print(f"Произошла ошибка: {e}")
