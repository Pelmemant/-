import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder

# Чтение данных из CSV-файла
df = pd.read_csv('имена.csv', encoding='utf-8')

# Подготовка данных
X = df[['Составляющее 1', 'Составляющее 2']].astype(str).values
y_gender = df['Пол'].values
y_parts = df[['Составляющее 1', 'Составляющее 2']].astype(str).values

# Токенизация и векторизация
mlb = MultiLabelBinarizer()
X_encoded = mlb.fit_transform(X)

le = LabelEncoder()
y_gender_encoded = le.fit_transform(y_gender)

# Бинаризация многозначных меток y_parts
y_parts_binarized = mlb.fit_transform(y_parts)

# Разделение данных на тренировочный и тестовый наборы
X_train, X_test, y_train_gender, y_test_gender, y_train_parts, y_test_parts = train_test_split(X_encoded, y_gender_encoded, y_parts_binarized, test_size=0.2, random_state=42)

# Создание моделей
clf_gender = LogisticRegression(random_state=0)
clf_gender.fit(X_train, y_train_gender)

clf_parts = OneVsRestClassifier(LogisticRegression())
clf_parts.fit(X_train, y_train_parts)

# Оценка моделей
y_pred_gender = clf_gender.predict(X_test)
accuracy_gender = (y_pred_gender == y_test_gender).mean() * 100
print(f"Точность модели для определения пола: {accuracy_gender:.2f}%")

y_pred_parts = clf_parts.predict(X_test)
accuracy_parts = (y_pred_parts == y_test_parts).mean(axis=0).mean() * 100
print(f"Точность модели для определения частей имени: {accuracy_parts:.2f}%")
