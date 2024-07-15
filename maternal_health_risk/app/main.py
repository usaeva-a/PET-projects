import streamlit as st
import pandas as pd
import os
from xgboost import XGBClassifier

st.header('ОПРЕДЕЛЕНИЕ УРОВНЯ РИСКА ПАЦИЕНТА', divider='gray')

@st.cache_data
def load_dataset(data_path):
    dataset = pd.read_csv(data_path, sep=',')
    return dataset

path = 'data\maternal_health_clean_csv.csv'
data = load_dataset(path)
X = data.drop(['risk_level'], axis=1)
y = data['risk_level']

col1, col2 = st.columns(2, gap='large')
col1.subheader('Введите параметры:')

def user_input_features():
        
    age = col1.number_input('Возраст пациента', min_value=10, max_value=70, value=11, step=1)
    systolic_bp = col1.number_input('Систолическое давление, мм.рт.ст.', min_value=70, max_value=160, value=71, step=1)
    diastolic_bp = col1.number_input('Диастолическое давление, мм.рт.ст.', min_value=50, max_value=100, value=51, step=1)
    bs = col1.number_input('Уровень глюкозы в крови, ммоль/л', min_value=6.0, max_value=19.0, value=6.1, step=0.1)
    body_temp = col1.number_input('Температура тела, Фаренгейт', min_value=98.0, max_value=103.0, value=98.1, step=0.1)
    heart_rate = col1.number_input('Пульс в состоянии покоя, уд./мин', min_value=60, max_value=90, value=61, step=1)

    user_features = {'age': age,
                'systolic_bp': systolic_bp,
                'diastolic_bp': diastolic_bp,
                'bs': bs,
                'body_temp': body_temp,
                'heart_rate': heart_rate} 
    features = pd.DataFrame(user_features, index=[0])
    return features

df = user_input_features()

clf = XGBClassifier(learning_rate=0.05, max_depth=5, n_estimators=100, random_state=10)
clf.fit(X, y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

col2.subheader('Результат:')

if prediction == 0:
    col2.success(':green[**Низкий уровень риска**]')
elif prediction == 1:
    col2.warning(':yellow[**Средний уровень риска**]')  
else:
    col2.error('''
    :red[**Высокий уровень риска**]
                ''')  
col2.write(f'Вероятность {prediction_proba.max():.0%}')