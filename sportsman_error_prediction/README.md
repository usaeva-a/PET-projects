# Предсказание 
Статус: Проект в работе.

## Описание проекта

В ходе данного проекта необходимо 

## Использованные инструменты и библиотеки
Python, re, pandas, sklearn, matplotlib, phik, Pipeline, RandomForest, CatBoostRegressor.

## Вывод
На этапе <u>обучения модели</u> было построено 3 модели: LogisticRegression, CatBoostClassifier, RandomForestClassifier.
По результатам кросс-валидации лучшей оказалась модель CatBoostClassifier с параметрами {'depth': 8, 'learning_rate': 0.1}. Ее метрика ROC-AUC на кросс-валидации равна 98.9%. Метрика ROC-AUC на тестовых данных равна 99.3%.
