# WaveAccess Test Task
Тестовое задание в компанию WaveAccess на позицию Data Scientist.
## Установка
```git clone https://github.com/lethxlity/WaveAccess_test_task```
## 1. Income Classification
```
cd income_classification
```
Необходимые для запуска и повторения экспериментов библиотеки устанавливаются в Jupyter Notebook через ```%```.
```
%pip install pandas
%pip install catboost
%pip install sklearn
%pip install seaborn
```
### Описание
Перед нами датасет Adult Census Income.\
Задача - бинарная классификация. Необходимо определить, получает ли человек годовой доход больше суммы N, или меньше.\
Результат выполнения задания представлен в файле ```income_classification.ipynb```. Предсказание для тестовой выборки - в файле ```predictions.txt```.

## 2. Data Summarization
### Установка
При установке рекомендуется использовать виртуальное окружение, например venv.\
Для Unix или MacOS:
```
python -m venv venv
source venv/bin/activate
```
Для Windows:
```
python -m venv venv
source venv\Scripts\activate
```
Затем:
```
cd data_summarizer
pip install -r requirements.txt
```
### Описание
Модуль для создания статистического описания Pandas датафрейма.
Пример работы с Iris:
|        | sepal length (cm)   | sepal width (cm)   | petal length (cm)   | petal width (cm)   | target   |
|:-------|:--------------------|:-------------------|:--------------------|:-------------------|:---------|
| dtype  | float64             | float64            | float64             | float64            | category |
| min    | 4.3                 | 2.0                | 1.0                 | 0.1                | nan      |
| max    | 7.9                 | 4.4                | 6.9                 | 2.5                | nan      |
| mean   | 5.84                | 3.06               | 3.76                | 1.2                | nan      |
| median | 5.8                 | 3.0                | 4.35                | 1.3                | nan      |
| mode   | 5.0                 | 3.0                | 1.4                 | 0.2                | nan      |
| var    | 0.69                | 0.19               | 3.12                | 0.58               | nan      |
| std    | 0.83                | 0.44               | 1.77                | 0.76               | nan      |
| Q1     | 5.1                 | 2.8                | 1.6                 | 0.3                | nan      |
| Q3     | 6.4                 | 3.3                | 5.1                 | 1.8                | nan      |
| IQR    | 1.3                 | 0.5                | 3.5                 | 1.5                | nan      |
| % NaN  | 0.0                 | 0.0                | 0.0                 | 0.0                | nan      |
| unique | 35.0                | 23.0               | 43.0                | 22.0               | 3.0      |
| count  | 150.0               | 150.0              | 150.0               | 150.0              | 150.0    |
| top    | nan                 | nan                | nan                 | nan                | 0.0      |
| freq   | nan                 | nan                | nan                 | nan                | 0.33     |
### Использование
Для использования в виде отдельного модуля требуется импортировать класс DataSummarizer.\
Датафрейм, для которого требуется описание, передается в качестве аргумента в метод ```summarize```.
```
from data_summarizer import DataSummarizer

summarizer = DataSummarizer()
summarizer.summarize(df)
summarizer.save_to_disk()
```
или
```
summary = summarizer.summarize(df)
print(summary)
```
При запуске через командную строку ```python data_summarizer.py``` сгенерируется описание датафрейма Iris.\
Так же в директории присутствуют тесты, для запуска необходимо ввести ```python -m unittest test_summarizer.py```
