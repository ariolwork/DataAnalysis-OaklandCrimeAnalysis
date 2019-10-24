```python
import pandas as pd 
import datetime
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error
from math import sqrt
from pylab import rcParams
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
```

Датасет "Oakland Crime Statistics 2011 to 2016" был взят с:
    https://www.kaggle.com/cityofoakland/oakland-crime-statistics-2011-to-2016
(This dataset is distributed under the following licenses: Open Data Commons Public Domain Dedication and License, NA)
В нем содержатся записи о преступлениях, совершенных в городе Окленд с 2011 по 2016 года. 

Каждое из преступлений в датасете описаывется набором данных, в том числе датой и временем совершения.
Подробнее содержание таблицы можно посмотреть ниже.

Наша цель по данной таблице построить предсказание количества преступлений на следующий промежуток времени.(Например для определения количества требуемых сотрудников на службе)


```python
# подключаем датасет
# для начала подключаем лишь часть таблиц(только за последние 3 года), предполагая, что далее не существует
# связи с текущим моментом времени
import operator
crimes = pd.read_csv('db1/records-for-2014.csv', sep =',')
#crimes = crimes.append(pd.read_csv('db1/records-for-2012.csv', sep =','), sort=True,  ignore_index=True)
#crimes = crimes.append(pd.read_csv('db1/records-for-2013.csv', sep =','), sort=True,  ignore_index=True)
#crimes = crimes.append(pd.read_csv('db1/records-for-2014.csv', sep =','), sort=True,  ignore_index=True)
crimes = crimes.append(pd.read_csv('db1/records-for-2015.csv', sep =','), sort=True,  ignore_index=True)
crimes = crimes.append(pd.read_csv('db1/records-for-2016.csv', sep =','), sort=True,  ignore_index=True)
crimes.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Agency</th>
      <th>Area Id</th>
      <th>Beat</th>
      <th>Closed Time</th>
      <th>Create Time</th>
      <th>Event Number</th>
      <th>Incident Type Description</th>
      <th>Incident Type Id</th>
      <th>Location</th>
      <th>Location 1</th>
      <th>Priority</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>OP</td>
      <td>1</td>
      <td>02X</td>
      <td>2014-01-01T03:22:08</td>
      <td>2014-01-01T00:00:00</td>
      <td>LOP140101000001</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"LINDEN ST","cit...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OP</td>
      <td>2</td>
      <td>26Y</td>
      <td>2014-01-01T02:56:31</td>
      <td>2014-01-01T00:00:00</td>
      <td>LOP140101000002</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"AV&amp;amp;INTERNAT...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>OP</td>
      <td>2</td>
      <td>30Y</td>
      <td>2014-01-01T00:49:53</td>
      <td>2014-01-01T00:00:00</td>
      <td>LOP140101000004</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"AV&amp;amp;MACARTHU...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OP</td>
      <td>2</td>
      <td>30Y</td>
      <td>2014-01-01T02:51:11</td>
      <td>2014-01-01T00:00:00</td>
      <td>LOP140101000005</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"MACARTHUR BLVD"...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OP</td>
      <td>2</td>
      <td>35X</td>
      <td>2014-01-01T05:33:22</td>
      <td>2014-01-01T00:01:04</td>
      <td>LOP140101000010</td>
      <td>SUBJECT ARMED WITH W</td>
      <td>CODE7</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"AV&amp;amp;DOWLING ...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>OP</td>
      <td>2</td>
      <td>32X</td>
      <td>2014-01-01T03:53:57</td>
      <td>2014-01-01T00:01:08</td>
      <td>LOP140101000006</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"AV&amp;amp;INTERNAT...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>OP</td>
      <td>2</td>
      <td>31Z</td>
      <td>2014-01-01T00:01:32</td>
      <td>2014-01-01T00:01:23</td>
      <td>LOP140101000007</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"AV&amp;amp;PIPPIN S...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>OP</td>
      <td>2</td>
      <td>23X</td>
      <td>2014-01-01T08:23:08</td>
      <td>2014-01-01T00:01:31</td>
      <td>LOP140101000008</td>
      <td>911 HANG-UP</td>
      <td>911H</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"E 16TH ST","cit...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>OP</td>
      <td>1</td>
      <td>02Y</td>
      <td>2014-01-01T03:30:10</td>
      <td>2014-01-01T00:01:40</td>
      <td>LOP140101000009</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"ST&amp;amp;WILLOW S...</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>OP</td>
      <td>2</td>
      <td>35X</td>
      <td>2014-01-01T10:35:45</td>
      <td>2014-01-01T00:02:10</td>
      <td>LOP140101000011</td>
      <td>415 GUNSHOTS</td>
      <td>415GS</td>
      <td>NaN</td>
      <td>{'human_address': '{"address":"86TH AV","city"...</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



<br>Далее делаем предобработку данных т.к. нам надо построить на основе данного датасет другой.
<br>Нас интересует только количество в определенную дату
<br>Все остальные параметры можно откинуть


```python
# удаляем дубликаты по столбцу "Event Number" т.к. он задает уникальный ключ пресупления  и заодно проверям сколько их было
# в нашем случае их не обнаружилось
print(crimes.shape)
crimes.drop_duplicates(subset ="Event Number", keep = False, inplace = True)
print(crimes.shape)
```

    (490889, 11)
    (490889, 11)



```python
# удаляем преступления с незаданным временем совершения и уникальным номером(Event Number),
# считая эти данные некоректными
# в нашем случае их не обнаружилось
print(crimes.shape)
crimes.dropna(subset=['Create Time'], inplace=True)
crimes.dropna(subset=['Event Number'], inplace=True)
print(crimes.shape)
```

    (490889, 11)
    (490888, 11)



```python
# преобразовываем столбец даты совершения преступления(Create Time), удаляя конкретное время
# далее для каждой из дат считаем количество совершенных в этот день преступлений
crimes.rename(columns={'Create Time': 'date'}, inplace=True)
crimes['date'] = crimes['date'].str.split("T").str[0]
crimes = crimes.date.value_counts()
```


```python
# создаем таблицу соответствия дате количства совершенных преступлений и сортируем по дате
# смотрим результат
table = pd.DataFrame({'date':crimes.index, 'value':crimes.values})
table['date'] = pd.to_datetime(table.date, format='%Y-%m-%d')
table.index = table.date
#table.drop(columns='date', inplace=True)
table = table.sort_values('date', ascending = True)
#table = table.resample('W', how='mean')
#table = table.rolling(3).median()
print(table.shape)
table.head(10)
```

    (943, 2)


    C:\Users\Artem\Anaconda3\lib\site-packages\ipykernel_launcher.py:7: FutureWarning: 'date' is both an index level and a column label.
    Defaulting to column, but this will raise an ambiguity error in a future version
      import sys





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-01</th>
      <td>2014-01-01</td>
      <td>529</td>
    </tr>
    <tr>
      <th>2014-01-02</th>
      <td>2014-01-02</td>
      <td>477</td>
    </tr>
    <tr>
      <th>2014-01-03</th>
      <td>2014-01-03</td>
      <td>509</td>
    </tr>
    <tr>
      <th>2014-01-04</th>
      <td>2014-01-04</td>
      <td>488</td>
    </tr>
    <tr>
      <th>2014-01-05</th>
      <td>2014-01-05</td>
      <td>489</td>
    </tr>
    <tr>
      <th>2014-01-06</th>
      <td>2014-01-06</td>
      <td>502</td>
    </tr>
    <tr>
      <th>2014-01-07</th>
      <td>2014-01-07</td>
      <td>463</td>
    </tr>
    <tr>
      <th>2014-01-08</th>
      <td>2014-01-08</td>
      <td>452</td>
    </tr>
    <tr>
      <th>2014-01-09</th>
      <td>2014-01-09</td>
      <td>492</td>
    </tr>
    <tr>
      <th>2014-01-10</th>
      <td>2014-01-10</td>
      <td>526</td>
    </tr>
  </tbody>
</table>
</div>




```python
# сохраним таблицу, чтобы в дальнейшем не понадобилась заново преобразовывать датасет
table.to_csv('out.csv', sep =',')
```

<br>Визуальный анализ данных
<br>Используем различные типы графиков, чтобы описать данные
<br>И посмотрим, какие зависимости или интересные инсайты были получены.


```python
table.value.plot(figsize=(15,6), fontsize=12)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b169507080>




![png](output_10_1.png)


Видим странные большие выбросы в июле каждого года.
Найдем даты 30 самых больших выбрососв:


```python
# найдем 30 наибольших выбросов и
# для лучшего анализа представим данные, игнорируя год  и сортируя по дню и месяцу
table['temp1'] = table.index.day
table['temp2'] = table.index.month
table.nlargest(50, table.columns[1]).sort_values(by=['temp1', 'temp2'])
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>value</th>
      <th>temp1</th>
      <th>temp2</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-05-01</th>
      <td>2015-05-01</td>
      <td>624</td>
      <td>1</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2015-07-02</th>
      <td>2015-07-02</td>
      <td>612</td>
      <td>2</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2016-06-03</th>
      <td>2016-06-03</td>
      <td>590</td>
      <td>3</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2015-07-03</th>
      <td>2015-07-03</td>
      <td>618</td>
      <td>3</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-07-04</th>
      <td>2015-07-04</td>
      <td>831</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>2016-07-04</td>
      <td>786</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2014-07-04</th>
      <td>2014-07-04</td>
      <td>638</td>
      <td>4</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-09-04</th>
      <td>2015-09-04</td>
      <td>603</td>
      <td>4</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2016-03-05</th>
      <td>2016-03-05</td>
      <td>587</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2015-05-05</th>
      <td>2015-05-05</td>
      <td>612</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2015-07-05</th>
      <td>2015-07-05</td>
      <td>619</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2014-07-05</th>
      <td>2014-07-05</td>
      <td>603</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>2016-07-05</td>
      <td>593</td>
      <td>5</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-02-06</th>
      <td>2015-02-06</td>
      <td>597</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2016-04-06</th>
      <td>2016-04-06</td>
      <td>594</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2016-02-08</th>
      <td>2016-02-08</td>
      <td>597</td>
      <td>8</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2015-09-09</th>
      <td>2015-09-09</td>
      <td>620</td>
      <td>9</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2015-10-09</th>
      <td>2015-10-09</td>
      <td>592</td>
      <td>9</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2016-06-10</th>
      <td>2016-06-10</td>
      <td>587</td>
      <td>10</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2016-06-11</th>
      <td>2016-06-11</td>
      <td>606</td>
      <td>11</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2014-10-11</th>
      <td>2014-10-11</td>
      <td>588</td>
      <td>11</td>
      <td>10</td>
    </tr>
    <tr>
      <th>2015-12-11</th>
      <td>2015-12-11</td>
      <td>619</td>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2014-12-11</th>
      <td>2014-12-11</td>
      <td>600</td>
      <td>11</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2015-06-12</th>
      <td>2015-06-12</td>
      <td>589</td>
      <td>12</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2014-06-13</th>
      <td>2014-06-13</td>
      <td>591</td>
      <td>13</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2014-05-14</th>
      <td>2014-05-14</td>
      <td>590</td>
      <td>14</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2015-08-15</th>
      <td>2015-08-15</td>
      <td>592</td>
      <td>15</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2015-08-17</th>
      <td>2015-08-17</td>
      <td>600</td>
      <td>17</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2015-07-18</th>
      <td>2015-07-18</td>
      <td>589</td>
      <td>18</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-06-19</th>
      <td>2015-06-19</td>
      <td>619</td>
      <td>19</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2015-09-19</th>
      <td>2015-09-19</td>
      <td>615</td>
      <td>19</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2014-12-19</th>
      <td>2014-12-19</td>
      <td>590</td>
      <td>19</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2015-02-20</th>
      <td>2015-02-20</td>
      <td>590</td>
      <td>20</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2016-05-20</th>
      <td>2016-05-20</td>
      <td>608</td>
      <td>20</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2015-07-20</th>
      <td>2015-07-20</td>
      <td>612</td>
      <td>20</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-09-21</th>
      <td>2015-09-21</td>
      <td>600</td>
      <td>21</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2016-04-23</th>
      <td>2016-04-23</td>
      <td>587</td>
      <td>23</td>
      <td>4</td>
    </tr>
    <tr>
      <th>2015-09-23</th>
      <td>2015-09-23</td>
      <td>604</td>
      <td>23</td>
      <td>9</td>
    </tr>
    <tr>
      <th>2015-01-25</th>
      <td>2015-01-25</td>
      <td>598</td>
      <td>25</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2016-06-25</th>
      <td>2016-06-25</td>
      <td>609</td>
      <td>25</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2015-07-25</th>
      <td>2015-07-25</td>
      <td>617</td>
      <td>25</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2014-07-25</th>
      <td>2014-07-25</td>
      <td>599</td>
      <td>25</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2015-02-27</th>
      <td>2015-02-27</td>
      <td>622</td>
      <td>27</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2016-05-28</th>
      <td>2016-05-28</td>
      <td>607</td>
      <td>28</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2015-08-28</th>
      <td>2015-08-28</td>
      <td>663</td>
      <td>28</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2014-06-29</th>
      <td>2014-06-29</td>
      <td>606</td>
      <td>29</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2014-06-30</th>
      <td>2014-06-30</td>
      <td>597</td>
      <td>30</td>
      <td>6</td>
    </tr>
    <tr>
      <th>2015-08-30</th>
      <td>2015-08-30</td>
      <td>606</td>
      <td>30</td>
      <td>8</td>
    </tr>
    <tr>
      <th>2014-12-30</th>
      <td>2014-12-30</td>
      <td>651</td>
      <td>30</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2015-08-31</th>
      <td>2015-08-31</td>
      <td>598</td>
      <td>31</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>




```python
table.drop(columns='temp1', inplace=True)
table.drop(columns='temp2', inplace=True)
```

Замечаем, что некоторые конкретные даты входят в этот топ
<br>Например, 04.07 или 5.07.
<br>Заметим, что 4.07 проходин один из крупнейших праздников Independence Day
<br>Следовательно, можно считать, что это вбросы, всвязи с этим праздником
<br>Конечно будут вбросы связанные с событиями в городе, но они не будут повторяться в другие года так что их можно проигнорировать

<br>Логично предположить, что в другие праздники так же дложны происходить выбросы
<br>Проверим это, сравнив значения в определенные определенные дни(празникики) со средним по всем дням
<br>Выберем глаыне праздники в той местности:
    <br>New Year's Day - Observed, Monday, January 1 - 01.01
    <br>Martin Luther King Jr.'s Birthday - Observed, Monday, January 15 - 15.01
    <br>Lincoln's Birthday - Observed, Monday, February 12 - 12.02
    <br>Washington's Birthday - Observed, Monday, February 19 - 19.02
    <br>Memorial Day - Observed, Monday, May 28 - 28.05
    <br>Independence Day - Wednesday, July 4 - 04.07
    <br>Labor Day - Observed, Monday, September 3 - 03.09
    <br>California Admission Day holiday observed - Sunday, September 9 - 09.09
    <br>Veteran's Day holiday observed - Sunday, November 11 - 11.11
    <br>Veteran's Day - Observed, Monday, November 12 - 12.11
    <br>Thanksgiving Day holiday observed - Thursday, November 22 - 22.11
    <br>Thanksgiving - Observed, Thursday, November 22 & Friday, November 23 - 23.11
    <br>Christmas - Tuesday, December 25 - 25.12


```python
ax = table[((table.index.day == 1) & (table.index.month == 1)) |
      ((table.index.day == 15) & (table.index.month == 1)) |
      ((table.index.day == 12) & (table.index.month == 2)) |
      ((table.index.day == 19) & (table.index.month == 2)) |
      ((table.index.day == 28) & (table.index.month == 5)) |
      ((table.index.day == 4) & (table.index.month == 7)) |
      ((table.index.day == 3) & (table.index.month == 9)) |
      ((table.index.day == 9) & (table.index.month == 9)) |
      ((table.index.day == 11) & (table.index.month == 11)) |
      ((table.index.day == 12) & (table.index.month == 11)) |
      ((table.index.day == 22) & (table.index.month == 11)) |
      ((table.index.day == 23) & (table.index.month == 11)) |
      ((table.index.day == 25) & (table.index.month == 12)) |
      ((table.index.day == 31) & (table.index.month == 12))
     ].value.plot(figsize=(15,6), fontsize=12)

ax.axhline(y=table.mean().value, color='r', linestyle='--', lw=2)
print("table.mean() - ",table.mean().value)
```

    table.mean() -  520.559915164369



![png](output_15_1.png)


Замечаем, что в некоторые праздники количество преступлений гораздо меньше среднего
<br>А в некоторые приблизительно равны среднему:
<br> давайте посмотрим что это за праздники
<br>считаем ,будем с погрешностью 3% что бы хотя бы попытаться сравнять тренды


```python
t1 = table[(((table.index.day == 1) & (table.index.month == 1)) |
      ((table.index.day == 15) & (table.index.month == 1)) |
      ((table.index.day == 12) & (table.index.month == 2)) |
      ((table.index.day == 19) & (table.index.month == 2)) |
      ((table.index.day == 28) & (table.index.month == 5)) |
      ((table.index.day == 4) & (table.index.month == 7)) |
      ((table.index.day == 3) & (table.index.month == 9)) |
      ((table.index.day == 9) & (table.index.month == 9)) |
      ((table.index.day == 11) & (table.index.month == 11)) |
      ((table.index.day == 12) & (table.index.month == 11)) |
      ((table.index.day == 22) & (table.index.month == 11)) |
      ((table.index.day == 23) & (table.index.month == 11)) |
      ((table.index.day == 25) & (table.index.month == 12)) |
      ((table.index.day == 31) & (table.index.month == 12)))
     ]
```


```python
#Cильно меньше среднего:
t1[t1.value < table.mean().value*0.97]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-02-12</th>
      <td>2014-02-12</td>
      <td>471</td>
    </tr>
    <tr>
      <th>2014-09-09</th>
      <td>2014-09-09</td>
      <td>469</td>
    </tr>
    <tr>
      <th>2014-11-11</th>
      <td>2014-11-11</td>
      <td>489</td>
    </tr>
    <tr>
      <th>2014-11-12</th>
      <td>2014-11-12</td>
      <td>489</td>
    </tr>
    <tr>
      <th>2014-11-23</th>
      <td>2014-11-23</td>
      <td>498</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>2015-01-15</td>
      <td>485</td>
    </tr>
    <tr>
      <th>2015-05-28</th>
      <td>2015-05-28</td>
      <td>477</td>
    </tr>
    <tr>
      <th>2015-11-11</th>
      <td>2015-11-11</td>
      <td>424</td>
    </tr>
    <tr>
      <th>2015-11-12</th>
      <td>2015-11-12</td>
      <td>499</td>
    </tr>
    <tr>
      <th>2015-11-22</th>
      <td>2015-11-22</td>
      <td>466</td>
    </tr>
    <tr>
      <th>2015-12-25</th>
      <td>2015-12-25</td>
      <td>438</td>
    </tr>
    <tr>
      <th>2015-12-31</th>
      <td>2015-12-31</td>
      <td>501</td>
    </tr>
  </tbody>
</table>
</div>



<br>Veteran's Day holiday observed - Sunday, November 11 - 11.11
<br>Veteran's Day - Observed, Monday, November 12 - 12.11


```python
#Приблизительно равно:
t1[(t1.value > table.mean().value*0.97)&(t1.value < table.mean().value*1.03)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2014-01-01</th>
      <td>2014-01-01</td>
      <td>529</td>
    </tr>
    <tr>
      <th>2014-01-15</th>
      <td>2014-01-15</td>
      <td>535</td>
    </tr>
    <tr>
      <th>2014-02-19</th>
      <td>2014-02-19</td>
      <td>521</td>
    </tr>
    <tr>
      <th>2014-12-25</th>
      <td>2014-12-25</td>
      <td>518</td>
    </tr>
    <tr>
      <th>2014-12-31</th>
      <td>2014-12-31</td>
      <td>509</td>
    </tr>
    <tr>
      <th>2015-02-19</th>
      <td>2015-02-19</td>
      <td>536</td>
    </tr>
    <tr>
      <th>2015-11-23</th>
      <td>2015-11-23</td>
      <td>526</td>
    </tr>
    <tr>
      <th>2016-01-01</th>
      <td>2016-01-01</td>
      <td>524</td>
    </tr>
    <tr>
      <th>2016-01-15</th>
      <td>2016-01-15</td>
      <td>531</td>
    </tr>
    <tr>
      <th>2016-02-19</th>
      <td>2016-02-19</td>
      <td>517</td>
    </tr>
  </tbody>
</table>
</div>



<br>New Year's Day - Observed, Monday, January 1 - 01.01
<br>Martin Luther King Jr.'s Birthday - Observed, Monday, January 15 - 15.01
<br>Washington's Birthday - Observed, Monday, February 19 - 19.02

<br>Можно сделать вывод, что в большинстве случаев в праздники количество преступлений выше среднего
<br>В некоторые примерно столько же, сколько и обычно 
<br>Однако в некоторые значительно меньше(например 11.11 - Veteran's Day) количество преступлений значительно падает
<br>Эту информацию можно использовать что бы статически вносить правки в предсказания на дни праздников
<br> Так же можно кластеризовать праздники в зависимости от количества преступлений и использовать эту кластеризацию как характеристику праздника


```python
table.hist(bins=100, color= 'r', alpha=0.59, figsize=(15,6)) 
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001B15DC89C50>]],
          dtype=object)




![png](output_23_1.png)


<br>Гистограмма по виду близка к нормальному распределению
<br>Можно обратить внимание, что в целом, количество преступлений составляет 500 гистограмма имеет очень крутые уклоны
Это значит, что в среднем количество преступлений весьма стационарно(хотя это можно заметить и просто по построению всех дней)

 

Теперь попробуем выяснить влияет ли время года на количество совершаемых преступлений
<br>
<br> Для этого разобъем датасет на сезоны и рассмотрим среднее число преступлений на этих участках.


```python
fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(111)
ax1.plot(table.value)

winter = pd.date_range(start='1/1/2013', end='3/1/2013', freq='D')
ax1.fill_between(winter, 600, facecolor='silver', alpha=0.5)
m = table.value.loc[winter].mean()
ax1.plot(winter, [m] * winter.shape[0], 'r')

spring = pd.date_range(start='3/1/2013', end='6/1/2013', freq='D')
ax1.fill_between(spring, 600, facecolor='brown', alpha=0.5)
m = table.value.loc[spring].mean()
ax1.plot(spring, [m] * spring.shape[0], 'r')

summer = pd.date_range(start='6/1/2013', end='9/1/2013', freq='D')
ax1.fill_between(summer, 600, facecolor='green', alpha=0.5)
m = table.value.loc[summer].mean()
ax1.plot(summer, [m] * summer.shape[0], 'r')

autumn = pd.date_range(start='9/1/2013', end='12/1/2013', freq='D')
ax1.fill_between(autumn, 600, facecolor='gold', alpha=0.5)
m = table.value.loc[autumn].mean()
ax1.plot(autumn, [m] * autumn.shape[0], 'r')


winter = pd.date_range(start='12/1/2013', end='3/1/2014', freq='D')
ax1.fill_between(winter, 600, facecolor='silver', alpha=0.5)
m = table.value.loc[winter].mean()
ax1.plot(winter, [m] * winter.shape[0], 'r')

spring = pd.date_range(start='3/1/2014', end='6/1/2014', freq='D')
ax1.fill_between(spring, 600, facecolor='brown', alpha=0.5)
m = table.value.loc[spring].mean()
ax1.plot(spring, [m] * spring.shape[0], 'r')

summer = pd.date_range(start='6/1/2014', end='9/1/2014', freq='D')
ax1.fill_between(summer, 600, facecolor='green', alpha=0.5)
m = table.value.loc[summer].mean()
ax1.plot(summer, [m] * summer.shape[0], 'r')

autumn = pd.date_range(start='9/1/2014', end='12/1/2014', freq='D')
ax1.fill_between(autumn, 600, facecolor='gold', alpha=0.5)
m = table.value.loc[autumn].mean()
ax1.plot(autumn, [m] * autumn.shape[0], 'r')


winter = pd.date_range(start='12/1/2014', end='3/1/2015', freq='D')
ax1.fill_between(winter, 600, facecolor='silver', alpha=0.5)
m = table.value.loc[winter].mean()
ax1.plot(winter, [m] * winter.shape[0], 'r')

spring = pd.date_range(start='3/1/2015', end='6/1/2015', freq='D')
ax1.fill_between(spring, 600, facecolor='brown', alpha=0.5)
m = table.value.loc[spring].mean()
ax1.plot(spring, [m] * spring.shape[0], 'r')

summer = pd.date_range(start='6/1/2015', end='9/1/2015', freq='D')
ax1.fill_between(summer, 600, facecolor='green', alpha=0.5)
m = table.value.loc[summer].mean()
ax1.plot(summer, [m] * summer.shape[0], 'r')

autumn = pd.date_range(start='9/1/2015', end='12/1/2015', freq='D')
ax1.fill_between(autumn, 600, facecolor='gold', alpha=0.5)
m = table.value.loc[autumn].mean()
ax1.plot(autumn, [m] * autumn.shape[0], 'r')


winter = pd.date_range(start='12/1/2015', end='3/1/2016', freq='D')
ax1.fill_between(winter, 600, facecolor='silver', alpha=0.5)
m = table.value.loc[winter].mean()
ax1.plot(winter, [m] * winter.shape[0], 'r')

spring = pd.date_range(start='3/1/2016', end='6/1/2016', freq='D')
ax1.fill_between(spring, 600, facecolor='brown', alpha=0.5)
m = table.value.loc[spring].mean()
ax1.plot(spring, [m] * spring.shape[0], 'r')

summer = pd.date_range(start='6/1/2016', end='9/1/2016', freq='D')
ax1.fill_between(summer, 600, facecolor='green', alpha=0.5)
m = table.value.loc[summer].mean()
ax1.plot(summer, [m] * summer.shape[0], 'r')

ax1.legend(loc='best')
plt.show()

```

    C:\Users\Artem\Anaconda3\lib\site-packages\ipykernel_launcher.py:80: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike



![png](output_27_1.png)


Видим, что осенью к земе количество падает, зимой по чуть чуть растет(наверное отходят от праздников)) а потом весной держится стационарно или растет. Летом держится в целом стационарно.

Попробуем найти сезонность и тренд:


```python
rcParams['figure.figsize'] = 12, 7
sm.tsa.seasonal_decompose(table.value).plot()
plt.show()
```


![png](output_30_0.png)



```python
fig, ax = plt.subplots(figsize=(15,6))
sm.graphics.tsa.plot_acf(table.value, lags=50, ax = ax)
plt.show()
```


![png](output_31_0.png)


Как видим у ряда выражена сезонность но не вырожен тренд. 
<br>По автокорреляции можно сказать, что сезонность составляет 7 дней.
<br>Можно еще заметить некоторый подтренд, который повторяется дважды в неделю(3 и 4 дня поочередно)

 

 

# Попробуем предсказать отрезок времени.


```python
# для начала разобьем датасет на обучающую и тетовую выборки
# и построим их для наглядного понмания
train = table.iloc[:-20]
test = table.iloc[-20:]

# здесь будем хранить предсказания
y = test.copy()

train.value.plot( figsize=(15,6), fontsize=12)
test.value.plot( figsize=(15,6),fontsize=12)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1b152e45048>




![png](output_36_1.png)


 

 

## Рассмотрим наивный подход: завтра = вчера
<br>По графику очевидно, что этот подход даст не очень большую погрешность в одни промедутки времени, но просто гигантскую в другие(например при резком падении которые наблюдаются в Jul 2014 или Jan 2016)
<br>
<br> Но для чистоты эксперимента конечно построим эту модель


```python
y['naive'] = train.value[-1]
plt.figure(figsize=(15,8))
plt.plot(train.index, train.value, label='Train')
plt.plot(test.index,test.value, label='Test')
plt.plot(y.index,y['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
print(sqrt(mean_squared_error(test.value, y.naive)))
plt.show()
```

    26.936035343012154



![png](output_40_1.png)


Видим весьма неплохую оценку,но это только потому что наше тестирование попало на удачный отрезок.
<br>
<br>Для примера построим предсказание для большего тестового ряда(что бы большой уклон попал в предсказываемую часть)


```python
train1 = table.iloc[:-40]
test1 = table.iloc[-40:]
y1 = test1.copy()
y1['naive'] = train1.value[-1]
plt.figure(figsize=(15,8))
plt.plot(train1.index, train1.value, label='Train')
plt.plot(test1.index,test1.value, label='Test')
plt.plot(y1.index,y1['naive'], label='Naive Forecast')
plt.legend(loc='best')
plt.title("Naive Forecast")
print(sqrt(mean_squared_error(test1.value, y1.naive)))
plt.show()
```

    61.44224930778495



![png](output_42_1.png)


Видим, что ошибка резко возрасла, что нам конечно же не хочется видеть

## Теперь рассмотрим метод Холта-Уинтерса


```python
# выше видим, что у нас есть сезональность, но нет тренда
fit1 = ExponentialSmoothing(np.asarray(train.value) ,seasonal_periods=7 ,trend=None, seasonal='add',).fit()
y['Holt_Winter'] = fit1.forecast(len(test))
plt.figure(figsize=(16,8))
plt.plot( train.value, label='Train')
plt.plot(test.value, label='Test')
plt.plot(y['Holt_Winter'], label='Holt_Winter')
plt.legend(loc='best')
print(sqrt(mean_squared_error(test.value, y.Holt_Winter)))
plt.show()
```

    51.77765190169041



![png](output_45_1.png)


Ошибка достаточно большая, хотя и меньше чем у наивного подхода.

## SARIMA

Для начала сделаем ряд стационарным


```python
train_diff = train.value.diff(periods=7).dropna()
```


```python
stat_test = sm.tsa.adfuller(train_diff)
print ('adf: ', stat_test[0] )
print ('p-value: ', stat_test[1])
print('Critical values: ', stat_test[4])
if stat_test[0]> stat_test[4]['5%']: 
    print ('есть единичные корни, ряд не стационарен')
else:
    print ('единичных корней нет, ряд стационарен')
```

    adf:  -9.001559649585763
    p-value:  6.4766932433361876e-15
    Critical values:  {'1%': -3.4376857669714957, '5%': -2.864778351359889, '10%': -2.5684943199755765}
    единичных корней нет, ряд стационарен



```python
plt.figure(figsize=(15,8))
plt.plot(train_diff, label='Train')
plt.show()
```


![png](output_51_0.png)



```python
fig = plt.figure(figsize=(15,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(train_diff.values.squeeze(), lags=25, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(train_diff, lags=25, ax=ax2)
```


![png](output_52_0.png)


подберем параметры:


```python
#ACF
q = 2
Q = 1
#PACF
p = 15
P = 1

d = 0
D = 1
s = 7
```


```python
best_model=sm.tsa.statespace.SARIMAX(train.value.squeeze(), order=(p, d, q), 
                                        seasonal_order=(P, D, Q, s)).fit()
```

    C:\Users\Artem\Anaconda3\lib\site-packages\statsmodels\tsa\base\tsa_model.py:171: ValueWarning: No frequency information was provided, so inferred frequency D will be used.
      % freq, ValueWarning)
    C:\Users\Artem\Anaconda3\lib\site-packages\statsmodels\base\model.py:508: ConvergenceWarning: Maximum Likelihood optimization failed to converge. Check mle_retvals
      "Check mle_retvals", ConvergenceWarning)



```python
y_r = best_model.predict(start = train.shape[0], end = train.shape[0] + test.shape[0]-1)

predict = pd.Series(data=y_r, index=test.index)
plt.figure(figsize=(15,8))
plt.plot(train.value, label='Train')
plt.plot(test.value, label='Test')
plt.plot(predict, label='SARIMA')
print(sqrt(mean_squared_error(test.value, y_r)))
plt.legend(loc='best')
plt.show()
```

    48.10990326376206



![png](output_56_1.png)


Заметим, что даже так мы не получили желаемого эффекта. Возможно можно лучше подобрать параметры

     

# Рассмотрим различные статистические гипотезы:

## Normality Tests

Проверяет, имеют ли данные распределение Гаусса:


```python
#Shapiro-Wilk Test
from scipy.stats import normaltest
print(normaltest(table.value))
#D’Agostino’s K^2 Test
from scipy.stats import shapiro
print(shapiro(table.value))
#Anderson-Darling Test
from scipy.stats import anderson
print(anderson(table.value))
```

    NormaltestResult(statistic=169.9643986953336, pvalue=1.237940476967876e-37)
    (0.9640318751335144, 1.7276297444501418e-14)
    AndersonResult(statistic=2.3323951500963176, critical_values=array([0.574, 0.653, 0.784, 0.914, 1.087]), significance_level=array([15. , 10. ,  5. ,  2.5,  1. ]))


На основании теста можно отказаться от этой гипотезы, хотя на гистограмме получается похожее распределение.


```python
table.hist(bins=100, color= 'r', alpha=0.59, figsize=(15,6)) 
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x000001B15B4500B8>]],
          dtype=object)




![png](output_64_1.png)


## Время года

Сформулируем гипотезу о том, что количество преступлений не зависит от врменеи года:


```python
from scipy import stats
#разобьем на времена года
winter1 = table.loc[pd.date_range(start='1/1/2014', end='3/1/2014', freq='D')].dropna()
spring1 = table.loc[pd.date_range(start='3/1/2014', end='6/1/2014', freq='D')].dropna()
summer1 = table.loc[pd.date_range(start='6/1/2014', end='9/1/2014', freq='D')].dropna()
autumn1 = table.loc[pd.date_range(start='9/1/2014', end='12/1/2014', freq='D')].dropna()
winter2 = table.loc[pd.date_range(start='12/1/2014', end='3/1/2015', freq='D')].dropna()
spring2 = table.loc[pd.date_range(start='3/1/2015', end='6/1/2015', freq='D')].dropna()
summer2 = table.loc[pd.date_range(start='6/1/2015', end='9/1/2015', freq='D')].dropna()
autumn2 = table.loc[pd.date_range(start='9/1/2015', end='12/1/2015', freq='D')].dropna()
winter3 = table.loc[pd.date_range(start='12/1/2015', end='3/1/2016', freq='D')].dropna()
spring3 = table.loc[pd.date_range(start='3/1/2016', end='6/1/2016', freq='D')].dropna()
summer3 = table.loc[pd.date_range(start='6/1/2016', end='9/1/2016', freq='D')].dropna()
#проверим распределение
fig, axes = plt.subplots(11, 2)
fig.set_figheight(20)
fig.set_figwidth(15)


axes[0, 0].hist(winter1.value)
stats.probplot(winter1.value, dist = "norm", plot = axes[0, 1])

axes[1, 0].hist(winter2.value, 25)
stats.probplot(winter2.value, dist = "norm", plot = axes[1, 1])

axes[2, 0].hist(winter3.value, 25)
stats.probplot(winter3.value, dist = "norm", plot = axes[2, 1])

axes[3, 0].hist(spring1.value, 25)
stats.probplot(spring1.value, dist = "norm", plot = axes[3, 1])

axes[4, 0].hist(spring2.value, 25)
stats.probplot(spring2.value, dist = "norm", plot = axes[4, 1])

axes[5, 0].hist(spring3.value, 25)
stats.probplot(spring3.value, dist = "norm", plot = axes[5, 1])

axes[6, 0].hist(summer1.value, 25)
stats.probplot(summer1.value, dist = "norm", plot = axes[6, 1])

axes[7, 0].hist(summer2.value, 25)
stats.probplot(summer2.value, dist = "norm", plot = axes[7, 1])

axes[8, 0].hist(summer3.value, 25)
stats.probplot(summer3.value, dist = "norm", plot = axes[8, 1])

axes[9, 0].hist(autumn1.value, 25)
stats.probplot(autumn1.value, dist = "norm", plot = axes[9, 1])

axes[10, 0].hist(autumn2.value, 25)
stats.probplot(autumn2.value, dist = "norm", plot = axes[10, 1])

plt.subplots_adjust()

```

    C:\Users\Artem\Anaconda3\lib\site-packages\ipykernel_launcher.py:13: FutureWarning: 
    Passing list-likes to .loc or [] with any missing label will raise
    KeyError in the future, you can use .reindex() as an alternative.
    
    See the documentation here:
    https://pandas.pydata.org/pandas-docs/stable/indexing.html#deprecate-loc-reindex-listlike
      del sys.path[0]



![png](output_67_1.png)


Видим, что по каждому из сезонов распределение нормально<br>
Проверяем гипотезу о том, что имеют одинаковое среднее:


```python
stats.f_oneway(winter1.value, 
               winter2.value,
               winter3.value,
               summer1.value,
               summer2.value,
               summer3.value,
               spring1.value,
               spring2.value,
               spring3.value,
               autumn1.value,
               autumn2.value)
```




    F_onewayResult(statistic=13.062815942839908, pvalue=1.3389314618117005e-21)




```python
from scipy import stats
stats.kruskal(winter1.value, 
               winter2.value,
               winter3.value,
               summer1.value,
               summer2.value,
               summer3.value,
               spring1.value,
               spring2.value,
               spring3.value,
               autumn1.value,
               autumn2.value)
```




    KruskalResult(statistic=117.44207964952913, pvalue=1.670402812234698e-20)



Односторонний ANOVA проверяет нулевую гипотезу о том, что две или более группы имеют одно и то же среднее значение.
<br>for_check = 0.05*2/11! = 2.5052108385441724*^-9
<br>pvalue < for_check
<br>следовательно наша гипотеза неверна и сезон влияет на количество преступлений доставки

# Вывод

<br> Мы провели анализ количества преступлений в штате.
<br> В итоге мы установили, что праздники оказывают существенное влияние на количество преступлений
<br> (Особенно выделется 4.07 так как там происходит просто гигантский скачок количества преступлений)
<br> Так же было замечено, что количество преступлений зависит от сезона и приблизительно установиили эту зависимость
<br> Было установленно, что так же существует недельная сезональность
<br> В конце мы построили предсказания для разных моделей, ни одна из которых не дала желаемого результата


```python

```
