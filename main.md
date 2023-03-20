## Libraries


```python
# !pip install ctgan
# !pip install table_evaluator
```


```python
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:.2f}'.format
import scipy

import matplotlib.pyplot  as plt
import seaborn as sns
sns.set()

from sklearn.preprocessing import StandardScaler
from datetime import datetime

import scipy.io

# for one-hot encoder
from sklearn.preprocessing import OneHotEncoder

# for feature extraction
from sklearn.decomposition import PCA

#MinMaxScaler
from sklearn import preprocessing

# anomalous model
from sklearn.ensemble import IsolationForest

# !pip install ctgan
# for synthetic data generation
from ctgan import CTGAN

# !pip install table_evaluator
# for evaluation
from table_evaluator import TableEvaluator

# Visualising multidimensional data in 2D plane
# reeducing dimension, it has similar effect as PCA
from sklearn.manifold import TSNE

# confusion matrix
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
```

## Import Data


```python
csv_file = '2022-10-25 cdn_customer_qoe_anon.xlsx'
```


```python
from google.colab import drive
drive.mount('/content/drive')

df = pd.read_excel('/content/drive/My Drive/Colab Notebooks/DS_lab2/dataset/'+csv_file, 'cdn_customer_qoe_anon')
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
    


```python
# df = pd.read_excel('dataset/'+csv_file, 'cdn_customer_qoe_anon')
```


```python
df_buffer = df.copy() 
```


```python
df = df_buffer
```


```python
df
```





  <div id="df-18b5e5d0-fbee-4d11-8e39-6029d2d80407">
    <div class="colab-df-container">
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
      <th>Column1</th>
      <th>Start Time</th>
      <th>Playtime</th>
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>CDN Node Host</th>
      <th>Connection Type</th>
      <th>Device</th>
      <th>...</th>
      <th>End of Playback Status</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Program_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Content_TV_Show_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>2022-07-12 00:00:14</td>
      <td>11</td>
      <td>10</td>
      <td>0</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>Ethernet-100</td>
      <td>Android TV</td>
      <td>...</td>
      <td>On Stop</td>
      <td>564</td>
      <td>784</td>
      <td>0</td>
      <td>16</td>
      <td>64</td>
      <td>2672</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>2022-07-12 00:00:38</td>
      <td>73</td>
      <td>72</td>
      <td>0</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>...</td>
      <td>On Stop</td>
      <td>480</td>
      <td>1</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2022-07-12 00:02:02</td>
      <td>21</td>
      <td>20</td>
      <td>0</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>...</td>
      <td>On Stop</td>
      <td>346</td>
      <td>786</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>2022-07-12 00:02:24</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>...</td>
      <td>On Stop</td>
      <td>346</td>
      <td>997</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>2022-07-12 00:02:25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>...</td>
      <td>On Stop</td>
      <td>346</td>
      <td>997</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>102251</td>
      <td>2022-07-25 23:06:05</td>
      <td>15282</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>Mobile</td>
      <td>iPhone</td>
      <td>...</td>
      <td>On Stop</td>
      <td>570</td>
      <td>1504</td>
      <td>0</td>
      <td>2</td>
      <td>153</td>
      <td>2434</td>
      <td>3</td>
      <td>367</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>102252</td>
      <td>2022-07-25 22:55:39</td>
      <td>16582</td>
      <td>16581</td>
      <td>0</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>11377663</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>...</td>
      <td>On Stop</td>
      <td>475</td>
      <td>1014</td>
      <td>0</td>
      <td>13</td>
      <td>63</td>
      <td>2672</td>
      <td>3</td>
      <td>39</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>102253</td>
      <td>2022-07-25 23:09:33</td>
      <td>21166</td>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android</td>
      <td>...</td>
      <td>On Stop</td>
      <td>249</td>
      <td>1076</td>
      <td>0</td>
      <td>16</td>
      <td>41</td>
      <td>2672</td>
      <td>3</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>102254</td>
      <td>2022-07-25 11:47:37</td>
      <td>65122</td>
      <td>65115</td>
      <td>2</td>
      <td>6.10</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>None</td>
      <td>PC( Windows )</td>
      <td>...</td>
      <td>On Stop</td>
      <td>622</td>
      <td>1437</td>
      <td>0</td>
      <td>8</td>
      <td>158</td>
      <td>694</td>
      <td>3</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>102255</td>
      <td>2022-07-25 14:07:08</td>
      <td>75837</td>
      <td>75717</td>
      <td>0</td>
      <td>120.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android</td>
      <td>...</td>
      <td>On Stop</td>
      <td>101</td>
      <td>902</td>
      <td>0</td>
      <td>13</td>
      <td>15</td>
      <td>2672</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 33 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-18b5e5d0-fbee-4d11-8e39-6029d2d80407')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-18b5e5d0-fbee-4d11-8e39-6029d2d80407 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-18b5e5d0-fbee-4d11-8e39-6029d2d80407');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# bringing the Numerical valued attributes names
# ====================================================
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df.select_dtypes(include=numerics).columns

## 19 columns
```




    Index(['Column1', 'Playtime', 'Effective Playtime', 'Interruptions',
           'Join Time', 'Buffer Ratio', 'Happiness Score', 'Playback Stalls',
           'Startup Error (Count)', 'Latency', 'User_ID_N', 'Title_N', 'Program_N',
           'Device_Vendor_N', 'Device_Model_N', 'Content_TV_Show_N', 'Country_N',
           'City_N', 'Region_N'],
          dtype='object')




```python
# bringing the Categorical object valued attributes names
# ====================================================
df.select_dtypes(include=['object']).columns

## 12 columns
```




    Index(['CDN Node Host', 'Connection Type', 'Device', 'Device Type', 'Browser',
           'Browser Version', 'OS', 'OS Version', 'Device ID', 'Happiness Value',
           'Crash Status', 'End of Playback Status'],
          dtype='object')




```python
# Date attributes names
# ====================================================

# ['Start Time, 'End Time']

## 2 columns
```

## Explore Data

### plot correlation matrix


```python
plt.figure(figsize = (16, 9))
s = sns.heatmap(df.corr(),
                annot = True,
                cmap = 'RdBu',
                vmin = -1,
                vmax = 1)
s.set_yticklabels(s.get_yticklabels(), rotation = 0, fontsize = 12)
s.set_xticklabels(s.get_xticklabels(), rotation = 90, fontsize = 12)
plt.title('Correlation Heatmap')
plt.show()
```


    
![png](main_files/main_15_0.png)
    



```python
df[['Start Time', 'End Time']]

# the time series is observed over the period of 2 weeks
# the difference between start time and end time almost fixed: 2 months and 22 days
```





  <div id="df-aa731ff7-bdf3-4ab7-a5ff-b6b0d8b0c701">
    <div class="colab-df-container">
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
      <th>Start Time</th>
      <th>End Time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-07-12 00:00:14</td>
      <td>2022-10-04 00:00:26</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-07-12 00:00:38</td>
      <td>2022-10-04 00:01:52</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-07-12 00:02:02</td>
      <td>2022-10-04 00:02:24</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2022-07-12 00:02:24</td>
      <td>2022-10-04 00:02:26</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2022-07-12 00:02:25</td>
      <td>2022-10-04 00:02:28</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>2022-07-25 23:06:05</td>
      <td>2022-10-18 03:20:47</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>2022-07-25 22:55:39</td>
      <td>2022-10-18 03:32:02</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>2022-07-25 23:09:33</td>
      <td>2022-10-18 05:02:21</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>2022-07-25 11:47:37</td>
      <td>2022-10-18 05:53:00</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>2022-07-25 14:07:08</td>
      <td>2022-10-18 11:11:06</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-aa731ff7-bdf3-4ab7-a5ff-b6b0d8b0c701')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-aa731ff7-bdf3-4ab7-a5ff-b6b0d8b0c701 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-aa731ff7-bdf3-4ab7-a5ff-b6b0d8b0c701');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df.describe()
```





  <div id="df-1a826043-5fa3-48a3-8eda-81ae52ed8e96">
    <div class="colab-df-container">
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
      <th>Column1</th>
      <th>Playtime</th>
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Program_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Content_TV_Show_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
      <td>102256.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>51127.50</td>
      <td>328.97</td>
      <td>288.34</td>
      <td>0.10</td>
      <td>1.16</td>
      <td>0.26</td>
      <td>5.17</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>13360.82</td>
      <td>392.98</td>
      <td>809.66</td>
      <td>0.00</td>
      <td>13.13</td>
      <td>69.46</td>
      <td>2421.41</td>
      <td>3.93</td>
      <td>150.77</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>std</th>
      <td>29518.91</td>
      <td>1799.36</td>
      <td>1697.49</td>
      <td>12.01</td>
      <td>2.84</td>
      <td>3.57</td>
      <td>4.38</td>
      <td>0.20</td>
      <td>0.12</td>
      <td>23550.86</td>
      <td>161.31</td>
      <td>527.74</td>
      <td>0.00</td>
      <td>4.62</td>
      <td>33.35</td>
      <td>631.70</td>
      <td>2.55</td>
      <td>107.74</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>-1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>25563.75</td>
      <td>4.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.58</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>295.00</td>
      <td>261.00</td>
      <td>0.00</td>
      <td>13.00</td>
      <td>63.00</td>
      <td>2672.00</td>
      <td>3.00</td>
      <td>76.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>51127.50</td>
      <td>22.00</td>
      <td>17.00</td>
      <td>0.00</td>
      <td>0.79</td>
      <td>0.00</td>
      <td>6.65</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>17862.00</td>
      <td>383.00</td>
      <td>997.00</td>
      <td>0.00</td>
      <td>15.00</td>
      <td>64.00</td>
      <td>2672.00</td>
      <td>3.00</td>
      <td>76.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>76691.25</td>
      <td>90.00</td>
      <td>75.00</td>
      <td>0.00</td>
      <td>1.30</td>
      <td>0.00</td>
      <td>9.61</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>19235.00</td>
      <td>487.00</td>
      <td>1170.25</td>
      <td>0.00</td>
      <td>16.00</td>
      <td>64.00</td>
      <td>2672.00</td>
      <td>3.00</td>
      <td>240.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>102255.00</td>
      <td>86404.00</td>
      <td>86402.00</td>
      <td>3786.00</td>
      <td>120.00</td>
      <td>100.00</td>
      <td>10.00</td>
      <td>44.41</td>
      <td>1.00</td>
      <td>359477.00</td>
      <td>699.00</td>
      <td>1638.00</td>
      <td>0.00</td>
      <td>24.00</td>
      <td>163.00</td>
      <td>2746.00</td>
      <td>14.00</td>
      <td>405.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1a826043-5fa3-48a3-8eda-81ae52ed8e96')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1a826043-5fa3-48a3-8eda-81ae52ed8e96 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1a826043-5fa3-48a3-8eda-81ae52ed8e96');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df['Buffer Ratio'].value_counts()
```




    0.00      99621
    100.00       17
    0.01         11
    0.01          9
    0.01          8
              ...  
    18.23         1
    0.94          1
    15.89         1
    9.42          1
    0.26          1
    Name: Buffer Ratio, Length: 2076, dtype: int64




```python
BR_EP = df[['Buffer Ratio', 'Effective Playtime']].copy()
```


```python
df_buffer_ratio = BR_EP.groupby(['Buffer Ratio']).mean()
```


```python
df_buffer_ratio = df_buffer_ratio.sort_values(by=['Buffer Ratio'], ascending=True)
```


```python
df_buffer_ratio = df_buffer_ratio.reset_index()
```


```python
df_buffer_ratio
```





  <div id="df-ef0263b3-6671-426e-91d0-3d0da4721f8c">
    <div class="colab-df-container">
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
      <th>Buffer Ratio</th>
      <th>Effective Playtime</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>257.49</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>25369.80</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>8790.60</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>11359.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>11358.43</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2071</th>
      <td>98.69</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2072</th>
      <td>98.84</td>
      <td>3.00</td>
    </tr>
    <tr>
      <th>2073</th>
      <td>98.94</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>2074</th>
      <td>99.00</td>
      <td>2.00</td>
    </tr>
    <tr>
      <th>2075</th>
      <td>100.00</td>
      <td>0.00</td>
    </tr>
  </tbody>
</table>
<p>2076 rows × 2 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-ef0263b3-6671-426e-91d0-3d0da4721f8c')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-ef0263b3-6671-426e-91d0-3d0da4721f8c button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-ef0263b3-6671-426e-91d0-3d0da4721f8c');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
print((df_buffer_ratio.columns))
```

    Index(['Buffer Ratio', 'Effective Playtime'], dtype='object')
    

### QoE metrice effect on playtime


```python
plt.plot( df_buffer_ratio["Buffer Ratio"],df_buffer_ratio["Effective Playtime"], 'bo')
```




    [<matplotlib.lines.Line2D at 0x7f819f7f9d60>]




    
![png](main_files/main_26_1.png)
    



```python
# for better view
df_buffer_ratio_graph = df_buffer_ratio.loc[(df_buffer_ratio['Buffer Ratio'] < 40)]
df_buffer_ratio_graph = df_buffer_ratio.loc[(df_buffer_ratio['Effective Playtime'] < 8000)]
```


```python
plt.plot( df_buffer_ratio_graph["Buffer Ratio"],df_buffer_ratio_graph["Effective Playtime"], 'bo')
```




    [<matplotlib.lines.Line2D at 0x7f81774b9d30>]




    
![png](main_files/main_28_1.png)
    


it shows the effect of buffer ratio on the effective playing time


```python
# null values
null_values = df.isnull().sum()
df_nulls = pd.DataFrame({'Column':null_values.index, 'Nulls':null_values.values})

# Count non empty cells
df_count = df_nulls
df_count = df_count.rename(columns={"Nulls": "Count"})
nb_rows = df.shape[0] # 102256
df_count.iloc[:,1] = nb_rows - df_count.iloc[:,1]

# count unique cells
uniqueValues = df.nunique()
df_unique = pd.DataFrame({'Column':uniqueValues.index, 'Unique':uniqueValues.values})

```


```python
# merge all together in one dataframe
analyzed_df = pd.concat([df_nulls.iloc[:,0], df_count.iloc[:,1], df_nulls.iloc[:,1], df_unique.iloc[:,1]], axis = 1)
print(analyzed_df)
```

                        Column   Count   Nulls  Unique
    0                  Column1  102256       0  102256
    1               Start Time  102256       0   96503
    2                 Playtime  102256       0    4752
    3       Effective Playtime  102256       0    4440
    4            Interruptions  102256       0      43
    5                Join Time  102256       0    5554
    6             Buffer Ratio  102256       0    2076
    7            CDN Node Host   36979   65277     342
    8          Connection Type  102256       0      19
    9                   Device  102256       0      15
    10             Device Type  102256       0      13
    11                 Browser  102256       0      15
    12         Browser Version  101411     845      64
    13                      OS  102256       0       9
    14              OS Version  102256       0      78
    15               Device ID  102256       0    1692
    16         Happiness Value  102255       1       5
    17         Happiness Score  102256       0    6534
    18         Playback Stalls  102256       0     425
    19   Startup Error (Count)  102256       0       2
    20                 Latency  102256       0   13416
    21                End Time  102256       0   96251
    22            Crash Status    1845  100411       2
    23  End of Playback Status  102255       1       4
    24               User_ID_N  102256       0     700
    25                 Title_N  102256       0    1639
    26               Program_N  102256       0       1
    27         Device_Vendor_N  102256       0      25
    28          Device_Model_N  102256       0     164
    29       Content_TV_Show_N  102256       0    2747
    30               Country_N  102256       0      15
    31                  City_N  102256       0     406
    32                Region_N  102256       0       2
    

## Both are categorical: **1st** with strings, **2nd** with numbers, should do one-hot encoding except 'Happiness Score' because it is ordinal

'Connection Type', 'Device', 'Device Type', 'Browser', "Browser Version", 'OS', "OS Version", 'Crash Status', 'End of Playback Status', 'CDN Node Host'

'Device ID', 'Country_N', 'Region_N'

! 'Happiness Score' (will not)

! 'Content_TV_Show_N', will be removed due to high number of categories


```python
df['Interruptions'].value_counts()
```




    0       99615
    1        1941
    2         346
    3         135
    4          62
    5          30
    6          29
    7          21
    8          11
    10          8
    13          6
    12          4
    9           4
    11          4
    20          3
    21          3
    16          3
    15          2
    17          2
    24          2
    55          2
    18          2
    143         1
    25          1
    23          1
    44          1
    179         1
    66          1
    83          1
    125         1
    37          1
    85          1
    45          1
    43          1
    103         1
    26          1
    58          1
    489         1
    14          1
    3786        1
    27          1
    30          1
    188         1
    Name: Interruptions, dtype: int64




```python
# bringing the Numerical valued attributes names
# ====================================================
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df.select_dtypes(include=numerics).columns

## 19 columns
```




    Index(['Column1', 'Playtime', 'Effective Playtime', 'Interruptions',
           'Join Time', 'Buffer Ratio', 'Happiness Score', 'Playback Stalls',
           'Startup Error (Count)', 'Latency', 'User_ID_N', 'Title_N', 'Program_N',
           'Device_Vendor_N', 'Device_Model_N', 'Content_TV_Show_N', 'Country_N',
           'City_N', 'Region_N'],
          dtype='object')




```python
# bringing the Categorical object valued attributes names
# ====================================================
df.select_dtypes(include=['object']).columns

## 12 columns
```




    Index(['CDN Node Host', 'Connection Type', 'Device', 'Device Type', 'Browser',
           'Browser Version', 'OS', 'OS Version', 'Device ID', 'Happiness Value',
           'Crash Status', 'End of Playback Status'],
          dtype='object')



## Data Cleaning

remove 'Column1'


```python
df.drop(['Column1'], axis=1, inplace=True)
```

remove 'Program_N', it adds no value, all Zeros


```python
df.drop(['Program_N'], axis=1, inplace=True)
```

remove 'Playtime' due to high correlation with 'Effective Playtime'


```python
df.drop(['Playtime'], axis=1, inplace=True)
```

remove 'CDN Node Host' due to high percentage of missing values


```python
# df.drop(['CDN Node Host'], axis=1, inplace=True)
```

remove ' Happiness Value' as  Happiness score exists


```python
df.drop(['Happiness Value'], axis=1, inplace=True)
```

remove 'Start Time', 'End Time'
Because this in this approuch will not use time series in ADetection 


```python
df.drop(['Start Time'], axis=1, inplace=True)
df.drop(['End Time'], axis=1, inplace=True)
```

Removing 'Content_TV_Show_N' due to the high cardinality


```python
df.drop(['Content_TV_Show_N'], axis=1, inplace=True)
```

Removing 'Device ID'


```python
df.drop(['Device ID'], axis=1, inplace=True)
```

keeping the 'Crash Status' as the empty values means no crash happened


```python
df['Crash Status'].value_counts()
```




    Startup Error Crash      1276
    In Stream Error Crash     569
    Name: Crash Status, dtype: int64



# filling missing data

### !!! This step is skipped as missing data can be a data invention, which is usually unwanted in real-life envs.


```python
# df['Browser Version'].fillna(df['Browser Version'].mode()[0], inplace = True)
```


```python
# df['Crash Status'] = df['Crash Status'].fillna('no crash')
```


```python
# df['End of Playback Status'].fillna(df['End of Playback Status'].mode()[0], inplace = True)
```

Some histogram plotting 


```python
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
df_numerical_fts = df.select_dtypes(include=numerics).columns

fig, axes = plt.subplots(len(df[df_numerical_fts].columns)//3, 3, figsize=(12, 48))

i = 0
for triaxis in axes:
    for axis in triaxis:
        df[df_numerical_fts].hist(column = df[df_numerical_fts].columns[i], bins = 100, ax=axis)
        i = i+1
```


    
![png](main_files/main_61_0.png)
    


# Milestone


```python
# null values
null_values = df.isnull().sum()
df_nulls = pd.DataFrame({'Column':null_values.index, 'Nulls':null_values.values})

# Count non empty cells
df_count = df_nulls
df_count = df_count.rename(columns={"Nulls": "Count"})
nb_rows = df.shape[0] # 102256
df_count.iloc[:,1] = nb_rows - df_count.iloc[:,1]

# count unique cells
uniqueValues = df.nunique()
df_unique = pd.DataFrame({'Column':uniqueValues.index, 'Unique':uniqueValues.values})
```


```python
# merge all together in one dataframe
analyzed_df = pd.concat([df_nulls.iloc[:,0], df_count.iloc[:,1], df_nulls.iloc[:,1], df_unique.iloc[:,1]], axis = 1)
print(analyzed_df)
```

                        Column   Count   Nulls  Unique
    0       Effective Playtime  102256       0    4440
    1            Interruptions  102256       0      43
    2                Join Time  102256       0    5554
    3             Buffer Ratio  102256       0    2076
    4            CDN Node Host   36979   65277     342
    5          Connection Type  102256       0      19
    6                   Device  102256       0      15
    7              Device Type  102256       0      13
    8                  Browser  102256       0      15
    9          Browser Version  101411     845      64
    10                      OS  102256       0       9
    11              OS Version  102256       0      78
    12         Happiness Score  102256       0    6534
    13         Playback Stalls  102256       0     425
    14   Startup Error (Count)  102256       0       2
    15                 Latency  102256       0   13416
    16            Crash Status    1845  100411       2
    17  End of Playback Status  102255       1       4
    18               User_ID_N  102256       0     700
    19                 Title_N  102256       0    1639
    20         Device_Vendor_N  102256       0      25
    21          Device_Model_N  102256       0     164
    22               Country_N  102256       0      15
    23                  City_N  102256       0     406
    24                Region_N  102256       0       2
    


```python
df_2 = df.copy() 
```


```python
# df = df_2
```

# Encoding with get_dummies


```python
# to be encoded
df_categorical_fields_encod = ['Connection Type', 'Device', 'Device Type', 'Browser', "Browser Version", 'OS', "OS Version", 'Crash Status', 'End of Playback Status', 'Country_N', 'Region_N', 'CDN Node Host']
```


```python
df.shape
```




    (102256, 25)




```python
encoded_features =pd.get_dummies(data=df, columns= df_categorical_fields_encod)
```


```python
encoded_features
```





  <div id="df-b7cb05fe-a1cc-49a5-8eea-37a0bfed39fa">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>...</th>
      <th>CDN Node Host_e401ecd48</th>
      <th>CDN Node Host_e5a3a18d4</th>
      <th>CDN Node Host_ea23f8087</th>
      <th>CDN Node Host_ea8d72ad8</th>
      <th>CDN Node Host_ef71f4254</th>
      <th>CDN Node Host_efbbe67ab</th>
      <th>CDN Node Host_f058c444b</th>
      <th>CDN Node Host_fe41d9cfb</th>
      <th>CDN Node Host_feaa97dbf</th>
      <th>CDN Node Host_ff4e3e8ab</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>7.39</td>
      <td>0.00</td>
      <td>0</td>
      <td>19504</td>
      <td>564</td>
      <td>784</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72</td>
      <td>0</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>9.40</td>
      <td>0.00</td>
      <td>0</td>
      <td>19033</td>
      <td>480</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>7.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>19071</td>
      <td>346</td>
      <td>786</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>570</td>
      <td>1504</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>16581</td>
      <td>0</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>18191</td>
      <td>475</td>
      <td>1014</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>249</td>
      <td>1076</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>65115</td>
      <td>2</td>
      <td>6.10</td>
      <td>0.00</td>
      <td>7.46</td>
      <td>0.00</td>
      <td>0</td>
      <td>27550</td>
      <td>622</td>
      <td>1437</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>75717</td>
      <td>0</td>
      <td>120.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>36285</td>
      <td>101</td>
      <td>902</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 591 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-b7cb05fe-a1cc-49a5-8eea-37a0bfed39fa')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-b7cb05fe-a1cc-49a5-8eea-37a0bfed39fa button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-b7cb05fe-a1cc-49a5-8eea-37a0bfed39fa');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# drop the features that was encoeded from get_dummies
df.drop(df_categorical_fields_encod, axis=1 ,inplace=True)
```


```python
df=pd.concat([encoded_features,df], axis='columns')
```


```python
df.shape
```




    (102256, 604)




```python
df
```





  <div id="df-4d623d48-e771-43be-a222-306bf2313eb9">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>...</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>City_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>7.39</td>
      <td>0.00</td>
      <td>0</td>
      <td>19504</td>
      <td>564</td>
      <td>784</td>
      <td>...</td>
      <td>0.00</td>
      <td>7.39</td>
      <td>0.00</td>
      <td>0</td>
      <td>19504</td>
      <td>564</td>
      <td>784</td>
      <td>16</td>
      <td>64</td>
      <td>263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72</td>
      <td>0</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>9.40</td>
      <td>0.00</td>
      <td>0</td>
      <td>19033</td>
      <td>480</td>
      <td>1</td>
      <td>...</td>
      <td>0.00</td>
      <td>9.40</td>
      <td>0.00</td>
      <td>0</td>
      <td>19033</td>
      <td>480</td>
      <td>1</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>7.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>19071</td>
      <td>346</td>
      <td>786</td>
      <td>...</td>
      <td>0.00</td>
      <td>7.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>19071</td>
      <td>346</td>
      <td>786</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>570</td>
      <td>1504</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>570</td>
      <td>1504</td>
      <td>2</td>
      <td>153</td>
      <td>367</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>16581</td>
      <td>0</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>18191</td>
      <td>475</td>
      <td>1014</td>
      <td>...</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>18191</td>
      <td>475</td>
      <td>1014</td>
      <td>13</td>
      <td>63</td>
      <td>39</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>249</td>
      <td>1076</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>249</td>
      <td>1076</td>
      <td>16</td>
      <td>41</td>
      <td>56</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>65115</td>
      <td>2</td>
      <td>6.10</td>
      <td>0.00</td>
      <td>7.46</td>
      <td>0.00</td>
      <td>0</td>
      <td>27550</td>
      <td>622</td>
      <td>1437</td>
      <td>...</td>
      <td>0.00</td>
      <td>7.46</td>
      <td>0.00</td>
      <td>0</td>
      <td>27550</td>
      <td>622</td>
      <td>1437</td>
      <td>8</td>
      <td>158</td>
      <td>56</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>75717</td>
      <td>0</td>
      <td>120.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>36285</td>
      <td>101</td>
      <td>902</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>36285</td>
      <td>101</td>
      <td>902</td>
      <td>13</td>
      <td>15</td>
      <td>263</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 604 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-4d623d48-e771-43be-a222-306bf2313eb9')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-4d623d48-e771-43be-a222-306bf2313eb9 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-4d623d48-e771-43be-a222-306bf2313eb9');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




# Milestone


```python
df_3 = df.copy() 
```

## setup of anomalous detection

### min-max scaler for the model


```python
# MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df)
df = pd.DataFrame(df_scaled, columns=df.columns)
```


```python
df
```





  <div id="df-c8f7f28e-8d07-4b3c-9062-9ba230c6e66f">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>...</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>City_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.76</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.81</td>
      <td>0.48</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.76</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.81</td>
      <td>0.48</td>
      <td>0.67</td>
      <td>0.39</td>
      <td>0.65</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.95</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.69</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.95</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.69</td>
      <td>0.00</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.49</td>
      <td>0.48</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.49</td>
      <td>0.48</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.82</td>
      <td>0.92</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.82</td>
      <td>0.92</td>
      <td>0.08</td>
      <td>0.94</td>
      <td>0.91</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>0.19</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.68</td>
      <td>0.62</td>
      <td>...</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.68</td>
      <td>0.62</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.10</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.36</td>
      <td>0.66</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.36</td>
      <td>0.66</td>
      <td>0.67</td>
      <td>0.25</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>0.75</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.77</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.77</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>0.33</td>
      <td>0.97</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>0.88</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.55</td>
      <td>...</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.55</td>
      <td>0.54</td>
      <td>0.09</td>
      <td>0.65</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 604 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-c8f7f28e-8d07-4b3c-9062-9ba230c6e66f')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-c8f7f28e-8d07-4b3c-9062-9ba230c6e66f button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-c8f7f28e-8d07-4b3c-9062-9ba230c6e66f');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Generating of the Isolation Forest model and setting the parameters


```python
model =  IsolationForest(n_jobs=-1, n_estimators=200, max_features=3, random_state=42, contamination=0.01)
```

### Model fitting with the dataset we have for detecting anomaly


```python
model.fit(df)
```

    /usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but IsolationForest was fitted with feature names
      warnings.warn(
    




    IsolationForest(contamination=0.01, max_features=3, n_estimators=200, n_jobs=-1,
                    random_state=42)



### Adding the anomaly score after applying Isolation Forest to the data we have 


```python
df['Anomaly'] = pd.Series(model.predict(df))
```


```python
df.shape
```




    (102256, 605)




```python
df
```





  <div id="df-dfbbde69-46dc-436d-8c76-e1870643762d">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>...</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>City_N</th>
      <th>Anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.76</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.81</td>
      <td>0.48</td>
      <td>...</td>
      <td>0.76</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.81</td>
      <td>0.48</td>
      <td>0.67</td>
      <td>0.39</td>
      <td>0.65</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.95</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.69</td>
      <td>0.00</td>
      <td>...</td>
      <td>0.95</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.69</td>
      <td>0.00</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.49</td>
      <td>0.48</td>
      <td>...</td>
      <td>0.73</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.49</td>
      <td>0.48</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>...</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>...</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.49</td>
      <td>0.61</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.82</td>
      <td>0.92</td>
      <td>...</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.82</td>
      <td>0.92</td>
      <td>0.08</td>
      <td>0.94</td>
      <td>0.91</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>0.19</td>
      <td>0.00</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.68</td>
      <td>0.62</td>
      <td>...</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.68</td>
      <td>0.62</td>
      <td>0.54</td>
      <td>0.39</td>
      <td>0.10</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.36</td>
      <td>0.66</td>
      <td>...</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.36</td>
      <td>0.66</td>
      <td>0.67</td>
      <td>0.25</td>
      <td>0.14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>0.75</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>0.77</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>...</td>
      <td>0.77</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.08</td>
      <td>0.89</td>
      <td>0.88</td>
      <td>0.33</td>
      <td>0.97</td>
      <td>0.14</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>0.88</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.55</td>
      <td>...</td>
      <td>0.09</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>0.14</td>
      <td>0.55</td>
      <td>0.54</td>
      <td>0.09</td>
      <td>0.65</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 605 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-dfbbde69-46dc-436d-8c76-e1870643762d')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-dfbbde69-46dc-436d-8c76-e1870643762d button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-dfbbde69-46dc-436d-8c76-e1870643762d');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Labelling anomalous requests detected by Isolation Forest 
df['Anomaly'] = df['Anomaly'].map( {1: 0, -1: 1} )
```


```python
df['Anomaly'].value_counts()
```




    0    101233
    1      1023
    Name: Anomaly, dtype: int64




```python
tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=500, random_state=42)
tsne_results = tsne.fit_transform(df)
reduced_fts = np.array(tsne_results)

```

    /usr/local/lib/python3.8/dist-packages/sklearn/manifold/_t_sne.py:780: FutureWarning: The default initialization in TSNE will change from 'random' to 'pca' in 1.2.
      warnings.warn(
    /usr/local/lib/python3.8/dist-packages/sklearn/manifold/_t_sne.py:790: FutureWarning: The default learning rate in TSNE will change from 200.0 to 'auto' in 1.2.
      warnings.warn(
    


```python
reduced_fts = pd.DataFrame(reduced_fts, columns = ['ft_one','ft_two'])
```


```python
reduced_fts = pd.DataFrame(reduced_fts, columns = ['ft_one','ft_two'])
```


```python
# add the output to the reduced features
reduced_fts['Anomaly'] = df['Anomaly']
```


```python
reduced_fts
```





  <div id="df-bd35d7a9-108d-4a3b-8929-69df95adb107">
    <div class="colab-df-container">
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
      <th>ft_one</th>
      <th>ft_two</th>
      <th>Anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-2.8470</td>
      <td>-14.4195</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5377</td>
      <td>7.7299</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.6814</td>
      <td>1.1628</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15.5829</td>
      <td>-7.4792</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15.5827</td>
      <td>-7.4791</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>16.2048</td>
      <td>11.2491</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>4.4919</td>
      <td>3.4044</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>-7.9090</td>
      <td>8.6243</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>-3.3133</td>
      <td>6.3363</td>
      <td>1</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>-6.2647</td>
      <td>7.9329</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 3 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-bd35d7a9-108d-4a3b-8929-69df95adb107')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-bd35d7a9-108d-4a3b-8929-69df95adb107 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-bd35d7a9-108d-4a3b-8929-69df95adb107');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
# Scatterplot visualisation 
sns.scatterplot(
    x="ft_one", y="ft_two",
    hue="Anomaly",
    data= reduced_fts,
    legend="full",
    alpha=1
)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f810a0d34f0>




    
![png](main_files/main_97_1.png)
    


## Dispaly the properites of anomalous records



```python
df_buffer
```





  <div id="df-30f146ac-0473-425d-9141-be7944855bee">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>City_N</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>7.39</td>
      <td>0.00</td>
      <td>0</td>
      <td>19504</td>
      <td>564</td>
      <td>784</td>
      <td>16</td>
      <td>64</td>
      <td>263</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72</td>
      <td>0</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>9.40</td>
      <td>0.00</td>
      <td>0</td>
      <td>19033</td>
      <td>480</td>
      <td>1</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>7.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>19071</td>
      <td>346</td>
      <td>786</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>76</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>570</td>
      <td>1504</td>
      <td>2</td>
      <td>153</td>
      <td>367</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>16581</td>
      <td>0</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>10.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>18191</td>
      <td>475</td>
      <td>1014</td>
      <td>13</td>
      <td>63</td>
      <td>39</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0</td>
      <td>0</td>
      <td>249</td>
      <td>1076</td>
      <td>16</td>
      <td>41</td>
      <td>56</td>
    </tr>
    <tr>
      <th>102254</th>
      <td>65115</td>
      <td>2</td>
      <td>6.10</td>
      <td>0.00</td>
      <td>7.46</td>
      <td>0.00</td>
      <td>0</td>
      <td>27550</td>
      <td>622</td>
      <td>1437</td>
      <td>8</td>
      <td>158</td>
      <td>56</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>75717</td>
      <td>0</td>
      <td>120.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1</td>
      <td>36285</td>
      <td>101</td>
      <td>902</td>
      <td>13</td>
      <td>15</td>
      <td>263</td>
    </tr>
  </tbody>
</table>
<p>102256 rows × 13 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-30f146ac-0473-425d-9141-be7944855bee')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-30f146ac-0473-425d-9141-be7944855bee button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-30f146ac-0473-425d-9141-be7944855bee');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df_2_all = df_2.copy()
```


```python
df_2_all['Anomaly'] = df['Anomaly']
# df_2_all['Column1'] = df_buffer['Column1']
```


```python
df_anom = df_2_all[df_2_all['Anomaly'] == 1]
```


```python
df_buffer['Interruptions'].value_counts()
# df_buffer['Country_N'].value_counts()
```




    0       99615
    1        1941
    2         346
    3         135
    4          62
    5          30
    6          29
    7          21
    8          11
    10          8
    13          6
    12          4
    9           4
    11          4
    20          3
    21          3
    16          3
    15          2
    17          2
    24          2
    55          2
    18          2
    143         1
    25          1
    23          1
    44          1
    179         1
    66          1
    83          1
    125         1
    37          1
    85          1
    45          1
    43          1
    103         1
    26          1
    58          1
    489         1
    14          1
    3786        1
    27          1
    30          1
    188         1
    Name: Interruptions, dtype: int64




```python
df_anom['Interruptions'].value_counts()
# df_anom['Country_N'].value_counts()
```




    1       535
    0       234
    2       111
    3        51
    4        20
    5        12
    6        11
    7        10
    8         5
    13        3
    20        2
    10        2
    18        2
    16        2
    9         2
    55        2
    24        2
    23        1
    179       1
    83        1
    143       1
    15        1
    188       1
    21        1
    103       1
    66        1
    25        1
    85        1
    26        1
    125       1
    489       1
    3786      1
    12        1
    17        1
    Name: Interruptions, dtype: int64




```python
df_anom.describe()
```





  <div id="df-1a7e2f8b-002b-4386-9117-937fab962d78">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
      <th>Anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
      <td>1023.00</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1463.87</td>
      <td>6.80</td>
      <td>2.20</td>
      <td>9.38</td>
      <td>4.65</td>
      <td>0.08</td>
      <td>0.05</td>
      <td>26838.03</td>
      <td>444.55</td>
      <td>843.67</td>
      <td>9.73</td>
      <td>91.80</td>
      <td>6.39</td>
      <td>198.85</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>std</th>
      <td>5441.36</td>
      <td>119.83</td>
      <td>3.16</td>
      <td>21.77</td>
      <td>2.70</td>
      <td>0.21</td>
      <td>0.22</td>
      <td>49984.32</td>
      <td>183.42</td>
      <td>535.55</td>
      <td>5.16</td>
      <td>63.61</td>
      <td>3.58</td>
      <td>118.65</td>
      <td>0.00</td>
      <td>0.00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>6.00</td>
      <td>1.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>8.00</td>
      <td>1.00</td>
      <td>0.98</td>
      <td>0.01</td>
      <td>3.10</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>354.50</td>
      <td>261.00</td>
      <td>8.00</td>
      <td>17.00</td>
      <td>3.00</td>
      <td>76.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>32.00</td>
      <td>1.00</td>
      <td>1.53</td>
      <td>0.74</td>
      <td>4.50</td>
      <td>0.01</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>438.00</td>
      <td>937.00</td>
      <td>8.00</td>
      <td>72.00</td>
      <td>4.00</td>
      <td>167.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>525.00</td>
      <td>1.00</td>
      <td>2.58</td>
      <td>6.32</td>
      <td>7.13</td>
      <td>0.04</td>
      <td>0.00</td>
      <td>32180.00</td>
      <td>589.00</td>
      <td>1412.00</td>
      <td>15.00</td>
      <td>158.00</td>
      <td>11.00</td>
      <td>314.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>83321.00</td>
      <td>3786.00</td>
      <td>42.65</td>
      <td>99.00</td>
      <td>9.99</td>
      <td>0.99</td>
      <td>1.00</td>
      <td>333214.00</td>
      <td>699.00</td>
      <td>1638.00</td>
      <td>23.00</td>
      <td>163.00</td>
      <td>11.00</td>
      <td>391.00</td>
      <td>0.00</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1a7e2f8b-002b-4386-9117-937fab962d78')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1a7e2f8b-002b-4386-9117-937fab962d78 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1a7e2f8b-002b-4386-9117-937fab962d78');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
plt.plot( df_anom["Buffer Ratio"],df_anom["Effective Playtime"], 'bo')
```




    [<matplotlib.lines.Line2D at 0x7f817b2afd30>]




    
![png](main_files/main_106_1.png)
    


Statisitcal analysis:
--------------------
anomaly = 1023;
normal = 101233;

anomaly % = (anomaly/(anomaly + normal)) * 100 = (1023/102256) * 100 = 1.00043 ~ 1 %

#########



#########

# Synthetic data gerneration
in this task, two spereate sets of data will be generated: **anomalous** and **non-anomalous** .

Note: theb data generated is the usfeull, meaningfull data after cleaning and preprocessing that wil be usefull for the model


```python
original_set = df_2.copy()
```


```python
original_set['Anomaly'] = df['Anomaly']
```


```python
original_set_without_anom = original_set.loc[original_set['Anomaly'] == 0]
original_set_with_anom = original_set.loc[original_set['Anomaly'] == 1]
```

Prepare fields for the model, the discrete fields should be passed through an array.


```python
# Names of the columns that are discrete
discrete_columns =  [
    'Connection Type', 
    'Device', 
    'Device Type', 
    'Browser', 
    'Browser Version', 
    'OS', 
    'OS Version', 
    'Crash Status', 
    'End of Playback Status', 
    'Country_N', 
    'Region_N',
    'CDN Node Host'
    ]
```


```python
print(original_set_without_anom.columns[0:8])
```

    Index(['Effective Playtime', 'Interruptions', 'Join Time', 'Buffer Ratio',
           'CDN Node Host', 'Connection Type', 'Device', 'Device Type'],
          dtype='object')
    


```python
print(original_set_without_anom.columns[8:16])
```

    Index(['Browser', 'Browser Version', 'OS', 'OS Version', 'Happiness Score',
           'Playback Stalls', 'Startup Error (Count)', 'Latency'],
          dtype='object')
    


```python
original_set_without_anom
```





  <div id="df-005701b4-aaf8-4db0-be5a-e34f7a63c55a">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>CDN Node Host</th>
      <th>Connection Type</th>
      <th>Device</th>
      <th>Device Type</th>
      <th>Browser</th>
      <th>Browser Version</th>
      <th>...</th>
      <th>Crash Status</th>
      <th>End of Playback Status</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>Country_N</th>
      <th>City_N</th>
      <th>Region_N</th>
      <th>Anomaly</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10</td>
      <td>0</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>Ethernet-100</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>564</td>
      <td>784</td>
      <td>16</td>
      <td>64</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>72</td>
      <td>0</td>
      <td>1.17</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>480</td>
      <td>1</td>
      <td>13</td>
      <td>63</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>20</td>
      <td>0</td>
      <td>1.13</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>346</td>
      <td>786</td>
      <td>13</td>
      <td>63</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>346</td>
      <td>997</td>
      <td>13</td>
      <td>63</td>
      <td>3</td>
      <td>76</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>102249</th>
      <td>13211</td>
      <td>1</td>
      <td>41.12</td>
      <td>0.26</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>In Stream Error Crash</td>
      <td>On Error</td>
      <td>224</td>
      <td>1412</td>
      <td>13</td>
      <td>63</td>
      <td>3</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102251</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>Mobile</td>
      <td>iPhone</td>
      <td>SmartPhone</td>
      <td>Mobile Safari</td>
      <td>Mobile Safari</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>570</td>
      <td>1504</td>
      <td>2</td>
      <td>153</td>
      <td>3</td>
      <td>367</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102252</th>
      <td>16581</td>
      <td>0</td>
      <td>0.99</td>
      <td>0.00</td>
      <td>11377663</td>
      <td>WiFi-5</td>
      <td>Android TV</td>
      <td>TV</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>475</td>
      <td>1014</td>
      <td>13</td>
      <td>63</td>
      <td>3</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102253</th>
      <td>0</td>
      <td>0</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android</td>
      <td>STBAndroid</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>249</td>
      <td>1076</td>
      <td>16</td>
      <td>41</td>
      <td>3</td>
      <td>56</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>102255</th>
      <td>75717</td>
      <td>0</td>
      <td>120.00</td>
      <td>0.00</td>
      <td>NaN</td>
      <td>WiFi-5</td>
      <td>Android</td>
      <td>STBAndroid</td>
      <td>Android Browser</td>
      <td>Android Browser</td>
      <td>...</td>
      <td>NaN</td>
      <td>On Stop</td>
      <td>101</td>
      <td>902</td>
      <td>13</td>
      <td>15</td>
      <td>3</td>
      <td>263</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>101233 rows × 26 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-005701b4-aaf8-4db0-be5a-e34f7a63c55a')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-005701b4-aaf8-4db0-be5a-e34f7a63c55a button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-005701b4-aaf8-4db0-be5a-e34f7a63c55a');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




### Building the CTGAN model and setting the epochs which is total number of iterations of all the training data in one cycle for training the machine learning model.
### After model is trained, **9900** record is generated.


```python
ctgan = CTGAN(epochs=6)
ctgan.fit(original_set_without_anom, discrete_columns)

# Create synthetic data
synthetic_data_without_anom = ctgan.sample(9900)
```

    /usr/local/lib/python3.8/dist-packages/joblib/externals/loky/process_executor.py:700: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      warnings.warn(
    

Saving model for later use to avoid training time.


```python
ctgan.save('CTGAN_model.pkl')
```


```python
# loaded = CTGAN.load('CTGAN_model.pkl')
# synthetic_data_without_anom = loaded.sample(9900)
```

### Building the CTGAN model and setting the epochs which is total number of iterations of all the training data in one cycle for training the machine learning model.
### After model is trained, **100** record is generated.


```python
ctgan_anom = CTGAN(epochs=15)
ctgan_anom.fit(original_set_with_anom, discrete_columns)

# Create synthetic data
synthetic_data_with_anom = ctgan_anom.sample(100)
```

Saving model for later use to avoid training time.


```python
ctgan_anom.save('CTGAN_model_anom.pkl')
```


```python
# ctgan_anom = CTGAN.load('CTGAN_model_anom.pkl')
# synthetic_data_with_anom = ctgan_anom.sample(100)
```

## Add data to each other


```python
synthetic_data = synthetic_data_without_anom.append(synthetic_data_with_anom, ignore_index=True)
```

# for testing similarity

## in the whole features (normal)


```python
table_evaluator = TableEvaluator(original_set_without_anom, synthetic_data_without_anom)
table_evaluator.visual_evaluation()
```


    
![png](main_files/main_132_0.png)
    



    
![png](main_files/main_132_1.png)
    



    
![png](main_files/main_132_2.png)
    



    
![png](main_files/main_132_3.png)
    



    
![png](main_files/main_132_4.png)
    


## in the specific feature (normal)


```python
table_evaluator.evaluate(target_col='Effective Playtime')
```

    /usr/local/lib/python3.8/dist-packages/scipy/stats/stats.py:4023: PearsonRConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
      warnings.warn(PearsonRConstantInputWarning())
    

    
    Classifier F1-scores and their Jaccard similarities::
                                 f1_real  f1_fake  jaccard_similarity
    index                                                            
    DecisionTreeClassifier_fake   0.0136   0.0061              0.0076
    DecisionTreeClassifier_real   0.3662   0.0030              0.0023
    LogisticRegression_fake       0.0121   0.0197              0.0238
    LogisticRegression_real       0.2439   0.0247              0.0056
    MLPClassifier_fake            0.0086   0.0121              0.0015
    MLPClassifier_real            0.2051   0.0086              0.0008
    RandomForestClassifier_fake   0.0131   0.0096              0.0061
    RandomForestClassifier_real   0.3818   0.0111              0.0053
    
    Privacy results:
                                               result
    Duplicate rows between sets (real/fake)  (483, 0)
    nearest neighbor mean                      2.6278
    nearest neighbor std                       1.6904
    
    Miscellaneous results:
                                      Result
    Column Correlation Distance RMSE  0.2172
    Column Correlation distance MAE   0.1276
    
    Results:
                                                    result
    Basic statistics                                0.9663
    Correlation column correlations                 0.6884
    Mean Correlation between fake and real columns     NaN
    1 - MAPE Estimator results                      0.2899
    Similarity Score                                   NaN
    

## in the whole features (anomalous)


```python
table_evaluator_anom = TableEvaluator(original_set_with_anom, synthetic_data_with_anom)
table_evaluator_anom.visual_evaluation()
```


    
![png](main_files/main_136_0.png)
    



    
![png](main_files/main_136_1.png)
    



    
![png](main_files/main_136_2.png)
    



    
![png](main_files/main_136_3.png)
    



    
![png](main_files/main_136_4.png)
    


## in the specific feature (anomalous)


```python
table_evaluator_anom.evaluate(target_col='Effective Playtime')
```

    /usr/local/lib/python3.8/dist-packages/scipy/stats/stats.py:4023: PearsonRConstantInputWarning: An input array is constant; the correlation coefficient is not defined.
      warnings.warn(PearsonRConstantInputWarning())
    /usr/local/lib/python3.8/dist-packages/table_evaluator/metrics.py:42: RuntimeWarning: invalid value encountered in true_divide
      return np.mean(np.abs((y_true - y_pred) / y_true))
    

    
    Classifier F1-scores and their Jaccard similarities::
                                 f1_real  f1_fake  jaccard_similarity
    index                                                            
    DecisionTreeClassifier_fake   0.0000   0.0000              0.0000
    DecisionTreeClassifier_real   0.0500   0.0000              0.0000
    LogisticRegression_fake       0.0000   0.0000              0.0000
    LogisticRegression_real       0.0000   0.0000              0.0000
    MLPClassifier_fake            0.0000   0.0000              0.0000
    MLPClassifier_real            0.0500   0.0000              0.0000
    RandomForestClassifier_fake   0.0000   0.0000              0.0000
    RandomForestClassifier_real   0.0500   0.0000              0.0000
    
    Privacy results:
                                             result
    Duplicate rows between sets (real/fake)  (0, 0)
    nearest neighbor mean                    4.1469
    nearest neighbor std                     1.1101
    
    Miscellaneous results:
                                      Result
    Column Correlation Distance RMSE  0.2606
    Column Correlation distance MAE   0.1669
    
    Results:
                                                    result
    Basic statistics                                0.8952
    Correlation column correlations                 0.5117
    Mean Correlation between fake and real columns     NaN
    1 - MAPE Estimator results                         NaN
    Similarity Score                                   NaN
    

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

# Validation through anomaly detection 

# Encoding with get_dummies


```python
synthetic_data_cpy = synthetic_data.copy() 
```


```python
# to be encoded
df_categorical_fields_encod = ['Connection Type', 'Device', 'Device Type', 'Browser', "Browser Version", 'OS', "OS Version", 'Crash Status', 'End of Playback Status', 'Country_N', 'Region_N', 'CDN Node Host']
```


```python
synthetic_data_cpy.shape
```




    (10000, 26)




```python
encoded_features = pd.get_dummies(data=synthetic_data_cpy, columns= df_categorical_fields_encod)
```


```python
# drop the features that was encoeded from get_dummies
synthetic_data_cpy.drop(df_categorical_fields_encod, axis=1 ,inplace=True)
```


```python
df_synthetic_data = pd.concat([encoded_features,synthetic_data_cpy], axis='columns')
```


```python
df_synthetic_data.shape
```




    (10000, 505)



### min-max scaler for the model


```python
# MinMaxScaler
min_max_scaler = preprocessing.MinMaxScaler()
df_scaled = min_max_scaler.fit_transform(df_synthetic_data)
df_synthetic_data = pd.DataFrame(df_scaled, columns=df_synthetic_data.columns)
```


```python
df_synthetic_data.drop(['Anomaly'], axis=1, inplace=True)
```

### Generating of the Isolation Forest model and setting the parameters


```python
model =  IsolationForest(n_jobs=-1, n_estimators=200, max_features=3, random_state=42, contamination=0.01)
```

### Model fitting with the dataset we have for detecting anomaly


```python
model.fit(df_synthetic_data)
```

    /usr/local/lib/python3.8/dist-packages/sklearn/base.py:450: UserWarning: X does not have valid feature names, but IsolationForest was fitted with feature names
      warnings.warn(
    




    IsolationForest(contamination=0.01, max_features=3, n_estimators=200, n_jobs=-1,
                    random_state=42)



### Adding the anomaly score after applying Isolation Forest to the data we have 


```python
df_synthetic_data['Anomaly_pred'] = pd.Series(model.predict(df_synthetic_data))
```


```python
df_synthetic_data.shape
```




    (10000, 504)




```python
# Labelling anomalous requests detected by Isolation Forest 
df_synthetic_data['Anomaly_pred'] = df_synthetic_data['Anomaly_pred'].map( {1: 0, -1: 1} )
```


```python
df_synthetic_data['Anomaly_pred'].value_counts()
```




    0    9900
    1     100
    Name: Anomaly_pred, dtype: int64




```python
df_synthetic_data
```





  <div id="df-f38f3f0f-1638-41bb-84f1-1f54720f3b86">
    <div class="colab-df-container">
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
      <th>Effective Playtime</th>
      <th>Interruptions</th>
      <th>Join Time</th>
      <th>Buffer Ratio</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>...</th>
      <th>Happiness Score</th>
      <th>Playback Stalls</th>
      <th>Startup Error (Count)</th>
      <th>Latency</th>
      <th>User_ID_N</th>
      <th>Title_N</th>
      <th>Device_Vendor_N</th>
      <th>Device_Model_N</th>
      <th>City_N</th>
      <th>Anomaly_pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0089</td>
      <td>0.6604</td>
      <td>0.0284</td>
      <td>0.0105</td>
      <td>0.1351</td>
      <td>0.0238</td>
      <td>0.0000</td>
      <td>0.2222</td>
      <td>0.5653</td>
      <td>0.2148</td>
      <td>...</td>
      <td>0.1351</td>
      <td>0.0238</td>
      <td>0.0000</td>
      <td>0.2222</td>
      <td>0.5653</td>
      <td>0.2148</td>
      <td>0.2188</td>
      <td>0.1093</td>
      <td>0.2676</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0086</td>
      <td>0.6604</td>
      <td>0.0243</td>
      <td>0.0107</td>
      <td>0.1333</td>
      <td>0.0210</td>
      <td>0.0000</td>
      <td>0.2229</td>
      <td>0.4008</td>
      <td>0.6562</td>
      <td>...</td>
      <td>0.1333</td>
      <td>0.0210</td>
      <td>0.0000</td>
      <td>0.2229</td>
      <td>0.4008</td>
      <td>0.6562</td>
      <td>0.2188</td>
      <td>0.7760</td>
      <td>0.8370</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0087</td>
      <td>0.6604</td>
      <td>0.0538</td>
      <td>0.0106</td>
      <td>0.1344</td>
      <td>0.0218</td>
      <td>0.0000</td>
      <td>0.2236</td>
      <td>0.8825</td>
      <td>0.6668</td>
      <td>...</td>
      <td>0.1344</td>
      <td>0.0218</td>
      <td>0.0000</td>
      <td>0.2236</td>
      <td>0.8825</td>
      <td>0.6668</td>
      <td>0.2188</td>
      <td>0.5683</td>
      <td>0.9034</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0089</td>
      <td>0.6604</td>
      <td>0.0248</td>
      <td>0.0107</td>
      <td>0.8280</td>
      <td>0.0232</td>
      <td>0.0000</td>
      <td>0.2234</td>
      <td>0.5522</td>
      <td>0.9202</td>
      <td>...</td>
      <td>0.8280</td>
      <td>0.0232</td>
      <td>0.0000</td>
      <td>0.2234</td>
      <td>0.5522</td>
      <td>0.9202</td>
      <td>0.2188</td>
      <td>0.9399</td>
      <td>0.2233</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0087</td>
      <td>0.6604</td>
      <td>0.0951</td>
      <td>0.0107</td>
      <td>0.1344</td>
      <td>0.0239</td>
      <td>0.0000</td>
      <td>0.2237</td>
      <td>0.8734</td>
      <td>0.6859</td>
      <td>...</td>
      <td>0.1344</td>
      <td>0.0239</td>
      <td>0.0000</td>
      <td>0.2237</td>
      <td>0.8734</td>
      <td>0.6859</td>
      <td>0.2188</td>
      <td>0.9508</td>
      <td>0.9014</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>0.4269</td>
      <td>0.6321</td>
      <td>0.1482</td>
      <td>0.3772</td>
      <td>0.4843</td>
      <td>0.0362</td>
      <td>0.0000</td>
      <td>0.2142</td>
      <td>0.0078</td>
      <td>0.3850</td>
      <td>...</td>
      <td>0.4843</td>
      <td>0.0362</td>
      <td>0.0000</td>
      <td>0.2142</td>
      <td>0.0078</td>
      <td>0.3850</td>
      <td>0.6562</td>
      <td>0.0437</td>
      <td>0.2636</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>0.0105</td>
      <td>0.6274</td>
      <td>1.0000</td>
      <td>0.0466</td>
      <td>0.6468</td>
      <td>0.0323</td>
      <td>0.0000</td>
      <td>0.1797</td>
      <td>0.5235</td>
      <td>0.6494</td>
      <td>...</td>
      <td>0.6468</td>
      <td>0.0323</td>
      <td>0.0000</td>
      <td>0.1797</td>
      <td>0.5235</td>
      <td>0.6494</td>
      <td>0.6562</td>
      <td>0.0437</td>
      <td>0.2274</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>0.0127</td>
      <td>0.6132</td>
      <td>0.0851</td>
      <td>0.0184</td>
      <td>0.4261</td>
      <td>0.0390</td>
      <td>0.0000</td>
      <td>0.1822</td>
      <td>0.5783</td>
      <td>0.9482</td>
      <td>...</td>
      <td>0.4261</td>
      <td>0.0390</td>
      <td>0.0000</td>
      <td>0.1822</td>
      <td>0.5783</td>
      <td>0.9482</td>
      <td>0.4375</td>
      <td>0.9071</td>
      <td>0.8350</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>0.2449</td>
      <td>0.6557</td>
      <td>0.1254</td>
      <td>0.0185</td>
      <td>0.6328</td>
      <td>0.0395</td>
      <td>0.0000</td>
      <td>0.1931</td>
      <td>0.5444</td>
      <td>0.3166</td>
      <td>...</td>
      <td>0.6328</td>
      <td>0.0395</td>
      <td>0.0000</td>
      <td>0.1931</td>
      <td>0.5444</td>
      <td>0.3166</td>
      <td>0.6250</td>
      <td>0.0492</td>
      <td>0.2636</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>0.0168</td>
      <td>0.6179</td>
      <td>0.0225</td>
      <td>0.0087</td>
      <td>0.9415</td>
      <td>0.0513</td>
      <td>0.0000</td>
      <td>0.2309</td>
      <td>0.2272</td>
      <td>0.3472</td>
      <td>...</td>
      <td>0.9415</td>
      <td>0.0513</td>
      <td>0.0000</td>
      <td>0.2309</td>
      <td>0.2272</td>
      <td>0.3472</td>
      <td>0.4375</td>
      <td>0.9399</td>
      <td>0.4245</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 504 columns</p>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-f38f3f0f-1638-41bb-84f1-1f54720f3b86')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-f38f3f0f-1638-41bb-84f1-1f54720f3b86 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-f38f3f0f-1638-41bb-84f1-1f54720f3b86');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
df_synthetic_data['Anomaly'] = synthetic_data['Anomaly']
```


```python
df_confusion = pd.crosstab(df_synthetic_data['Anomaly'], df_synthetic_data['Anomaly_pred'] )
```

### Building confusion matrix


```python
df_confusion
```





  <div id="df-34195ac8-baff-4ad9-a6dc-e40510447095">
    <div class="colab-df-container">
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
      <th>Anomaly_pred</th>
      <th>0</th>
      <th>1</th>
    </tr>
    <tr>
      <th>Anomaly</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9884</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>16</td>
      <td>84</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-34195ac8-baff-4ad9-a6dc-e40510447095')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-34195ac8-baff-4ad9-a6dc-e40510447095 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-34195ac8-baff-4ad9-a6dc-e40510447095');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
conf_m = metrics.confusion_matrix(df_synthetic_data['Anomaly'], df_synthetic_data['Anomaly_pred'])
print(conf_m)
sns.heatmap(conf_m, annot = True,  linewidths=.5, cbar =None, fmt='g', cmap='Greens')
plt.title('Synthetic data anomaly detection confusion matrix')
```

    [[9884   16]
     [  16   84]]
    




    Text(0.5, 1.0, 'Synthetic data anomaly detection confusion matrix')




    
![png](main_files/main_166_2.png)
    


### Calculating precision and recall


```python
precision = precision_score(df_synthetic_data['Anomaly'], df_synthetic_data['Anomaly_pred'])
recall = recall_score(df_synthetic_data['Anomaly'], df_synthetic_data['Anomaly_pred'])
```


```python
precision
```




    0.84




```python
recall
```




    0.84




```python
df_synthetic_data.to_csv('df_synthetic_data.csv')
```
