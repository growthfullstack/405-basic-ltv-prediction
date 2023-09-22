import pandas as pd
import numpy as np
from scipy import stats

data = datasets["campaigns_ltv_monitoring"]

#model 1 with 3 days of input
x_3 = [1,2,3]
log_x_3 = np.log(x_3)

def ltv_3days_data(day, line):
  slope, intercept, r_value, p_value, std_err = stats.linregress(log_x_3, list(line))
  return intercept + np.log(day) * slope

n = len(data)
df = pd.DataFrame()

for i in range(n):
  test_campaign = data.iloc[i,2:5]
  y_pred = ltv_3days_data(30,test_campaign)
  
  original_data = list(data.iloc[i,:5])
  original_data.append(y_pred)
  original_data.append((data.iloc[i,1] - y_pred)/data.iloc[i,1])
  df = df.append([original_data],ignore_index = True)

print(df[1].describe())
print(df)

#model 1 with 7 days of input
x_7 = [1,2,3,4,5,6,7]
log_x_7 = np.log(x_7)

def ltv_7days_data(day, line):
  slope, intercept, r_value, p_value, std_err = stats.linregress(log_x_7, list(line))
  return intercept + np.log(day) * slope

n = len(data)
df = pd.DataFrame()

for i in range(n):
  test_campaign = data.iloc[i,2:9]
  y_pred = ltv_7days_data(30,test_campaign)
  original_data = list(data.iloc[i,:])
  original_data.append(y_pred)
  original_data.append((data.iloc[i,1] - y_pred)/data.iloc[i,1])
  df = df.append([original_data],ignore_index = True)

print(df[1].describe())
print(df)