import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv('data/Bank_Personal_Loan_Modelling.csv')
target_column = 'Personal Loan'

split = StratifiedShuffleSplit(n_splits=1, train_size=4000, test_size=1000, random_state=42)

for train_index, test_index in split.split(data, data[target_column]):
    train_data = data.iloc[train_index]
    test_data = data.iloc[test_index]

def random_date(start, end, n):
    delta = end - start
    random_days = np.random.randint(0, delta.days, size=n)
    return start + pd.to_timedelta(random_days, unit='D')

# Generate random dates (may to august)
start_train = datetime(2023, 5, 1)  
end_train = datetime(2023, 8, 31)   
train_data['Date'] = random_date(start_train, end_train, len(train_data))

# Generate random dates (september)
start_test = datetime(2023, 9, 1) 
end_test = datetime(2023, 9, 30)    
test_data['Date'] = random_date(start_test, end_test, len(test_data))

# add new column with constant values 
# train_data['Constant column'] = 100
# test_data['Constant column'] = 100

# add new random columnì
# train_data['Random column'] = np.random.rand(len(train_data))
# test_data['Random column'] = np.random.rand(len(test_data))

train_data.to_csv('data/train_data.csv', index=False)
test_data.to_csv('data/test_data.csv', index=False)

# add Nans to Education
if 'Education' in train_data.columns:
    random_indices = np.random.choice(train_data.index, size=100, replace=False)
    train_data.loc[random_indices, 'Education'] = np.nan
    train_data.to_csv('data/train_data.csv', index=False)
    print("100 null values added to the 'Education' column and dataset saved.")
else:
    print("'Education' column not found in the dataset.")


print('Train Data:')
train = pd.read_csv("data/train_data.csv")
print(train['Personal Loan'].value_counts())
print(train.head())

print('Test Data:')
test = pd.read_csv("data/test_data.csv")
print(test['Personal Loan'].value_counts())
print(test.head())
