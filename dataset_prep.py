import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import stats


data = pd.read_csv('data/Bank_Personal_Loan_Modelling.csv')
target_column = 'Personal Loan'

split = StratifiedShuffleSplit(n_splits=1, train_size=4000, test_size=1000, random_state=42)

for train_index, test_index in split.split(data, data[target_column]):
    train_data = data.iloc[train_index]
    validation_data = data.iloc[test_index]



def random_date(start, end, n):
    delta = end - start
    random_days = np.random.randint(0, delta.days, size=n)
    return start + pd.to_timedelta(random_days, unit='D')

start_train = datetime(2023, 5, 1)  
end_train = datetime(2023, 8, 31)   
train_data['Date'] = random_date(start_train, end_train, len(train_data))

start_test = datetime(2023, 9, 1) 
end_test = datetime(2023, 9, 30)    
validation_data['Date'] = random_date(start_test, end_test, len(validation_data))



train_data['Constant column'] = 100
validation_data['Constant column'] = 100

train_data['Random column'] = np.random.rand(len(train_data))
validation_data['Random column'] = np.random.rand(len(validation_data))

train_data.to_csv('data/train_data.csv', index=False)


if 'Education' in train_data.columns:
    random_indices = np.random.choice(train_data.index, size=100, replace=False)
    train_data.loc[random_indices, 'Education'] = np.nan
    train_data.to_csv('data/train_data.csv', index=False)
    print("100 null values added to the 'Education' column and dataset saved.")
else:
    print("'Education' column not found in the dataset.")


validation_data['CCAvg'] = validation_data['CCAvg'].astype(str).str.replace('/', '.')
validation_data['CCAvg'] = pd.to_numeric(validation_data['CCAvg'])

validation_data['Experience'] = validation_data['Experience'].apply(abs)

validation_data.drop(['ID', 'ZIP Code', 'Personal Loan'], axis=1, inplace=True)

#validation_data.drop(['Date'], axis=1, inplace=True)

validation_data['Education'] = validation_data.groupby('Income')['Education'].transform(
    lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x)
)


print('Train Data:')
print(train_data.head())


print('Validation Data:')
validation_data.to_csv('data/validation_data.csv', index=False)
print(validation_data.head())
