import pandas as pd

data = pd.read_csv('data/Bank_Personal_Loan_Modelling.csv')

features = data.drop(columns='Personal Loan')  
target = data['Personal Loan']  

features.to_csv('data/loan_data.csv', index=False)      