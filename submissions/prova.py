import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Load the actual dataset to get the number of rows (for correct number of predictions)
data = pd.read_csv('data/Bank_Personal_Loan_Modelling.csv')

# Number of rows in the dataset (same as the number of predictions needed)
num_rows = len(data)

# Generate random predictions for Group 1 and Group 2 (binary classification 0/1)
group_1_predictions = np.random.randint(0, 2, size=num_rows)
group_2_predictions = np.random.randint(0, 2, size=num_rows)

# Create DataFrame for Group 1
group_1_df = pd.DataFrame({
    'Prediction_Group_1': group_1_predictions
})

# Create DataFrame for Group 2
group_2_df = pd.DataFrame({
    'Prediction_Group_2': group_2_predictions
})

# Save the DataFrames as CSV files to simulate student submissions
group_1_df.to_csv('submissions/group_1_predictions.csv', index=False)
group_2_df.to_csv('submissions/group_2_predictions.csv', index=False)

print("Generated two random submission files: group_1_predictions.csv and group_2_predictions.csv")
