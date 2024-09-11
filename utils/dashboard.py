import os
import pandas as pd
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
from sklearn.metrics import accuracy_score

# Initialize the Dash app
app = Dash(__name__)

# Directory where student submissions are stored
submission_dir = 'submissions/'  # Make sure this directory contains the students' CSV files

# Load the actual dataset with the true target
data = pd.read_csv('data/Bank_Personal_Loan_Modelling.csv')
true_target = data['Personal Loan']

# Function to process predictions and calculate accuracy
def process_predictions():
    accuracies = {}
    predictions_df = pd.DataFrame()
    predictions_df['True_Target'] = true_target
    
    # Iterate over each student submission file
    for filename in os.listdir(submission_dir):
        if filename.endswith('.csv'):
            # Extract group number from the filename
            group_number = filename.split('_')[1]  # Assuming format: group_X_predictions.csv

            # Load the student predictions
            student_predictions = pd.read_csv(os.path.join(submission_dir, filename))

            # Extract the column with predictions and rename it
            column_name = f'Prediction_Group_{group_number}'
            predictions_df[column_name] = student_predictions.iloc[:, 0]  # Add predictions to DataFrame

            # Calculate accuracy for this group
            accuracy = accuracy_score(true_target, student_predictions.iloc[:, 0])

            # Store the accuracy in the dictionary
            accuracies[column_name] = accuracy

    # Convert accuracy results to a DataFrame for ranking
    accuracy_df = pd.DataFrame({
        'Group': list(accuracies.keys()),
        'Accuracy': list(accuracies.values())
    })

    # Rank by accuracy in descending order
    accuracy_df = accuracy_df.sort_values(by='Accuracy', ascending=False).reset_index(drop=True)

    return accuracy_df

# Create the predictions dataframe and accuracy metrics
accuracy_df = process_predictions()

# Dashboard Layout
app.layout = html.Div([
    html.H1("Student Group Rankings - Accuracy"),
    
    # Display the ranked table of group accuracies
    html.H3("Classifica of Groups Based on Accuracy"),
    
    html.Table([
        html.Thead(
            html.Tr([html.Th("Rank"), html.Th("Group"), html.Th("Accuracy")])
        ),
        html.Tbody([
            html.Tr([html.Td(i+1), html.Td(accuracy_df['Group'].iloc[i]), html.Td(f"{accuracy_df['Accuracy'].iloc[i]:.4f}")])
            for i in range(len(accuracy_df))
        ])
    ])
])

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
