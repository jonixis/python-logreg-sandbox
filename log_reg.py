"""Logistic regression"""

import pandas as pd
from sklearn.linear_model import LogisticRegression

# Load csv into DataFrame
df = pd.read_csv('datasets/admission_predict.csv')

# Cleaning
df.rename(columns={'Chance of Admit ': 'Chance of Admit', 'LOR ': 'LOR'}, inplace=True)
df.drop(labels='Serial No.', axis=1, inplace=True)

# Prepare for regression
targets = df['Chance of Admit']
targets_bin = (targets > 0.5).astype(int)
features = df.drop(columns={'Chance of Admit'})

# Get coefficients with gradient descent
logreg = LogisticRegression(tol=0.000001, max_iter=10, solver='liblinear')
logreg.fit(features, targets_bin)

print('Coefficients: ')
print(*logreg.coef_[0], sep='\n', end='\n\n')

print('Number of iterations: ')
print(*logreg.n_iter_, end='\n\n')
