import pandas as pd
import re
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import xgboost as xgb
import warnings

# Load the data from the CSV files
atom_pol_data = pd.read_csv('1_atom_exp_pol.csv')
diatomic_pol_data = pd.read_csv('diatomic_exp_pol.csv')

# Create a dictionary for quick lookup of atomic polarizabilities
atom_pol_dict = dict(zip(atom_pol_data['species'], atom_pol_data['alpha']))

# Function to parse species and extract atomic symbols and counts correctly
def parse_species(species):
    matches = re.findall(r'([A-Z][a-z]*)(\d*)', species)
    atoms = []
    for (element, count) in matches:
        count = int(count) if count else 1
        atoms.extend([element] * count)
    return atoms


# Apply the improved parsing function
diatomic_pol_data['atoms'] = diatomic_pol_data['species'].apply(parse_species)
diatomic_pol_data['atom1'] = diatomic_pol_data['atoms'].apply(lambda x: x[0])
diatomic_pol_data['atom2'] = diatomic_pol_data['atoms'].apply(lambda x: x[1] if len(x) > 1 else x[0])

# Retrieve polarizabilities for each atom in the diatomic molecules
def get_polarizability(atom1, atom2, atom_pol_dict):
    return atom_pol_dict.get(atom1, 0), atom_pol_dict.get(atom2, 0)

diatomic_pol_data['alpha1'], diatomic_pol_data['alpha2'] = zip(*diatomic_pol_data.apply(
    lambda row: get_polarizability(row['atom1'], row['atom2'], atom_pol_dict), axis=1))

# Drop the temporary 'atoms' column
diatomic_pol_data.drop(columns=['atoms'], inplace=True)

# Prepare the features and target variable
X = diatomic_pol_data[['alpha1', 'alpha2']]
y = diatomic_pol_data['alpha']

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Polynomial Features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Initialize the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror')

# Hyperparameter tuning with Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [6, 8, 10],
    'learning_rate': [0.01, 0.02, 0.05],
    'subsample': [0.7, 0.8, 0.9]
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Evaluate the model
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE: {rmse:.3f}")


first_36_elements = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr'
]

# Function to predict polarizability of a given molecule
def predict_polarizability(molecule, model, atom_pol_dict, poly):
    atoms = parse_species(molecule)
    if len(atoms) != 2:
        raise ValueError("The molecule should consist of exactly two atoms.")
    
    if not all(atom in first_36_elements for atom in atoms):
        raise ValueError("The molecule contains atoms not within the first 36 elements of the periodic table.")
    
    alpha1, alpha2 = get_polarizability(atoms[0], atoms[1], atom_pol_dict)
    features = np.array([[alpha1, alpha2]])
    features_poly = poly.transform(features)
    prediction = model.predict(features_poly)
    return round(prediction[0], 3)

# Example: Predict polarizability for H2 and Na2
molecules = ["NaN", "BeN", "HF"]

for molecule in molecules:
    try:
        predicted_alpha = predict_polarizability(molecule, best_model, atom_pol_dict, poly)
        print(f"Predicted polarizability for {molecule}: {predicted_alpha:.3f}")
    except ValueError as e:
        print(f"Error for {molecule}: {e}")