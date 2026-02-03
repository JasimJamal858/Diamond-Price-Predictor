# Diamond-Price-Predictor
Loads and previews the diamond dataset (carat, depth, table, x, y, z, categoricals). Users select features via multiselect (numeric defaults). Trains scikit-learn LinearRegression on-the-fly (70/30 split, one-hot encoding), shows MAPE/R² metrics and sample predictions. Manual prediction uses dynamic inputs based on data types/ranges.
