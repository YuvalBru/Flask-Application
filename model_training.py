# קובץ פייתון - model_training.py:
# נשתמש בקוד לאימון המודל ונשמור אותו כקובץ PKL:
#  יש לציין כי כל ההסברים על אימון המודל כבר קיימים בקובץ הגשת המטלה 2
# •  שמירת המודל המאומן: נשמור את המודל המאומן לקובץ trained_model.pkl.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import joblib
from car_data_prep import prepare_data

df = pd.read_csv('dataset.csv')

df_prepared = prepare_data(df)

X = df_prepared.drop(columns=['Price', 'Cre_date'])
y = df_prepared['Price']

categorical_cols = X.select_dtypes(include=['category']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False))
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(max_iter=10000, tol=0.001))
])

param_grid = {
    'regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.9]
}

grid_search = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

grid_search.fit(X_train, y_train)

joblib.dump(grid_search.best_estimator_, 'trained_model.pkl')