# Kaggle Competition: Road Accident Risk Prediction

This project is a submission for the [Playground Series - Season 5, Episode 10](https://www.kaggle.com/competitions/playground-series-s5e10) on Kaggle.

## Project Goal
The goal was to predict the `accident_risk` (a continuous value) based on various features of a road segment, such as road type, lighting, weather, and speed limit.

## My Approach
1.  **Exploratory Data Analysis (EDA):** Loaded the data and analyzed the features. I checked the distribution of the target variable (`accident_risk`) and visualized the categorical features to see their relationship with the target.
2.  **Data Processing:**
    * Converted boolean (True/False) columns into integers (1/0).
    * Used `sklearn`'s `ColumnTransformer` to One-Hot Encode the categorical features (`road_type`, `lighting`, etc.).
    * Split the training data into a training set and a validation set (80/20 split).
3.  **Modelling:**
    * Used an **XGBoost Regressor** (`XGBRegressor`) model.
    * Trained the model on the processed training data, using the validation set for early stopping to prevent overfitting.
4.  **Evaluation:**
    * The model was evaluated using the Root Mean Squared Error (RMSE) metric.

## Results
The final model achieved an RMSE of **0.05622** on the local validation set.