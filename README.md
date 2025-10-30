# Kaggle Competition: Road Accident Risk Prediction

This project is a submission for the [Playground Series - Season 5, Episode 10](https://www.kaggle.com/competitions/playground-series-s5e10) on Kaggle.

## Project Goal
The goal was to predict the `accident_risk` (a continuous, probabilistic value between 0 and 1) based on various features of a road segment, such as road type, lighting, weather, and speed limit.

## Workflow

1.  **Exploratory Data Analysis (EDA):** Loaded the data using `pandas` and analyzed the features. I visualized and gathered descriptions of the target variable (`accident_risk`) and features to see their relationship with the target by using `seaborn` and `matplotlib`.

2.  **Feature Engineering:** To capture more complex relationships in the data, I created new interaction and polynomial features. This included:
    * `accidents_per_lane`: A ratio to normalize accident counts by road size.
    * `weather_and_time`: A categorical interaction feature (e..g, 'Rain\_Night').
    * `square_speed`: A polynomial feature to capture the non-linear risk of higher speeds.

3.  **Data Processing:**
    * Converted boolean (True/False) columns into integers (1/0).
    * Used sklearn's ColumnTransformer to precisely manage preprocessing.
    * Categorical features (`road_type`, `weather`, etc.) were One Hot Encoded.
    * Numerical and boolean features were passed through to the model.

4.  **Modelling & Validation:**
    * **Model:** I used the XGBoost Regressor, a powerful gradient-boosting model.
    * **Validation:** Instead of a single train-test split, I implemented a robust 10 Fold Cross Validation strategy. This involves training 10 separate models on different 90% "folds" of the data. This provides a much more stable and reliable measure of the model's true performance and prevents overfitting to a single "lucky" validation set.

5.  **Hyperparameter Tuning:**
    * To find the best possible settings for the XGBoost model, I used Optuna, a Bayesian optimization framework.
    * Optuna automatically tested dozens of hyperparameter combinations (like `learning_rate`, `max_depth`, `subsample`, etc.) over several hours.
    * For each combination, it ran our entire 10-fold CV to get a true, reliable score, using the GPU (`tree_method='hist', device='cuda'`) to accelerate the process.

## Evaluation
The model was evaluated using the competition metric: Root Mean Squared Error (RMSE). A lower RMSE is better.

## Results
* **Baseline Model:** Our robust 10-fold CV on the un-tuned model with new features achieved an RMSE of **0.05619**.
* **Final Tuned Model:** After the Optuna study, the final optimized model achieved an RMSE of **0.05603**, with a test result of **0.05555**.