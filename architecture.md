Here is a document detailing the current implementation of your Smart Grid ML project.

***

# Implementation Overview: Smart Grid Price Prediction

This document outlines the technical implementation of the electricity price prediction model, detailing its data sources, machine learning architecture, and prediction mechanism.

## 1. Data Sources and Feature Engineering

The model's predictive power relies on aggregating and processing data from multiple diverse sources. This process is managed by `tier2_pipeline.py` and consolidated during training in `train.py`.

**Primary Data Sources:**

* **CAISO Price Data:** Fetched from the CAISO OASIS API using the `CAISOPriceFetcher`. This provides the core **Locational Marginal Price (LMP)** data, which serves as the target variable (`price`) for the model.
* **NOAA Weather Data:** The `NOAAWeather` class fetches historical weather data (like temperature, wind speed, and precipitation) for various California locations.
* **Power Plant Data:** The `PowerPlantDB` class downloads a database of power plants, providing static features like total capacity, average capacity, and plant count across the state.
* **Disaster Risk Data:** The `DisasterRisk` class fetches recent earthquake data from the USGS to provide features like quake count and average/max magnitude.

**Feature Engineering (`train.py`):**

The `build_training_data` function merges these sources and performs crucial feature engineering:

1.  **Weather Pivoting:** Raw NOAA data (long format) is pivoted into a wide format, where each weather type (e.g., `TMAX`, `PRCP`) becomes its own feature column.
2.  **Data Merging:** The CAISO price data is merged with the pivoted weather data using `pd.merge_asof`, aligning price timestamps with the nearest available weather reports.
3.  **Cyclical Time Features:** To help the model understand time-based patterns, features like `hour`, `month`, and `day_of_week` are encoded into `sin` and `cos` components (e.g., `hour_sin`, `hour_cos`).
4.  **Static Features:** Aggregated power plant and earthquake data are added as static features (the same value for all rows) to provide grid-wide context.
5.  **Lag Features:** The `add_lag_features` function creates historical lag features (e.g., `price_lag_1`, `temperature_lag_3`) by shifting time-series columns. This is critical for allowing the model to see recent trends.

## 2. Machine Learning Model Architecture

The system uses a deep learning model, encapsulated in the `PricePredictor` class, and a standardized training process defined in `train.py`.

**Training Pipeline (`train_model` function):**

1.  **Data Preparation:** The full feature DataFrame is split into training and test sets using `train_test_split`. Critically, `shuffle=False` is used, which is essential for preserving the temporal order of time-series data.
2.  **Preprocessing:** A `StandardScaler` from `sklearn` is fit **only** on the training data (`X_train`). This scaler is then used to transform both the training and test sets, preventing data leakage.
3.  **Model Building:** A `PricePredictor` object is initialized, which builds a TensorFlow neural network model (`predictor.build_model()`). The use of TensorFlow, epochs, and batch sizes indicates a deep learning architecture (e.g., a Dense Neural Network or RNN/LSTM).
4.  **Training:** The model is trained using the `predictor.train()` method, passing in the scaled training data, `epochs`, and `batch_size`.
5.  **Evaluation:** The trained model is evaluated against the held-out test set (`X_test`, `y_test`), and metrics like Mean Squared Error (Loss), Mean Absolute Error (MAE), and Mean Absolute Percentage (MAPE) are logged.
6.  **Persistence:** The trained model is saved as `price_model.h5`, the `StandardScaler` is saved as `scaler.pkl`, and the list of feature names is saved as `features.json`. This "package" contains everything needed for prediction.

## 3. Prediction Pipeline (Web Service)

The `app.py` Flask service exposes the trained model via a `/predict` API endpoint.

**Prediction Flow:**

1.  **Initialization:** When the Flask app starts, the `initialize_app` function loads the `price_model.h5`, `scaler.pkl`, and `features.json` files from disk, creating a single, shared `predictor` object in memory.
2.  **API Request:** A user sends a GET request to `/predict`, providing a `location_id` (e.g., `los_angeles`).
3.  **Live Feature Building:** The `build_features_for_location` function is called to construct a single feature vector for the *current* moment:
    * **Time:** Gets current time features (`hour`, `month`, `is_weekend`).
    * **Weather:** Fetches *live* weather data from the `NOAAWeather` service for the specified location.
    * **Demand/Generation:** Generates *synthetic* demand and generation features (`total_demand`, `solar_mw`) based on time of day, weather, and location-specific `DEMAND_PROFILES`.
    * **Lag Features:** This is the most complex step. The `get_recent_caiso_prices` function is called to fetch *actual* historical prices from the CAISO API for the last few hours/days. This data is used to populate the lag features (e.g., `price_lag_1`, `price_lag_3`) that the model was trained on.
4.  **Prediction:** The complete feature dictionary is passed to `predictor.predict_for_location`. This method (within the `PricePredictor` class) uses the loaded `features.json` to order the data correctly, scales it using the loaded `scaler.pkl`, and feeds the resulting array into the `price_model.h5` for a prediction.
5.  **Response:** The raw model output (a single price) is clipped to a reasonable range (`np.clip(price, 10, 200)`), classified into a "level" (e.g., LOW, HIGH), and returned to the user in a JSON response.