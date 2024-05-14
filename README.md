Project Summary: Human Activity Recognition using Smartphone Data

In this project, I developed a machine learning model to recognize human activities using smartphone data. The dataset contained sensor data from smartphones, including accelerometer and gyroscope readings, collected during various activities such as walking, standing, sitting, and others. The goal was to build a model that could accurately classify these activities based on the sensor data.

Key Steps:

Data Collection and Inspection: Acquired a dataset containing sensor readings from smartphones during various human activities. Inspected the dataset for any inconsistencies or anomalies.

Data Preprocessing: Cleaned the dataset by handling duplicates and missing values to ensure the quality of the data for analysis and modeling.

Exploratory Data Analysis (EDA): Conducted EDA to understand the distribution of sensor data and the patterns of different human activities. Visualized the data to identify any trends or relationships between features and activities.

Feature Engineering: Extracted relevant features from the raw sensor data to improve the predictive performance of the model. Engineered features such as magnitude of acceleration and angles between axes and gravity to capture important characteristics of human activities.

Dimensionality Reduction: Utilized dimensionality reduction techniques such as Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) to visualize high-dimensional data in lower-dimensional space. This helped in understanding the structure of the data and identifying potential clusters or patterns.

Model Training: Trained multiple machine learning models including Logistic Regression, Support Vector Machines (SVM), Decision Trees, and Random Forests. Each model was trained on the preprocessed dataset to learn the relationship between features and activity labels.

Hyperparameter Tuning: Optimized the hyperparameters of the models using techniques like RandomizedSearchCV to improve their performance and generalization capabilities.

Model Evaluation: Evaluated the performance of each model using metrics such as accuracy score and confusion matrix. This helped in assessing how well the models were able to classify different human activities.

Conclusion: Achieved high accuracy in classifying human activities, demonstrating proficiency in data preprocessing, exploratory data analysis, feature engineering, dimensionality reduction, model selection, hyperparameter tuning, and model evaluation, with Logistic Regression achieving an accuracy of 99.9%. Identified best parameters for each model using RandomizedSearchCV, with DecisionTreeClassifier(max_depth=8) and RandomForestClassifier(n_estimators=80, max_depth=14) being the optimal choices

This project demonstrates proficiency in data preprocessing, exploratory data analysis, feature engineering, and model training and evaluation techniques.
