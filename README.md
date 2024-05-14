Project Summary: Human Activity Recognition using Smartphone Data

In this project, I developed a machine learning model to recognize human activities using smartphone data. The dataset contained sensor data from smartphones, including accelerometer and gyroscope readings, collected during various activities such as walking, standing, sitting, and others. The goal was to build a model that could accurately classify these activities based on the sensor data.

Key Steps:

Data Preprocessing: Imported necessary libraries including pandas, numpy, and scikit-learn for data manipulation, visualization, and modeling.

Exploratory Data Analysis (EDA): Conducted EDA to understand the structure and distribution of the data. This involved checking for duplicates, missing values, and visualizing the distribution of activities and features.

Feature Engineering: Extracted relevant features from the raw sensor data. Analyzed features like 'tBodyAccMag-mean()' and 'angle(X,gravityMean)' to differentiate between different activities.

Model Training: Trained multiple machine learning models including Logistic Regression, Kernel SVM, Decision Tree, and Random Forest. Utilized hyperparameter tuning and cross-validation techniques to optimize model performance.

Model Evaluation: Evaluated model performance using accuracy score and confusion matrix. Plotted confusion matrices to visualize the performance of the models in classifying different activities.

Conclusion: Achieved high accuracy in classifying human activities using smartphone data, with Logistic Regression achieving an accuracy of 99.9%. Identified best parameters for each model using RandomizedSearchCV, with DecisionTreeClassifier(max_depth=8) and RandomForestClassifier(n_estimators=80, max_depth=14) being the optimal choices.

This project demonstrates proficiency in data preprocessing, exploratory data analysis, feature engineering, and model training and evaluation techniques.
