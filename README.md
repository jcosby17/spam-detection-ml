# spam-detection-ml

Spam Detection Using Machine Learning

This project builds a machine learning system that automatically classifies SMS messages as spam or ham. The goal is to detect unwanted or harmful messages using simple AI techniques.

I use the SMS Spam Collection dataset and apply text preprocessing such as lowercasing, removing stopwords, and converting messages into TF-IDF features. I train and compare two models: Multinomial Naive Bayes and Logistic Regression.

This project was completed for CAP 4630 – Intro to Artificial Intelligence at Florida Atlantic University.

**Table of Contents**

Project Overview
Dataset
Technologies Used
Project Pipeline
How to Run
Results
Conclusion

**Project Overview**

Spam messages are very common and can be risky due to phishing attempts and scam links. This project shows how machine learning can be used to automatically identify spam by analyzing the text of each message.

I compare two machine learning models:
Multinomial Naive Bayes
Logistic Regression

Both models are trained on TF-IDF features and evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

**Dataset**

I use the SMS Spam Collection Dataset, which contains over 5,000 SMS text messages labeled as ham (normal) or spam (unwanted or promotional).
If using the Kaggle version, the file is usually named spam.csv.

**Technologies Used**

Python
Google Colab
pandas
scikit-learn
matplotlib
seaborn

**Project Pipeline**

Load the dataset
Clean the text (lowercasing and removing stopwords)
Split the data into training and testing sets
Convert the text using TF-IDF
Train the Naive Bayes model
Train the Logistic Regression model
Evaluate accuracy, precision, recall, F1-score, and confusion matrices
Compare the two models

**How to Run**

Open the Colab notebook.
Upload the spam.csv dataset using:

from google.colab import files
uploaded = files.upload()

Install dependencies if needed:

!pip install pandas scikit-learn matplotlib seaborn

Run all cells.
The notebook will preprocess the messages, train the models, and display all results.

**Results**

Replace these sample values with your real results:

Model: Naive Bayes
Accuracy: 0.97
Precision (Spam): 0.90
Recall (Spam): 0.86
F1 (Spam): 0.88

Model: Logistic Regression
Accuracy: 0.98
Precision (Spam): 0.95
Recall (Spam): 0.91
F1 (Spam): 0.93


**Conclusion**

Both models performed well on the SMS spam classification task. Logistic Regression achieved slightly higher accuracy and F1-score, while Naive Bayes was very fast and still effective.

This project shows that classic machine learning combined with TF-IDF features can detect spam messages with high accuracy. Future improvements could include using deep learning, trying n-gram features, or testing more datasets.
