
# OIBSIP

# PROJECT 01(LEVEL 01)

## Supermart Grocery Sales – Retail Analytics (EDA Project)
📌 Project Title

Exploratory Data Analysis (EDA) on Supermart Grocery Sales Dataset

📖 Project Description

This project performs Exploratory Data Analysis (EDA) on the Supermart Grocery Sales - Retail Analytics Dataset.

The objective of this analysis is to understand sales performance, customer behavior, product trends, regional distribution, and profitability patterns to generate meaningful business insights.

The analysis was implemented using Python in Jupyter Notebook.

🎯 Objectives

Analyze overall sales and profit performance

Identify top-performing categories and sub-categories

Study region-wise and state-wise sales trends

Analyze customer contribution to revenue

Examine the relationship between discount and profit

Perform time-based sales trend analysis

Provide business recommendations based on insights

🛠️ Tools & Technologies Used

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

📂 Dataset Information

Dataset Name:
Supermart Grocery Sales - Retail Analytics Dataset.csv

📊 Dataset Columns:

Order ID

Customer Name

Category

Sub Category

City

Order Date

Region

Sales

Discount

Profit

State

🔎 Steps Performed in the Analysis
1️⃣ Data Loading

The dataset was loaded using pandas and initial inspection was performed using:

head()

info()

describe()

2️⃣ Data Cleaning

Removed duplicate records

Checked and handled missing values

Converted Order Date to datetime format

Ensured Sales and Profit were numeric

3️⃣ Descriptive Analysis

Total Sales calculation

Total Profit calculation

Average Discount calculation

Statistical summary of numeric features

4️⃣ Regional Analysis

Region-wise total sales

State-wise top-performing states

Visualization using bar charts

5️⃣ Category & Sub-Category Analysis

Sales by Category

Sales by Sub Category

Profit comparison between categories

6️⃣ Customer Analysis

Identified top 10 customers based on sales

Analyzed revenue contribution from customers

7️⃣ Time Series Analysis

Extracted Month and Year from Order Date

Analyzed monthly sales trends

Visualized sales growth patterns

8️⃣ Correlation Analysis

Studied relationship between:

Sales

Profit

Discount

Observed impact of discount on profitability

📈 Key Insights

Certain regions contribute significantly higher sales.

A few categories generate the majority of revenue.

Higher discounts tend to reduce profit margins.

Sales show variation across months.

Top customers contribute a large portion of revenue.

💡 Business Recommendations

Focus marketing efforts on high-performing regions.

Optimize discount strategies to protect profit margins.

Increase inventory for high-demand categories.

Develop loyalty programs for top customers.

Improve performance in low-performing states.

✅ Conclusion

This Exploratory Data Analysis helped uncover important sales patterns, customer behavior insights, and profitability trends within the Supermart Grocery dataset.

The findings can support data-driven decision-making to improve revenue growth and operational efficiency.

👩‍💻 Author

Internship Project – Data Analysis
Anitha A S

# PROJECT 01(LEVEL 02)

# House Price Prediction using Linear Regression
📌 Project Title

Predicting House Prices using Linear Regression

📖 Project Description

This project focuses on building a machine learning model to predict house prices using Linear Regression.

The objective is to estimate house prices based on relevant features such as area, number of bedrooms, bathrooms, and other property-related attributes.

The model is implemented using Python in Jupyter Notebook and evaluated using standard regression metrics.

🎯 Objectives

Perform data exploration and cleaning

Select relevant features affecting house prices

Split the dataset into training and testing sets

Train a Linear Regression model

Evaluate model performance using MSE and R² score

Visualize predicted vs actual values

Interpret model coefficients

🛠️ Tools & Technologies Used

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

Scikit-Learn

📂 Dataset Information

The dataset contains housing-related features and a target variable:

Example Features:

Area

Bedrooms

Bathrooms

Floors

Location (if available)

Price (Target Variable)

The target variable for prediction is:

➡ House Price

🔎 Steps Performed
1️⃣ Data Collection

The housing dataset was loaded into Jupyter Notebook using Pandas.

2️⃣ Data Exploration

Checked dataset shape and structure

Viewed summary statistics

Identified data types

Checked for missing values

3️⃣ Data Cleaning

Handled missing values

Ensured numeric format for features

Removed unnecessary columns

4️⃣ Feature Selection

Selected independent variables (X) such as:

Area

Bedrooms

Bathrooms

Target variable (y):

Price

5️⃣ Train-Test Split

The dataset was divided into:

80% Training data

20% Testing data

This ensures unbiased model evaluation.

6️⃣ Model Training

A Linear Regression model from Scikit-Learn was trained using the training dataset.

7️⃣ Model Evaluation

The model was evaluated using:

📌 Mean Squared Error (MSE)

📌 R² Score

These metrics help measure prediction accuracy.

8️⃣ Visualization

A scatter plot was created to compare:

Actual Prices

Predicted Prices

This helps visually assess model performance.

9️⃣ Interpretation

Model coefficients were analyzed to understand:

Which features impact price the most

Whether features increase or decrease house price

📈 Key Insights

House area has a strong positive relationship with price.

Additional bedrooms and bathrooms generally increase price.

The model demonstrates reasonable prediction capability based on R² score.

Linear Regression effectively models the relationship between features and house price.

💡 Conclusion

The Linear Regression model successfully predicts house prices using selected features.

This project demonstrates the practical implementation of:

Data preprocessing

Feature selection

Regression modeling

Performance evaluation

Model interpretation

It provides hands-on experience in applying machine learning techniques to real-world datasets.

👩‍💻 Author

Internship Project – Data Analysis / Machine Learning
Anitha Ani

# PROJECT 02(LEVEL_01)
# Marketing Analytics – Customer Segmentation
📌 Project Title

Customer Segmentation using Data Analytics Techniques

📖 Project Description

This project focuses on analyzing customer data to perform customer segmentation for marketing analytics.

Customer segmentation helps businesses divide customers into meaningful groups based on purchasing behavior, income, spending patterns, or demographic characteristics.

The objective is to identify distinct customer groups that can help businesses design targeted marketing strategies.

The project is implemented using Python in Jupyter Notebook.

🎯 Objectives

Perform exploratory data analysis (EDA) on customer data

Clean and preprocess the dataset

Analyze customer behavior patterns

Segment customers into meaningful groups

Visualize segmentation results

Provide actionable marketing recommendations

🛠️ Tools & Technologies Used

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

Scikit-Learn (for clustering if used)

📂 Dataset Overview

The dataset contains customer-related information such as:

Customer ID

Age

Gender

Annual Income

Spending Score

Purchase behavior metrics

(Exact features depend on the dataset used in the notebook.)

🔎 Steps Performed
1️⃣ Data Loading

Imported the dataset using Pandas

Displayed first few rows

Checked dataset dimensions

2️⃣ Data Exploration

Checked data types

Identified missing values

Generated statistical summaries

Observed distribution of key features

3️⃣ Data Cleaning

Handled missing values

Removed duplicates

Converted data types if required

4️⃣ Exploratory Data Analysis (EDA)

Analyzed income distribution

Studied spending score patterns

Compared customer behavior across demographics

Identified trends and correlations

5️⃣ Customer Segmentation

Customer segmentation was performed using analytical techniques such as:

Grouping based on income and spending

Clustering (e.g., K-Means if implemented)

This helps identify customer types like:

High income – High spending

High income – Low spending

Low income – High spending

Low income – Low spending

6️⃣ Visualization

Various visualizations were created to understand patterns:

Scatter plots

Bar charts

Cluster visualizations

Distribution plots

These visuals help interpret segmentation clearly.

📈 Key Insights

Customers can be divided into distinct spending groups.

High-income customers do not always spend more.

Certain customer groups offer high revenue potential.

Targeted marketing can improve business growth.

💡 Business Recommendations

Focus premium marketing on high-income, high-spending customers.

Offer discounts to high-income but low-spending customers.

Introduce loyalty programs for frequent buyers.

Personalize promotions based on customer segment.

✅ Conclusion

Customer segmentation enables businesses to:

Understand customer behavior

Improve marketing efficiency

Increase customer retention

Maximize revenue

This project demonstrates how data analytics can be applied to real-world marketing strategies.

👩‍💻 Author

Internship Project – Marketing Analytics
Anitha Ani

# PROJECT 02(LEVEL_02)
# Wine Quality Prediction using Machine Learning
📌 Project Title

Wine Quality Prediction using Random Forest, SGD, and Support Vector Classifier

📖 Project Description

This project aims to predict the quality of wine based on its physicochemical properties using Machine Learning classification algorithms.

The dataset used is WineQT.csv, which contains chemical attributes such as acidity, alcohol content, density, sulphates, and more.

The goal is to:

Perform data preprocessing

Conduct exploratory data analysis (EDA)

Train multiple classification models

Compare their performance

Identify the best-performing model

📂 Dataset Details

Dataset Name: WineQT.csv

Type: Classification Dataset

Target Variable: quality

🔬 Features Used

Fixed Acidity

Volatile Acidity

Citric Acid

Residual Sugar

Chlorides

Free Sulfur Dioxide

Total Sulfur Dioxide

Density

pH

Sulphates

Alcohol

Quality (Output Label)

🛠️ Tools & Technologies

Python

Jupyter Notebook

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

⚙️ Project Workflow
1️⃣ Data Loading

Import dataset using Pandas

Display dataset structure and summary

2️⃣ Data Preprocessing

Check missing values

Remove unnecessary columns (if any)

Separate features (X) and target (y)

3️⃣ Exploratory Data Analysis (EDA)

Correlation heatmap

Quality distribution analysis

Feature relationship visualization

4️⃣ Data Splitting

Train-test split (80% training, 20% testing)

5️⃣ Feature Scaling

StandardScaler used to normalize feature values

6️⃣ Model Implementation

Three classification models were implemented:

🌲 Random Forest Classifier

⚡ Stochastic Gradient Descent (SGD) Classifier

🎯 Support Vector Classifier (SVC)

7️⃣ Model Evaluation

Accuracy Score

Confusion Matrix

Classification Report

8️⃣ Model Comparison

Compare model accuracies

Select best performing model

📊 Expected Output

Accuracy comparison of all three models

Confusion matrix visualization

Classification performance metrics

🚀 How to Run the Notebook

Download the repository

Place WineQT.csv in the same folder as Wine_Quality.ipynb

Open Jupyter Notebook

Run all cells sequentially

🎯 Learning Outcomes

Understanding classification problems

Performing EDA effectively

Applying feature scaling

Implementing multiple ML models

Comparing and evaluating model performance

📌 Conclusion

This project demonstrates how machine learning algorithms can be used to predict wine quality based on chemical attributes. It emphasizes the importance of preprocessing, visualization, and model evaluation in building reliable predictive systems.

Author
internship project - Wine Quality Analysis
Anitha Ani

# PROJECT 03(LEVEL_01)
# Data Cleaning Project
📌 Project Title

Data Cleaning and Preprocessing for Data Analysis

🎯 Objective

The objective of this project is to clean and preprocess raw data to improve its quality, consistency, and reliability before performing data analysis or machine learning tasks.

Data cleaning ensures that the dataset is accurate, complete, and ready for further analysis.

📖 Project Description

Raw datasets often contain:

Missing values

Duplicate records

Incorrect data formats

Outliers

Inconsistent entries

This project focuses on identifying and fixing these issues using Python-based data analysis tools.

🛠 Tools & Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

🔎 Data Cleaning Steps Performed
1️⃣ Data Loading

Import dataset using Pandas

Display first few rows

Understand dataset structure

2️⃣ Data Exploration

Check data types

View summary statistics

Identify null values

3️⃣ Handling Missing Values

Detect missing data

Remove or fill missing values using:

Mean

Median

Mode

Forward/Backward fill

4️⃣ Removing Duplicates

Identify duplicate rows

Drop duplicate records

5️⃣ Data Type Conversion

Convert columns into appropriate formats

String to numeric

Object to datetime

Integer to float

6️⃣ Handling Outliers

Detect outliers using:

Boxplots

Z-score method

IQR method

Remove or cap extreme values

7️⃣ Feature Scaling (if required)

Normalize or standardize numeric features

📊 Outcome

After cleaning:

Dataset becomes consistent and structured

Missing values are handled

Duplicates are removed

Data types are corrected

Dataset is ready for analysis or machine learning

🚀 Importance of Data Cleaning

Data cleaning improves:

Accuracy of analysis

Model performance

Decision-making quality

Overall reliability of results

📌 Conclusion

Data cleaning is a crucial first step in any data analytics or machine learning project. Proper preprocessing ensures better insights and more accurate predictive models.

# PROJECT 03(LEVEL_02)
# Credit Card Fraud Detection
📌 Project Overview

This project focuses on detecting fraudulent credit card transactions using data analytics and machine learning techniques. The goal is to build a predictive model that can accurately identify fraud cases while minimizing false alarms.

Fraud detection is a critical application in financial security systems, helping prevent financial loss and protect customers.

🎯 Objective

The objective of this project is to:

Analyze credit card transaction data

Identify patterns that distinguish fraudulent transactions from legitimate ones

Handle class imbalance in the dataset

Build and evaluate machine learning models for fraud detection

Improve detection performance using appropriate techniques

📂 Dataset Information

Dataset: creditcard.csv

Total Transactions: 284,807

Features:

Time – Transaction time

V1 to V28 – PCA-transformed features

Amount – Transaction amount

Class – Target variable

0 → Normal Transaction

1 → Fraud Transaction

📌 The dataset is highly imbalanced (very few fraud cases).

🛠 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

🔎 Project Workflow
1️⃣ Data Loading

Import dataset using Pandas

Explore dataset structure

2️⃣ Data Cleaning

Check missing values

Remove duplicate records

Convert data types if required

3️⃣ Exploratory Data Analysis (EDA)

Analyze class distribution

Visualize fraud vs normal transactions

Correlation analysis

4️⃣ Data Preprocessing

Feature scaling (StandardScaler)

Drop unnecessary columns (if required)

Split data into training and testing sets

5️⃣ Model Building

Logistic Regression

Random Forest (optional improvement)

6️⃣ Model Evaluation

Accuracy

Precision

Recall

F1-score

Confusion Matrix

📌 In fraud detection, Recall and F1-score are more important than accuracy due to class imbalance.

📊 Results

Successfully identified fraud patterns

Improved detection performance using class balancing

Achieved strong recall for fraud class

🚀 Future Improvements

Apply SMOTE for better handling of imbalanced data

Use advanced models like XGBoost

Deploy model using Flask or Streamlit

Real-time fraud detection integration

📈 Key Learnings

Handling imbalanced datasets

Importance of feature scaling

Evaluating classification models using proper metrics

Understanding financial fraud detection systems

👩‍💻 Author

Anitha Ani
Data Analytics & Machine Learning Project

# PROJECT 04(LEVEL_01)
# Sentiment Analysis using Machine Learning
📌 Project Overview

This project focuses on performing Sentiment Analysis on textual data to determine whether a given text expresses a positive, negative, or neutral sentiment.

Sentiment analysis is widely used in:

Product reviews

Social media monitoring

Customer feedback analysis

Market research

The project uses Natural Language Processing (NLP) techniques and Machine Learning models to classify sentiments accurately.

🎯 Objective

The main objectives of this project are:

Analyze text data and extract meaningful insights

Preprocess textual data using NLP techniques

Convert text into numerical features

Build a classification model for sentiment prediction

Evaluate model performance using appropriate metrics

📂 Dataset Information

Dataset: Text-based dataset (e.g., reviews, comments, or tweets)

Key Columns:

Text → Input text data

Sentiment → Target label (Positive / Negative / Neutral)

🛠 Technologies Used

Python

Pandas

NumPy

NLTK / Scikit-learn

Matplotlib

Seaborn

🔎 Project Workflow
1️⃣ Data Loading

Import dataset using Pandas

Explore dataset structure

2️⃣ Data Cleaning

Remove null values

Remove special characters

Convert text to lowercase

Remove stopwords

Tokenization

3️⃣ Text Preprocessing

Stemming or Lemmatization

Removing punctuation

Cleaning unnecessary whitespace

4️⃣ Feature Extraction

Count Vectorizer

TF-IDF Vectorizer

5️⃣ Model Building

Logistic Regression

Naive Bayes

Support Vector Machine (optional)

6️⃣ Model Evaluation

Accuracy

Precision

Recall

F1-score

Confusion Matrix

📊 Results

Successfully classified text into sentiment categories

Achieved good accuracy using TF-IDF features

Improved model performance after preprocessing

🚀 Future Improvements

Use Deep Learning models (LSTM, BERT)

Deploy as a web app using Streamlit or Flask

Perform real-time sentiment analysis

Multi-language sentiment support

📈 Applications

Customer review analysis

Brand reputation monitoring

Political sentiment tracking

Feedback analysis for businesses

👩‍💻 Author
Anitha Ani

# PROJECT 04(LEVEL_02)
# Unveiling the Android App Market
📊 Analyzing Google Play Store Data
📌 Project Overview

This project focuses on analyzing the Google Play Store dataset to uncover insights about the Android app market. The analysis helps understand trends in app categories, user ratings, installs, pricing strategies, and user reviews.

By performing exploratory data analysis (EDA) and data cleaning, this project reveals patterns that influence app popularity and market performance.

🎯 Objective

The main objectives of this project are:

Analyze app market trends across different categories

Identify factors affecting app ratings and installs

Compare free vs paid app performance

Study user sentiment from reviews

Discover revenue and pricing patterns

📂 Dataset Information

The project uses two datasets:

1️⃣ app.csv

App – Application name

Category – App category

Rating – User rating

Reviews – Number of reviews

Size – App size

Installs – Number of downloads

Type – Free or Paid

Price – App price

Content Rating – Age group suitability

Genres – App genre

2️⃣ user_reviews.csv

App – Application name

Translated_Review – User review text

Sentiment – Positive / Negative / Neutral

Sentiment_Polarity

Sentiment_Subjectivity

🛠 Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn (optional for modeling)

🔎 Project Workflow
1️⃣ Data Loading

Import datasets using Pandas

Display structure and summary

2️⃣ Data Cleaning

Handle missing values

Remove duplicate entries

Clean "Installs" column (remove + and ,)

Convert "Price" to numeric

Clean "Size" column (convert MB/KB to numeric)

3️⃣ Exploratory Data Analysis (EDA)

Distribution of app ratings

Most popular app categories

Free vs Paid app comparison

Install trends by category

Correlation between reviews and ratings

4️⃣ User Sentiment Analysis (Optional)

Merge review dataset with app dataset

Analyze sentiment distribution

Compare rating vs sentiment

5️⃣ Visualization

Bar charts for top categories

Histograms for ratings and installs

Boxplots for price vs rating

Heatmap for correlations

📊 Key Insights

Most apps on Play Store are Free

Certain categories dominate installs (e.g., Games, Tools)

Higher reviews often correlate with higher installs

Paid apps generally have higher ratings in niche categories

🚀 Future Improvements

Build a prediction model for app success

Perform advanced sentiment analysis using NLP

Revenue estimation based on installs & pricing

Dashboard creation using Power BI / Tableau

Deploy as a web analytics app

📈 Business Applications

Helps developers understand market demand

Assists investors in identifying profitable categories

Supports marketing strategy planning

Improves app pricing decisions

👩‍💻 Author

Anitha Ani
Data Analytics & Market Research Project

# PROJECT 05
# Autocomplete & Autocorrect using Data Analytics
📌 Project Overview

This project focuses on building an Autocomplete and Autocorrect system using data analytics and Natural Language Processing (NLP) techniques.

Autocomplete predicts the next word based on previous input, while Autocorrect detects and corrects spelling mistakes. These systems are widely used in search engines, messaging apps, and text editors.

The project demonstrates how text data can be analyzed and transformed into intelligent predictive models.

🎯 Objective

The main objectives of this project are:

Analyze text data to understand word frequency patterns

Build an Autocomplete system using probabilistic models

Implement Autocorrect using edit distance algorithms

Improve typing efficiency and user experience

Evaluate prediction accuracy

📂 Dataset Information

Text corpus dataset (e.g., articles, reviews, chat data)

Preprocessed text used to build vocabulary and frequency dictionary

Dataset includes:

Words and their frequencies

Sentences for language modeling

Common spelling variations

🛠 Technologies Used

Python

Pandas

NumPy

NLTK

Scikit-learn (optional)

Regular Expressions

🔎 Project Workflow
1️⃣ Data Collection

Load text dataset

Combine and structure raw text data

2️⃣ Data Cleaning

Convert text to lowercase

Remove punctuation and special characters

Tokenization

Remove unwanted symbols

3️⃣ Vocabulary Building

Create word frequency dictionary

Identify most common words

Generate probability distribution

4️⃣ Autocomplete Implementation

Use N-gram model (Unigram, Bigram, Trigram)

Predict next word based on previous words

Rank suggestions by probability

5️⃣ Autocorrect Implementation

Implement Edit Distance (Levenshtein Distance)

Generate candidate corrections

Choose most probable correct word

Compare input word with vocabulary

6️⃣ Model Evaluation

Accuracy of word prediction

Correction success rate

Response time performance

📊 Key Features

Real-time word prediction

Spelling error detection

Probability-based suggestions

Customizable vocabulary

🚀 Future Improvements

Implement Deep Learning models (LSTM, Transformer)

Integrate with mobile or web application

Add multilingual support

Improve context understanding

📈 Applications

Search engines

Chat applications

Email typing assistants

Code editors

Smart keyboards

👩‍💻 Author

Anitha Ani
Data Analytics & NLP Project
