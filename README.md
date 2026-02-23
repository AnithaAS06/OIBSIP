
# OIBSIS

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
