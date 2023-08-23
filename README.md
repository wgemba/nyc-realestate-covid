# Regression Modelling of New York City Residential Property Values in Relation to the COVID-19 Pandemic
## I. Abstract
This project will study the various applications of the machine learning and regression algorithms on New York City real estate data to accurately predict property valuations. Various algorithms will be employed, including Linear Regression, Support Vector Regression (SVR), Random Forest Regression (RFR), and Gradient-Boosting Regression (GBR). The objectives of this paper will be to (a) determine which predictor variables from the dataset play the greatest role in determining property value, (b) identify the prediction algorithm that optimizes prediction performance, and (c) identify the marginal effect that COVID-19 cases have on predicting the value of the same set of real estate properties. To measure algorithm performance, I will employ the use of metrics including mean squared error (MSE) and root mean squared error (RMSE). The main goal of this project is to identify an optimized prediction algorithm that will be used as the foundation for an automated property appraisal system.

## II. Introduction
The real estate market is one of the largest investor markets in both the U.S. and global economy. When it comes to commercial real estate, institutional investors have, to some extent, perfected the property valuation process with the use of advanced software and cash flow projection models. With residential properties, however, most of the housing market is appraised using intuition rather than a systematic method. Thus, many sellers will ask for a listing price based on comparable “precedent transaction” basis. The set price is purely subjective, and the buyer will often negotiate for a better price. Whether or not a seller has sold themselves short or if a buyer overpaid for a house is impossible to discern without an objective benchmark to compare to. There are many factors that go into the valuation of a property, ranging from the geographic location to the material used in construction. The marginal value of many of these features are not intuitively quantifiable, thus here lies an opportunity to perform data mining operations to extract this non-trivial information.

New York City has traditionally been considered a safe “hot” market for real estate. In other words, residential property buyers would be able to count on, with a certain level of certainty, their purchase appreciating in value as time progressed. This understanding began to be questioned during the pandemic of 2020 which continues to this day. During the advent of the COVID-19 pandemic in early 2020, many city-dwellers in the New York City area began leaving the city. Uncertainty around the nature of the virus, dense population areas, lockdown restriction, work-from-home protocols, and a desire for more living space drove waves of potential buyers out of the city to the suburbs and rural parts of America. As mentioned previously, there are many factors that contribute to property values and influence market fluctuations. The pandemic has undoubtedly been a contributing factor to market movement in the past two years; however, this beckons the question of how significant of an impact COVID-19 has had on the market and whether we can measure it.

The objectives that I have set out for this research project are several-fold. My hypothesis is that the pandemic had a demonstrably negative significant influence on New York City residential property values:

H0 : The COVID-19 pandemic had an objectively significant negative impact on New York City residential property values.
H1 : The COVID-19 pandemic did not have an objectively significant negative impact on New York City residential property values.

My first objective is to determine which property attributes from the dataset play the greatest role in determining property valuation. My second objective will be to test several machine learning models on relevant data and identify the best performing one. The models that I used for my research are linear regression, support vector machine (SVM), random forest regression, and gradient-boosting machine (GBM). The evaluation of these models’ performance will be done using standard regression metrics such as mean squared error (MSE) and root mean squared error (RMSE). My last and final objective is to prove my null hypothesis and demonstrate that there was indeed a significant effect. For this objective I will compare prediction results for a random set of unseen properties where all variables remain constant, except for the inclusion of the COVID-19 infection count in one iteration of the prediction process.

## Methodology
### Process Outline
The following process flow chart demonstrates the simultaneous steps involved in implementing this this project, from the data source to the final prediction and analysis phase:

Figure 1. Process Flow
![image](https://github.com/wgemba/nyc-realestate-covid/assets/134420287/be0c4d7e-9519-4790-b911-e10d82cef599)

### Data Description
For this project, the datasets used were derived from several different sources and combined to great a final training dataset. The first of data sets used was the annualized property sales data published by the New York City Department of Finance Detailed Annualized Sales data for 2020. This data was separated by borough and had to be concatenated. The raw real estate property dataset contained the following 21 features:

Table 1. Raw Data
![image](https://github.com/wgemba/nyc-realestate-covid/assets/134420287/9372a7d6-1803-4cf1-b180-59416b33860d)

The raw data contains all property sales across the five boroughs for the year of 2020 – amounting to 108,550 records. For the purposes of this experiment, I filtered the raw data to only residential properties. This was done by filtering on “BUILDING CLASS AT TIME OF SALE” to include only those records that fall into one of the following building classes: 'A4', 'A5', 'A7', 'A9', 'R1', 'R2', 'R3', 'R4', 'R6', 'R7', 'R8', 'R9'; all of which are related to residential properties. After filtering, 20,472 samples remained.

This data needed to be further cleaned before any further preprocessing could be achieved. Namely, it was necessary to change data types of ‘SALE PRICE’ and ‘SALE DATE’ to their appropriate respective data types, remove redundant features (i.e., 'TAX CLASS AT PRESENT','EASE-MENT','BUILDING CLASS AT PRESENT','ADDRESS', 'APARTMENT NUMBER', ‘COMMERCIAL UNITS’, ‘TOTAL UNITS’, ‘NEIGHBORHOOD’, 'BUILDING CLASS CATEGORY', 'BLOCK', 'LOT'), remove null values, as well as remove all instances where ‘SALE PRICE’, ‘LAND SQUARE FEET’, or ‘GROSS SQUARE FEET’ equal to zero. After further cleaning there were 3,442 property sales remaining.

The second data set used was the COVID-19 Daily Counts of Cases, Hospitalizations, and Deaths provided by the New York City Department of Health and Mental Hygiene (DOHMH) and hosted on NYC Open Data. As the pandemic was declared in February, the data set begins the collection of data on February 29, 2020. To coincide with the real estate dataset that ends at the end of 2020, the COVID data set was filtered to include data up to December 31, 2020. The purpose of using this data set was to create the COVID-19 weekly average cases and weekly infection rate variables. The raw dataset contains 62 variables, however for the purposes of this research paper, only five were needed: 'MN_CASE_COUNT’, 'BK_CASE_COUNT’, 'BX_CASE_COUNT’, 'QN_CASE_COUNT’, and 'SI_CASE_COUNT; representing the daily case counts in Manhattan, Brooklyn, the Bronx, Queens, and Staten Island, respectfully. Between February 29th and the end of the year are 307 days. These days were divided into 44 weeks and for each borough the daily case count was grouped by week and the weekly average case count was calculated. Subsequently, a weekly percent change was calculated to demonstrate the trend of reported COVID infections week over week. This variable was called “WoW % CHANGE” and ended up being dropped in the feature selection process.

After all the cleanup and preprocessing, the final data set for training and testing the models contained 3,442 instances and 12 variables:

Table 2. Post Pre-Processing Data Set
![image](https://github.com/wgemba/nyc-realestate-covid/assets/134420287/75595a18-0e92-4c4f-a698-4b0eeb927c9c)

### Prediction Algorithms
For the prediction of property sales prices, four regression models were employed: Linear Regression, Random Forest Regression, Support Vector Regression, and Gradient Boosting Regressor. The models were created and run using the Scikit-Learn library in Python. For the training and testing of the dataset, a constant train/test split of 80/20 was used across the board as it demonstrated the best results.

### Feature Selection
To improve the training of the models, a layer of feature selection was introduced utilizing the correlation between each independent variable and the target variable, “SALE PRICE”.

Figure 2. Covariance Matrix of Variables in the Data Set
![image](https://github.com/wgemba/nyc-realestate-covid/assets/134420287/baff8457-b2ec-4fcc-83b4-5919371487fa)

The top four variables by absolute correlation to “SALE PRICE” were selected, those being “GROSS SQFT”, “BORO”, “YEAR BUILT”, “ZIP CODE”, and “WEEKLY_AVG_CASES”. The four regression models were trained and tested on these five variables and then the same variables (less the pandemic-related “WEEKLY_AVG_CASES” variable) were trained on the annualized sales data of the 44-week period prior to the start of the pandemic. The results section of this paper will contain a comparison the results of the two different training periods and discuss the marginal effect of COVID-19 infections on sale prices.

### Performance Metrics
To evaluate the results of the models, I used the Mean Absolute Error (MAE) and Root Mean Square Error (RMSE), with k-fold cross validation (using 10 folds). I also utilized R2 score to gauge how well the models explain the variance in the data. Based on previous research, these metrics were the most common in terms of use for measuring model performance in market value prediction.

### How to use the files
This respository includes the raw data sets and jupyter notebook files that perform the cleaning/preprocessing of data and the model construction. In the 'DATA' subdirectory, you will find two sets of raw real estate sales data from 2019 and 2020, separated by borough (Manhattam. Brooklyn, Queens, Bronx, and Staten Island). A file, , is also included for 
