# Big Picture of the Project and its process:  
    


### 1- Using AIML and Backtesting to Trade in Stock market

### 2- Structure designer of the project and Python code writer 

This code has user-friendly structure. We are working with time series and it is regression issue in machine learning and deep learning, as capstone project for `UC Berkeley- MFE-python-preprogram-2026`, written by `Reza Zamani`

### 3- Process of the project
1- Business Understanding (Problem statement, purposes, questions, methodology, steps, phases of project)  </span>

 <span style="color: Red; "> 2- Data understanding </span> 
 
 <span style="color: Red; "> 3- Feature Engineering </span> 
 
  <span style="color: Red; "> 4- Train/Test split and handling imbalanced dataset <span style="color: blue; ">  </span> 
  
  <span style="color: Red; ">5- Machine learning models and their evaluation   </span>
  
  <span style="color: Red; ">6- Comparison of Models performance, evaluation, best model</span>

  <span style="color: Red; ">7- Backtesting</span>
  
  <span style="color: Red; ">8- Comparison of prediction with Benchmakr (S&P500)</span>

## **4- Stages of Project** 

#### <span style="color: blue; "> **1-First Phase:** </span> 

<span style="color: blue; "> **Data Understanding** </span> 
- General checking: missing data, duplicate, incorrect data, ...
- Target Variable Visualization
- Numerical Features Visualization

<span style="color: blue; "> **Feature Engineering** </span> 

- <span style="color: blue; "> Creating financial indices to use them in model deployment </span> 

- <span style="color: blue; "> Choosing the final features to model  </span> 

####  <span style="color: green; "> **2-Second Phase:** </span> 

- <span style="color: green; "> Applying AI/ML algorithms  </span> 

- <span style="color: green; "> Choosing the best algorithm with different criteria </span> 

#### <span style="color: orange; "> **3-Third Phase:** </span> 

- <span style="color: orange; "> Backtesting </span> 

- <span style="color: orange; "> Stock Prediction </span> 


# 1- Business Understanding 

###   **1-1-Main Research Question**  
-  Using AIML model with backtesting we start to trade, how much is the final profit and portfolio value? 

###  **1-2-Other Research Questions**   
- What is the best model among different AIML algorithsm 

- What are the most important factors affecting our prediction and trading in stock market

###  **1-3-Problem Statement** 
Stock market has various volatility and different economic, social and political factors its performance. However, it would be interesting if we could design AIML model for trade, which learns from previous history and make decision for futrue. In this situation, if model works well, it would be possible to get more return than benchmark. For example, it that possible AIML algorithms can get more return that S&P500 or not. 

If we want to design AIML model for trade what we want to do. first we should check the data, uderstand it, and then make feature engineering. However, we have various features as input but it is necessary to make decision with attention to financial criteria, then we need to create financial indices. After that we need to apply different AI/ML algorithms to see which is the best model. The best approach is to have hypyerparameter tunign for each algorithms, then check their performance and choos the best model. After choosing the model, we can define backtest for trade (initial stock, enter, buy, hold, exit). after finishing backtest, we should apply it for data set of stock market. 

###  **1-4-Main Goal** 
- Using AIML models and backtet to buy and seell in stock market

###  **1-5-Other Goals** 
- Determine main features have the highest impact on stock marktet return in our best model 

- Is  that possible to have better perfomance that benchmark (S&P500) 


###  **1-6- Methodology** 
- **Regression**
###  <span style="color: red; "> **1-7-Stages of Project** </span> 

###  **1-8-Methods (AIML algorithms)**

-  **Ridge**

-  **Lasso**

-  **Random Forest**

-  **XGBoost**

-  **LSTM**

#  **2- Data Understanding** 
##  **2-1-Load Dataset**  
-  here is the dimension of our dataset (9697099, 34)

  
-  data range (beginign and end)
-   first data: 2015-01-02 
-  last date: 2025-12-31 
##  **2-2- Understanding the Features** 

***Input variables:*** 
-  Index(['id', 'symbol', 'date', 'open', 'high', 'low', 'close', 'volume',
       'adj_close', 'unadjusted_volume', 'vwap', 'reported_currency',
       'number_of_shares', 'cash_and_short_term_investments',
       'total_current_assets', 'total_assets', 'short_term_debt',
       'long_term_debt', 'total_debt', 'net_debt', 'total_liabilities',
       'net_income', 'revenue', 'gross_profit', 'ebitda', 'net_income_ttm',
       'revenue_ttm', 'gross_profit_ttm', 'ebitda_ttm', 'operating_cash_flow',
       'free_cash_flow', 'free_cash_flow_ttm', 'operating_cash_flow_ttm',
       'market_cap'],
      dtype='object')

 
###   **2-3- General check of data, info, missing and duplicate** 
###   **2-4- checking Variables correctness** 
   
# **3- Engineering Features** 
- 1- check the dataset info to be sure have all data or do not have extra

- 2- create financial indices

- 3- our final features are: ['pe_ratio', 'pb_ratio', 'fcf_yield', 'return_20d', 'return_60d', 'volatility_20d', 'volatility_60d', 'volume_ratio', 'volume_trend', 'roe', 'revenue_growth', 'profit_margin', 'rsi_14', 'return_to_vol_20d']

- 4- final data set size: X_train shape: (32838, 14), X_test shape: (86720, 14), X shape: (119558, 14), y shape: (119558,)

- 5- Split the data into features and target

- 6- `StandardScaler()` for numerical variables 

#  **4- Train/test split and Handling Imbalanced Datasets** </span>  
##  **4-1-Train/Test Split** 
With data prepared, we split it into a train and test set. here is the duration for train and test 

TRAIN_START = '2015-01-01'


TRAIN_END = '2017-12-31'


GAP_END = '2018-06-31'      # 6-month gap to prevent leakage


TEST_START = '2018-07-01'


TEST_END = '2025-12-31'

# <span style="color: red; ">  **5- Machine learning Models** </span> 
Strategy:

- 1-At first we define and fit each model
- 2- in each model, we see accuracy
- 3- create dataframe to see all score together
- 5- with attention to all criteria, choose the best model in this step
#### <span style="color: red; ">  **5-1-Ridge** </span> 
#### <span style="color: red; ">  **5-2-Lasso** </span> 
#### <span style="color: red; ">  **5-3-Random Forest** </span>
#### <span style="color: red; ">  **5-4-XGBoost** </span> 
#### <span style="color: red; ">  **5-5- LSTM** </span> 

Evaluation: 

============================================================
MODEL COMPARISON SUMMARY
============================================================

                  RMSE       MAE         R2  Direction_Accuracy
Ridge         0.187150  0.126259  -0.075480           61.057426
Lasso         0.188194  0.125815  -0.087509           61.358395
XGBoost       0.186200  0.128325  -0.064585           58.701568
RandomForest  0.185775  0.127520  -0.059734           59.757841
LSTM          1.197054  0.172032 -42.995353           60.803829


Best Models by Metric:
  Lowest RMSE: RandomForest (0.185775)
  Lowest MAE: Lasso (0.125815)
  Highest R²: RandomForest (-0.059734)
  Highest Direction Accuracy: Lasso (61.36%)

************************************************************
OVERALL BEST MODEL: RandomForest
************************************************************

## <span style="color: red; ">  **5-6-feature importance** </span> 

all features and their weight:
------------------------------------------------------------
          feature  importance
    profit_margin    0.261065
         pe_ratio    0.221704
   revenue_growth    0.179606
        fcf_yield    0.160002
              roe    0.062874
         pb_ratio    0.049997
   volatility_60d    0.038633
   volatility_20d    0.011707
       return_20d    0.008196
       return_60d    0.003487
return_to_vol_20d    0.001430
           rsi_14    0.000898
     volume_trend    0.000257
     volume_ratio    0.000145

## <span style="color: red; ">  **Part 5: The Backtester Class¶** </span> 

Now for the main event - the Backtester class. This orchestrates everything:

Tracks cash - How much money is available to invest
Manages positions - Opens and closes positions
Runs day-by-day - Iterates through the test period
Calls your model - Gets predictions each day
Calls your allocator - Decides how much to invest
Records history - Stores data for analysis


## <span style="color: red; ">  **Task 1: Build Your Prediction Model** </span> 

## <span style="color: red; ">  **Task 2: Prepare Training Data and Train Your Model¶** </span> 

You need to:

 - Calculate the target variable (90-day forward return)
- Select and engineer features
- Train your model on 2010-2015 data

## <span style="color: red; ">  **Task 3: Implement Your Allocation Function¶** </span> 

This function decides:

- Which stocks to buy based on predictions
- How much to invest in each stock
- The backtester calls this function every trading day.


## <span style="color: red; ">  **Task 5: Performance Metrics and Visualization** </span> 

After running the backtest, you need to calculate:

- Final portfolio value
- Annualized return
- Maximum drawdown
- Sharpe ratio
- And create a cumulative return plot comparing your strategy to the benchmark.

## <span style="color: red; ">  **Part 8: Running the Backtest** </span> 


============================================================
RUNNING BACKTEST
============================================================
This may take a few minutes...

Running backtest from 2018-04-01 to 2025-12-31
Trading days: 1950
Initial cash: $1,000,000.00
--------------------------------------------------
[2020-03-25] Day 500/1950 | Portfolio: $1,542,822 | Positions: 126 | Cash: $4,021
[2022-03-18] Day 1000/1950 | Portfolio: $3,080,950 | Positions: 149 | Cash: $208,509
[2024-03-15] Day 1500/1950 | Portfolio: $5,019,275 | Positions: 86 | Cash: $135,268
--------------------------------------------------
Backtest complete!
Final portfolio value: $10,170,165.47
Total positions opened: 4059
Positions still open: 66

==================================================
         PORTFOLIO PERFORMANCE
==================================================
Final Portfolio Value:    $10,170,165.47
Annualized Return:        34.90%
Maximum Drawdown:         -42.37%
Sharpe Ratio:             1.08

==================================================
         BENCHMARK (S&P 500)
==================================================
Final Value:              $6,898,249.18
Annualized Return:        28.31%

==================================================
              RESULT
==================================================
Beat Benchmark:           Yes
Outperformance:           +47.43%
==================================================



