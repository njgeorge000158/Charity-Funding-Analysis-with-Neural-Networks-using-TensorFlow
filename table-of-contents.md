# Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow

----

## Table of Contents (charity.ipynb)

----

# <br><br> **Section 1: Extraction**
> ## <br> **1.1: Read the CSV data into a Pandas DataFrame**
> ## <br> **1.2: Display Charity DataFrame**
# <br><br> **Section 2: Preprocessing**
> ## <br> **2.1: Change ASK_AMT Integers to Categories**
> ## <br> **2.2: Drop the Non-Beneficial ID Columns**
> ## <br> **2.3: Bin Columns**
>> ### **NAME**
>> ### **APPLICATION_TYPE**
>> ### **AFFILIATION**
>> ### **CLASSIFICATION**
>> ### **USE_CASE**
>> ### **ORGANIZATION**
>> ### **INCOME_AMT**
>> ### **ASK_AMT**
> ## <br> **2.4: Convert All Features to Uppercase**
> ## <br> **2.5: Display Updated DataFrame**
# <br><br> **Section 3: Compile, Train, Evaluate, and Export the Model**
> ## <br> **3.1: Convert and Split Data Set**
>> ### **Convert Categorical Data to Numeric with `pd.get_dummies`**
>> ### **Split Data Set into Features and Target Arrays**
>> ### **Split X-Y Arrays into Training and Testing Arrays**
> ## <br> **3.2: Create and Compile Model**
>> ### **Set Hyperparameters**
>> ### **Instantiate the Model**
>> ### **Model Summary**
>> ### **Compile**
> ## <br> **3.3: Fit, Train, and Evaluate Model**
> ## <br> **3.4: Save and Export Model**
# <br><br> **Section 4: Predict Charity Funding Success**
> ## <br> **4.1: Reload Model**
> ## <br> **4.2: Predictions**
> ## <br> **4.3: Compare Predictions and Actual Values**

----

## Table of Contents (charity_binning_optimization.ipynb)

----

# <br><br> **Section 1: Extraction**
> ## <br> **1.1: Read the CSV data into a Pandas DataFrame**
> ## <br> **1.2: Display Charity DataFrame**
# <br><br> **Section 2: Preprocessing**
> ## <br> **2.1: Number of Records**
> ## <br> **2.2: Column Counts**
> ## <br> **2.3: Unique Counts**
> ## <br>  **2.4: Value Counts**
>> ### **APPLICATION_TYPE**
>> ### **AFFILIATION**
>> ### **CLASSIFICATION**
>> ### **USE_CASE**
>> ### **ORGANIZATION**
>> ### **STATUS**
>> ### **INCOME_AMT**
>> ### **SPECIAL_CONSIDERATIONS**
>> ### **ASK_AMT**
> ## <br> **2.5: Change ASK_AMT Integers to Categories**
> ## <br> **2.6: Drop the Non-Beneficial ID Columns**
> ## <br> **2.7: Convert All Features to Uppercase**
> ## <br> **2.8: Display Updated DataFrame**
# <br><br> **Section 3: Find Optimal Binning Values**
> ## <br> **3.1: Set Dictionaries**
> ## <br> **3.2: Examine Each Column for Optimal Binning Values**
>> ### **NAME**
>> ### **APPLICATION_TYPE**
>> ### **AFFILIATION**
>> ### **CLASSIFICATION**
>> ### **USE_CASE**
>> ### **ORGANIZATION**
>> ### **INCOME_AMT**
>> ### **ASK_AMT**
> ## **3.3: Export Optimal Bins Dictionary**

----

## Table of Contents (charity_hyperparameters_optimization.ipynb)

----

# <br><br> **Section 1: Extraction**
> ## <br> **1.1: Read the CSV data into a Pandas DataFrame**
> ## <br> **1.2: Display Charity DataFrame**
# <br><br> **Section 2: Preprocessing**
> ## <br> **2.1: Change ASK_AMT Integers to Categories**
> ## <br> **2.2: Drop the Non-Beneficial ID Columns**
> ## <br> **2.3: Bin Columns**
>> ### **NAME**
>> ### **APPLICATION_TYPE**
>> ### **AFFILIATION**
>> ### **CLASSIFICATION**
>> ### **USE_CASE**
>> ### **ORGANIZATION**
>> ### **INCOME_AMT**
>> ### **ASK_AMT**
> ## <br> **2.4: Convert All Features to Uppercase**
> ## <br> **2.5: Display Updated DataFrame**
# <br><br> **Section 3: Compile, Train, Evaluate, and Export the Model**
> ## <br> **3.1: Convert and Split Data Set**
>> ### **Convert Categorical Data to Numeric with `pd.get_dummies`**
>> ### **Split Data Set into Features and Target Arrays**
>> ### **Split X-Y Arrays into Training and Testing Arrays**
> ## <br> **3.2: Define and Set Hyperparameter Ranges**
> ## <br> **3.3: Find Optimal Model Hyperparameters**

----

## Copyright

Nicholas J. George © 2024. All Rights Reserved.
