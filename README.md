<img width="769" alt="alphabet_charity_1" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/dde52c9f-5beb-418f-af31-6bb0b4386971">

----

# **Charity Funding Analysis with Neural Networks using TensorFlow**

## **Overview of the Analysis**

The purpose of this analysis is to create a binary classification model using deep learning techniques to predict if a particular charity organization, Alphabet Soup, will succeed in selecting successful applicants for funding. The model draws on a dataset of over 34,000 organizations that have received funding in the past.

## **Results**

### Data Preprocessing

- The binary classification model's target is the variable, IS_SUCCESSFUL.

- The model's feature include the variables: NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, INCOME_AMT, and ASK_AMT.

- EIN, STATUS, and SPECIAL_CONSIDERATIONS are neither targets nor features.

### Compiling, Training, and Evaluating the Model

![charity_analysisTable25UpdatedCharityDataTable](https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/dafdc9fa-c8ce-4bcb-8a37-9b85a14effb8)

- To maximze target performance, I dropped the EIN, STATUS, and SPECIAL_CONSIDERATIONS columns from the data set: these columns either had to many uniquely distributed values or virtually none.  Next, I calculated the following cutoff values for the features variables: NAME (1), APPLICATION_TYPE (156), AFFILIATION (33), CLASSIFICATION(6074), USE_CASE (146), ORGANIZATION (0), INCOME_AMT (728) and ASK_AMT (53).

<img width="657" alt="Screenshot 2024-05-07 at 9 45 07 AM" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/e94b2c58-5c17-44a2-9c16-2292760e6311">

- The model has two hidden layers and one output layer.  The two hidden layers consist of 57 and 20 neurons, respectively, and use SeLU activation functions.  The output layer has 1 neuron, which uses a Linear activation function. The model uses an Adamax optimizer and a Huber loss function. The structure maintains the ability to learn patterns effectively while striking a balance between complexity and overfitting.

<img width="925" alt="Screenshot 2024-05-07 at 9 46 47 AM" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/078081b0-ea0e-4e5b-bebf-063e5433c12a">

Once implemented, the optimized model attained a predictive accuracy of 81.41% and loss of 6.78%.

## **Summary**

Overall, through optimization, the model exceeded the target predictive accuracy of 75% with 81.41%. This exercised demonstrated the effectiveness of converting numerical data to categorical data and binning to reduce the effect the of outliers. If I were to attempt to improve performance in the future, I would, among other things, modify the optimization program to include other neural network configurations beyond sequential.

----

### Copyright

Nicholas J. George © 2023. All Rights Reserved.
