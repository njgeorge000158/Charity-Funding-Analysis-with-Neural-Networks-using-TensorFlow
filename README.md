<img width="769" alt="alphabet_charity_1" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/dde52c9f-5beb-418f-af31-6bb0b4386971">

----

# **Charity Funding Analysis with Neural Networks using TensorFlow**

## **Overview of the Analysis**

The purpose of this analysis is to create a binary classification model using deep learning techniques to predict if a particular charity organization, Alphabet Soup, will succeed in selecting successful applicants for funding. The model draws on a dataset of over 34,000 organizations that have received funding in the past.

## **Results**

### Data Preprocessing

- The variable, IS_SUCCESSFUL, is the target of the binary classification model.

- The variables – APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, ORGANIZATION, STATUS, INCOME_AMT, SPECIAL_CONSIDERATIONS, and ASK_AMT – are the features of the model.

- The variables, EIN and NAME, are neither targets nor features and are not part of the input data.

### Compiling, Training, and Evaluating the Model

- The model has two hidden layers and one output layer.  The two hidden layers consist of 95 and 38 neurons, respectively, and use ReLU activation functions.  Because this is a binary classification model, the output layer has 1 neuron and uses a Sigmoid activation function.  The structure maintains the ability to learn patterns effectively while striking a balance between complexity and overfitting.

<img width="652" alt="alphabet_charity_2" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/cef16092-2401-4a06-8313-f58cb3003619">

- Unfortunately, this model did not achieve the target performance of at least 75% predictive accuracy.

<img width="767" alt="alphabet_charity_3" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/7119ae12-7b58-451a-9e71-c6bb13331b0c">

- To achieve the target performance, I made numerous changes to the data set, preprocessing, and neural network configuration. First, I dropped the EIN, STATUS, and SPECIAL_CONSIDERATIONS columns from the data set: these columns either had to many uniquely distributed values or virtually none.  Next, I wrote a neural network optimization program, AlphabetSoupCharityOptimizationSearch.ipynb, that calculated the following cutoff values for the variables, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, and ASK_AMT: 2, 157, 65, 96, 147, and 4.  According to the program results, the optimal model for this data set contained three hidden layers with 54, 65, and 25 neurons, respectively, using tanh activation.

<img width="548" alt="alphabet_charity_4" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/5b0055b4-482b-4587-bf43-e3b0cb3a3cc6">

Once implemented, the optimized model attained a predictive accuracy of 81.11% and loss of 13.57%.

<img width="614" alt="alphabet_charity_5" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/8fd297fb-4989-4fbe-97a3-fce026f336cd">

<img width="687" alt="alphabet_charity_6" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/5789a9c4-5ce8-46e4-911c-43a9ecdcdf8a">

## **Summary**

Overall, through optimization, the model exceeded the target predictive accuracy of 75% with 81.24%.  If I were to attempt to improve performance in the future, I would, among other things, modify the optimization program to include other neural network configurations beyond sequential.

----

### Copyright

Nicholas J. George © 2023. All Rights Reserved.
