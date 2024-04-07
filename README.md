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

<img width="654" alt="alphabet_charity_2" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/ddafca29-0240-4c0e-9098-74b15ede7a5a">

- Unfortunately, this model did not achieve the target performance of at least 75% predictive accuracy.

<img width="909" alt="alphabet_charity_3" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/952b4894-2695-4153-a37e-cc572d119c2a">

- To achieve the target performance, I made numerous changes to the data set, preprocessing, and neural network configuration. First, I dropped the EIN, STATUS, and SPECIAL_CONSIDERATIONS columns from the data set: these columns either had to many uniquely distributed values or virtually none.  Next, I wrote a neural network optimization program, AlphabetSoupCharityOptimizationSearch.ipynb, that calculated the following cutoff values for the variables, NAME, APPLICATION_TYPE, AFFILIATION, CLASSIFICATION, USE_CASE, and ASK_AMT: 2, 157, 65, 96, 147, and 4.  According to the program results, the optimal model for this data set contained three hidden layers with 54, 65, and 25 neurons, respectively, using tanh activation.

<img width="582" alt="alphabet_charity_4" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/538a7640-c01e-4ab7-8b99-b97babfe7aa0">

Once implemented, the optimized model attained a predictive accuracy of 80.94% and loss of 6.36%.

<img width="646" alt="alphabet_charity_5" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/b0746930-e7bb-4b6f-8078-8db58725f2f5">

<img width="884" alt="alphabet_charity_6" src="https://github.com/njgeorge000158/Charity-Funding-Analysis-with-Neural-Networks-using-TensorFlow/assets/137228821/e1991ea9-b52d-40d2-90f7-ab9380e0e978">

## **Summary**

Overall, through optimization, the model exceeded the target predictive accuracy of 75% with 80.94%.  If I were to attempt to improve performance in the future, I would, among other things, modify the optimization program to include other neural network configurations beyond sequential.

----

### Copyright

Nicholas J. George © 2023. All Rights Reserved.
