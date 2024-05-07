#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  deep_learningx.py
 #
 #  File Description:
 #      This Python script, deep_learningx.py, contains Python functions for processing 
 #      deep learning models. Here is the list:
 #
 #      set_sequential_model_hyperparameters_dictionary
 #      return_compiler_optimizer
 #      return_sequential_neural_net_model
 #      return_keras_tuner
 #      return_best_sequential_model_hyperparameters
 #
 #      set_binning_parameters_dictionary
 #      return_optimal_binning_value
 #      bin_series
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  04/11/2024      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/

import logx

import keras_tuner as kt
import pandas as pd
import tensorflow as tf

from IPython.display import clear_output

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

pd.options.mode.chained_assignment = None


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'deep_learningx.py'


# In[3]:


TENSORFLOW_ACTIVATION_LIST \
    = ['relu', 'elu', 'exponential', 'gelu', 'hard_sigmoid', 'hard_silu', 'leaky_relu',
       'linear', 'log_softmax', 'mish', 'relu6', 'selu', 'sigmoid', 'silu', 'softmax',
       'softplus', 'softsign', 'tanh']

TENSORFLOW_OPTIMIZER_LIST \
    = ['adam', 'adadelta', 'adafactor', 'adagrad', 'adamw', 
       'adamax', 'ftrl', 'lion', 'nadam', 'rmsprop', 'sgd']

TENSORFLOW_LOSS_LIST \
    = ['log_cosh', 'binary_crossentropy', 'binary_focal_crossentropy', 'categorical_crossentropy', 
       'categorical_focal_crossentropy', 'categorical_hinge', 'cosine_similarity', 'ctc',
       'dice', 'huber', 'hinge', 'poisson', 'sparse_categorical_crossentropy', 'squared_hinge']


# In[4]:


SEQUENTIAL_HYPERPARAMETERS_DICTIONARY \
    = {'tuner_type': 'hyperband',
       'best_model_count': 5,
       'hyperband_iterations': 2,
       'patience': 100,
       'batch_size': None,
       'validation_split': 0.2,
       'max_epochs': 1000,
       'verbose': 'auto',
       'restore_best_weights': True,
       'activation_choice_list': ['relu'],
       'input_features': 10,
       'objective': 'val_accuracy',
       'objective_direction': 'max',
       'first_layer_units_range': (100, 100),
       'first_units_step': 1,
       'first_dropout_range': (0.0, 0.0),
       'first_dropout_step': 0.01,
       'first_dropout_sampling': 'linear',
       'hidden_layers': 5,
       'hidden_layer_units_range_list': \
           [(50, 50), (40, 40), (30, 30), (20, 20), (10, 10)],
       'hidden_units_step': 1,
       'hidden_dropout_range_list': \
           [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
       'hidden_dropout_step': 0.01,
       'hidden_dropout_sampling': 'linear',
       'learning_rate_range': (25e-4, 25e-4),
       'learning_rate_step': 1e-6,
       'learning_sampling': 'linear',
       'output_activation_choice_list': ['sigmoid'],
       'output_layer_units': 1,
       'loss_choice_list': ['binary_crossentropy'],
       'optimizer_choice_list': ['adam'],
       'metrics': 'accuracy'}

BINNING_PARAMETERS_DICTIONARY \
    = {'trials': 10,
       'target': 'IS_SUCCESSFUL',
       'random_state': 21,
       'test_size': 0.25,
       'first_layer_units': 53,
       'first_activation': 'relu',
       'hidden_layer_units': 20,
       'hidden_activation': 'relu',
       'output_layer_units': 1,
       'output_activation': 'sigmoid',
       'loss': 'binary_crossentropy',
       'optimizer': 'adam',
       'metrics': 'accuracy',
       'monitor': 'val_accuracy',
       'mode': 'max',
       'patience': 5,
       'restore_best_weights': True,
       'epochs': 1000,
       'verbose': 2}


# In[5]:


#*******************************************************************************************
 #
 #  Function Name:  set_sequential_model_hyperparameters_dictionary
 #
 #  Function Description:
 #      This function sets the hyperparameters dictionary for optimizing 
 #      deep learning models.
 #
 #  Return Type: n/a
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dictionary
 #          hyperparameters_dictionary
 #                          The parameter is the new hyperparameters dictionary.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def set_sequential_model_hyperparameters_dictionary(hyperparameters_dictionary):

    global SEQUENTIAL_HYPERPARAMETERS_DICTIONARY
    
    SEQUENTIAL_HYPERPARAMETERS_DICTIONARY = hyperparameters_dictionary


# In[6]:


#*******************************************************************************************
 #
 #  Function Name:  return_compiler_optimizer
 #
 #  Function Description:
 #      This function returns an optimizer function for compilation.
 #
 #  Return Type: n/a
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  object  hp              The parameter is hyperparameter object for keras_tuner.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_compiler_optimizer(hp):

    learning_rate \
        = hp.Float('learning_rate', 
                   min_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['learning_rate_range'][0],
                   max_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['learning_rate_range'][1],
                   step = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['learning_rate_step'],
                   sampling = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['learning_sampling'])

    optimizer_choice \
        = hp.Choice('optimizer', SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['optimizer_choice_list'])

    optimizer = None

    
    if optimizer_choice == 'adam':

        optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    elif optimizer_choice == 'adadelta':

        optimizer = tf.keras.optimizers.Adadelta(learning_rate = learning_rate)

    elif optimizer_choice == 'adafactor':

        optimizer = tf.keras.optimizers.Adafactor(learning_rate = learning_rate)

    elif optimizer_choice == 'adagrad':

        optimizer = tf.keras.optimizers.Adagrad(learning_rate = learning_rate)

    elif optimizer_choice == 'adamw':

        optimizer = tf.keras.optimizers.AdamW(learning_rate = learning_rate)

    elif optimizer_choice == 'adamax':

        optimizer = tf.keras.optimizers.Adamax(learning_rate = learning_rate)

    elif optimizer_choice == 'ftrl':

        optimizer = tf.keras.optimizers.Ftrl(learning_rate = learning_rate)

    elif optimizer_choice == 'lion':

        optimizer = tf.keras.optimizers.Lion(learning_rate = learning_rate)

    elif optimizer_choice == 'nadam':

        optimizer = tf.keras.optimizers.Nadam(learning_rate = learning_rate)

    elif optimizer_choice == 'rmsprop':

        optimizer = tf.keras.optimizers.RMSprop(learning_rate = learning_rate)

    elif optimizer_choice == 'sgd':

        optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)

    
    return optimizer


# In[7]:


#*******************************************************************************************
 #
 #  Function Name:  return_sequential_neural_net_model
 #
 #  Function Description:
 #      This function returns a sequential neural network model for the optimization process.
 #
 #  Return Type: n/a
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  object  hp              The parameter is hyperparameter object for keras_tuner.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_sequential_neural_net_model(hp):

    input_features_integer = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['input_features']

    sequential_neural_net_model = tf.keras.models.Sequential()

    
    activation_choice \
        = hp.Choice('activation', SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['activation_choice_list'])

    
    sequential_neural_net_model.add \
        (tf.keras.layers.Dense \
            (units = hp.Int('units_1',
                     min_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_layer_units_range'][0],
                     max_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_layer_units_range'][1],
                     step = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_units_step']),
             activation = activation_choice,
             input_dim = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['input_features']))

    sequential_neural_net_model.add \
        (tf.keras.layers.Dropout \
             (rate = hp.Float('first_dropout_rate', 
              min_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_dropout_range'][0],
              max_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_dropout_range'][1],
              step = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_dropout_step'],
              sampling = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['first_dropout_sampling'])))
         
     
    for index in range(SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_layers']):

        sequential_neural_net_model.add \
            (tf.keras.layers.Dense \
                (units = hp.Int('units_' + str(index + 1),
                         min_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_layer_units_range_list'][index][0],
                         max_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_layer_units_range_list'][index][1],
                         step = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_units_step']),
                 activation = activation_choice))

        sequential_neural_net_model.add \
            (tf.keras.layers.Dropout \
                (rate = hp.Float('hidden_dropout_rate', 
                        min_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_dropout_range_list'][index][0],
                        max_value = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_dropout_range_list'][index][1],
                        step = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_dropout_step'],
                        sampling = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hidden_dropout_sampling'])))

    
    output_activation_choice \
        = hp.Choice('output_activation', SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['output_activation_choice_list'])
    
    sequential_neural_net_model.add \
        (tf.keras.layers.Dense \
             (units = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['output_layer_units'], 
              activation = output_activation_choice))

    
    loss_choice = hp.Choice('loss', SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['loss_choice_list'])
    
    sequential_neural_net_model.compile \
        (loss = loss_choice, 
         optimizer = return_compiler_optimizer(hp),
         metrics = [SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['metrics']])


    return sequential_neural_net_model


# In[8]:


#*******************************************************************************************
 #
 #  Function Name:  return_keras_tuner
 #
 #  Function Description:
 #      This function returns the specified keras tuner.
 #
 #  Return Type: keras tuner
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  n/a     n/a             n/a
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_keras_tuner():

    keras_tuner = None

    if SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'hyperband':
    
        keras_tuner \
            = kt.Hyperband \
                (return_sequential_neural_net_model,
                 objective \
                     = kt.Objective \
                         (SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 max_epochs = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['max_epochs'],
                 hyperband_iterations = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['hyperband_iterations'],
                 overwrite = True)

    elif SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'grid_search':

        keras_tuner \
            = kt.GridSearch \
                (return_sequential_neural_net_model,
                 objective \
                     = kt.Objective \
                         (SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 overwrite = True)

    elif SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'random_search':

        keras_tuner \
            = kt.RandomSearch \
                (return_sequential_neural_net_model,
                 objective \
                     = kt.Objective \
                         (SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 overwrite = True)

    elif SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['tuner_type'] == 'bayesian_optimization':
        
        keras_tuner \
            = kt.BayesianOptimization \
                (return_sequential_neural_net_model,
                 objective \
                     = kt.Objective \
                         (SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective'], 
                          direction = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective_direction']),
                 overwrite = True)

    
    return keras_tuner


# In[9]:


#*******************************************************************************************
 #
 #  Function Name:  return_best_sequential_model_hyperparameters
 #
 #  Function Description:
 #      This function returns the objective score, loss score, and hyperparameters
 #      for the optimal neural network models based on the hyperparameters dictionary.
 #
 #  Return Type: list
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  nparray x_train_nparray
 #                          The parameter is the scaled feature training data for the model.
 #  nparray x_test_nparray
 #                          The parameter is the scaled feature test data for the model.
 #  nparray y_train_nparray The parameter is the target training data for the model.
 #  nparray y_test_nparray  The parameter is the target test data for the model.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_best_sequential_model_hyperparameters \
        (x_train_nparray, \
         x_test_nparray, \
         y_train_nparray, \
         y_test_nparray):

    current_keras_tuner = return_keras_tuner()

    earlystopping_callback \
        = tf.keras.callbacks.EarlyStopping  \
            (monitor = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective'],
             mode = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['objective_direction'],
             patience = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['patience'],
             restore_best_weights = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['restore_best_weights'])
            
    current_keras_tuner.search \
            (x_train_nparray,
             y_train_nparray,
             batch_size = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['batch_size'],
             validation_split = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['validation_split'],
             epochs = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['max_epochs'],
             verbose = SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['verbose'],
             validation_data = (x_test_nparray, y_test_nparray),
             callbacks = [earlystopping_callback])

            
    best_model \
        = current_keras_tuner.get_best_models(SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['best_model_count'])
            
    best_hyperparameters \
        = current_keras_tuner.get_best_hyperparameters(SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['best_model_count'])

    best_models_dictionary_list = []

            
    for i in range(SEQUENTIAL_HYPERPARAMETERS_DICTIONARY['best_model_count']):

        try:
        
            best_model_loss_float, best_model_objective_float \
                = best_model[i].evaluate(x_test_nparray, y_test_nparray, verbose = 2)

            print(best_model_loss_float, best_model_objective_float, best_hyperparameters)

            best_model_dictionary \
                = {'objective': best_model_objective_float,
                   'loss': best_model_loss_float,
                   'hyperparameters': best_hyperparameters[i].values}

            best_models_dictionary_list.append(best_model_dictionary)

        except:

            pass

            
    return best_models_dictionary_list


# In[10]:


#*******************************************************************************************
 #
 #  Function Name:  set_binning_parameters_dictionary
 #
 #  Function Description:
 #      This function sets the binning parameters dictionary for optimizing 
 #      deep learning models with categorical data.
 #
 #  Return Type: n/a
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dictionary
 #          parameters_dictionary
 #                          The parameter is the new parameters dictionary.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def set_binning_parameters_dictionary(parameters_dictionary):

    global BINNING_PARAMETERS_DICTIONARY
    
    BINNING_PARAMETERS_DICTIONARY = parameters_dictionary


# In[11]:


#*******************************************************************************************
 #
 #  Function Name:  return_optimal_binning_value
 #
 #  Function Description:
 #      This function examines a dataframe column of categorical strings and returns 
 #      the optimal binning value for a sequential neural network model and an updated 
 #      dataframe column.
 #
 #
 #  Return Type: series, dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  dataframe
 #          input_dataframe The parameter is the input dataframe.
 #  string  column_string   The parameter is the dataframe column name.
 #  string  replacement_string   
 #                          The parameter is the replacement string for binned values.
 #  float   previous_accuracy_float
 #                          The parameter is the best accuracy from the previous column.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def return_optimal_binning_value \
        (input_dataframe,
         column_string,
         replacement_string,
         previous_accuracy_float = 0.0):

    best_series = 0
            
    accuracy_float = 0.0

    trial_accuracy_float = 0.0

    bins_integer = 0


    value_counts_series \
        = input_dataframe[column_string] \
            .value_counts().sort_values(ascending = False)

    value_counts_series.name = column_string


    bins_integer_list = value_counts_series.unique().tolist()

    bins_integer_list.insert(len(bins_integer_list), 0)


    bins_last_index_integer = len(bins_integer_list)


    trial_index = 1
            
    while trial_accuracy_float <= previous_accuracy_float \
        and trial_index <= BINNING_PARAMETERS_DICTIONARY['trials']:
            
        for i, bin in enumerate(bins_integer_list):

            logx.print_and_log_text \
                ('\033[1m'
                 + 'Processing: {:,}'.format(i + 1)
                 + '/{:,}\n'.format(bins_last_index_integer)
                 + '\033[0m')

            temp_dataframe = input_dataframe.copy()


            logx.print_and_log_text('\033[1m' + 'Replacing values...\n' + '\033[0m')

            if bin > 0:
        
                values_to_replace \
                    = list(value_counts_series[value_counts_series <= bin].index)
        
                for name in values_to_replace:

                    temp_dataframe[column_string] \
                        = temp_dataframe[column_string] \
                            .replace(name, BINNING_PARAMETERS_DICTIONARY['replacement'])
                

            logx.print_and_log_text('\033[1m' + 'Setting dummy values...\n' + '\033[0m')

            dummies_dataframe = pd.get_dummies(temp_dataframe)

        
            logx.print_and_log_text \
                ('\033[1m' + 'Splitting data into training and testing...\n' + '\033[0m')

            y_nparray \
                = dummies_dataframe[BINNING_PARAMETERS_DICTIONARY['target']].values

            x_nparray \
                = dummies_dataframe \
                    .drop([BINNING_PARAMETERS_DICTIONARY['target']], axis = 1).values
        
            x_train_nparray, x_test_nparray, y_train_nparray, y_test_nparray \
                = train_test_split \
                    (x_nparray, y_nparray,
                     test_size = BINNING_PARAMETERS_DICTIONARY['test_size'],
                     random_state = BINNING_PARAMETERS_DICTIONARY['random_state'])
            
        
            logx.print_and_log_text('\033[1m' + 'Creating model...\n' + '\033[0m')
        
            sequential_neural_net_model = tf.keras.models.Sequential()

        
            input_dimension_integer = len(x_train_nparray[0])

            sequential_neural_net_model.add \
                (tf.keras.layers.Dense \
                    (units = BINNING_PARAMETERS_DICTIONARY['first_layer_units'],
                     activation = BINNING_PARAMETERS_DICTIONARY['first_activation'], 
                     input_dim = input_dimension_integer))

            sequential_neural_net_model.add \
                (tf.keras.layers.Dense \
                     (units = BINNING_PARAMETERS_DICTIONARY['hidden_layer_units'],  
                      activation = BINNING_PARAMETERS_DICTIONARY['hidden_activation']))

            sequential_neural_net_model.add \
                (tf.keras.layers.Dense \
                     (units = BINNING_PARAMETERS_DICTIONARY['output_layer_units'],
                      activation = BINNING_PARAMETERS_DICTIONARY['output_activation']))

        
            logx.print_and_log_text('\033[1m' + 'Compiling...\n' + '\033[0m')
        
            sequential_neural_net_model.compile \
                (loss = BINNING_PARAMETERS_DICTIONARY['loss'],
                 optimizer = BINNING_PARAMETERS_DICTIONARY['optimizer'],
                 metrics = [BINNING_PARAMETERS_DICTIONARY['metrics']])


            logx.print_and_log_text('\033[1m' + 'Fitting data...\n' + '\033[0m')

            earlystopping_callback \
                = tf.keras.callbacks.EarlyStopping  \
                    (monitor = BINNING_PARAMETERS_DICTIONARY['monitor'],
                     mode = BINNING_PARAMETERS_DICTIONARY['mode'],
                     patience = BINNING_PARAMETERS_DICTIONARY['patience'],
                     restore_best_weights = BINNING_PARAMETERS_DICTIONARY['restore_best_weights'])


            sequential_neural_net_model \
                .fit \
                    (x_train_nparray, y_train_nparray,
                     epochs = BINNING_PARAMETERS_DICTIONARY['epochs'],
                     validation_data = (x_test_nparray, y_test_nparray),
                     callbacks = [earlystopping_callback])

            model_loss_float, model_accuracy_float \
                = sequential_neural_net_model.evaluate \
                    (x_test_nparray, y_test_nparray, 
                     verbose = BINNING_PARAMETERS_DICTIONARY['verbose'])

        
            if model_accuracy_float > accuracy_float:

                best_series = temp_dataframe[column_string]

                accuracy_float = model_accuracy_float

                bins_integer = bin

            
            clear_output(wait = True)


        if accuracy_float > trial_accuracy_float:

            trial_best_series = best_series

            trial_bins_integer = bins_integer

            trial_accuracy_float = accuracy_float


        trial_index += 1


    best_dictionary = {'bin': trial_bins_integer, 'accuracy': trial_accuracy_float}
            

    return trial_best_series, best_dictionary


# In[12]:


#*******************************************************************************************
 #
 #  Function Name:  bin_series
 #
 #  Function Description:
 #      This function bins values in a series that are equal to or less than the bin value 
 #      and returns the new series.
 #
 #
 #  Return Type: series, dictionary
 #
 #
 #  Function Parameters:
 #
 #  Type    Name            Description
 #  -----   -------------   ----------------------------------------------
 #  series
 #          input_series    The parameter is the input series.
 #  integer bin_cutoff_integer   
 #                          The parameter is the cutoff value for the binning.
 #  string  replace_string  The parameter is the replacement string for binned values.
 #
 #
 #  Date                Description                                 Programmer
 #  ---------------     ------------------------------------        ------------------
 #  04/11/2024          Initial Development                         Nicholas J. George
 #
 #******************************************************************************************/

def bin_series \
        (input_series,
         bin_cutoff_integer,
         replace_string):

    temp_series = input_series.copy()

            
    value_counts_for_binning_series \
        = temp_series.value_counts().sort_values(ascending = False)

            
    values_to_replace \
        = list(value_counts_for_binning_series[value_counts_for_binning_series <= bin_cutoff_integer].index)

    for value in values_to_replace:

        temp_series = temp_series.replace(value, replace_string)


    return temp_series


# In[ ]:




