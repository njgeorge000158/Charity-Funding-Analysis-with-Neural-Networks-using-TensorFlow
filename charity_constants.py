#!/usr/bin/env python
# coding: utf-8

# In[1]:


#*******************************************************************************************
 #
 #  File Name:  charity_constants.py
 #
 #  File Description:
 #      This Python script, charity_constants.py, contains generic Python constants
 #      for tasks in the Google Colab Notebooks, charity_colab.ipynb and
 #      charity_optimization_colab.ipynb.
 #
 #
 #  Date            Description                             Programmer
 #  ----------      ------------------------------------    ------------------
 #  11/27/2023      Initial Development                     Nicholas J. George
 #
 #******************************************************************************************/

import logx


# In[2]:


CONSTANT_LOCAL_FILE_NAME = 'charity_constants.py'


# In[3]:


INPUT_FILE_PATH = logx.RESOURCES_DIRECTORY_PATH + '/charity_data.csv'

BINNING_PARAMETERS_FILE_PATH = logx.MODELS_DIRECTORY_PATH + '/charity_data_binning_parameters.sav'

BINNED_CHARITY_DATA_FILE_PATH = logx.MODELS_DIRECTORY_PATH + '/charity_data_binned.sav'

MODEL_FILE_PATH = logx.MODELS_DIRECTORY_PATH + '/charity_nn_model.keras'


# In[4]:


NN_BATCH_SIZE = 64

NN_TEST_SIZE = 0.2

NN_MAX_EPOCHS = 1000

NN_VERBOSE = 'auto'

NN_RANDOM_STATE = 21


# In[ ]:




