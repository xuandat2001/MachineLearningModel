# In[0]: IMPORT AND FUNCTIONS
#region 
# pip install scikit-learn # to install sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer  
from sklearn.preprocessing import OneHotEncoder      
from sklearn.model_selection import KFold   
from statistics import mean
import joblib 
#endregion

# In[1]: STEP 1. LOOK AT THE BIG PICTURE (DONE)



# In[2]: STEP 2. GET THE DATA (DONE). LOAD DATA
raw_data = pd.read_csv(r'C:\RMIT Programing\Python Programming\Asm3\datasets\CarDetailsV3.csv')


# In[3]: STEP 3. DISCOVER THE DATA TO GAIN INSIGHTS
#region
# 3.1 Quick view of the data
print('\n____________ Dataset info ____________')
print(raw_data.info())      
print('\n____________ Some first data examples ____________')
print(raw_data.head(3)) 
print('\n____________ Counts on a feature ____________')
print(raw_data['fuel'].value_counts()) 
print('\n____________ Statistics of numeric features ____________')
print(raw_data.describe())    