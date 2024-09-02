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
# Note: the path has to be edited corresponding to where it is on the local computer
raw_data = pd.read_csv(r'/Users/callysta/Documents/FoundOfAI/GroupAssignment/datasets/CarDetailsV3.csv')

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
# %%
# 3.2 Scatter plot b/w 2 features
# if 1:
#     raw_data.plot(kind="scatter", y="selling_price", x="year", alpha=0.2)
#     plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
#     plt.show()    
if 1:
    raw_data.plot(kind="scatter", y="selling_price", x="km_driven", alpha=0.2)
    plt.savefig('figures/scatter_selling_price_km_driven.png', format='png', dpi=300)
    plt.show()  


# %%
# # 3.3 Scatter plot b/w every pair of features
if 1:
    from pandas.plotting import scatter_matrix 
    features_to_plot = ["selling_price", "year", "km_driven", "max_power"]
    scatter_matrix(raw_data[features_to_plot], figsize=(12, 8)) # Note: histograms on the main diagonal
    plt.savefig('figures/scatter_matrix_car_features.png', format='png', dpi=300)
    plt.show()

# %%
