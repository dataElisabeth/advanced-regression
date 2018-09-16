
# coding: utf-8

# # House Prices Prediction using Advanced Regression Techniques
# 
# In this project, I will use Ames housing prices dataset available at **Kaggle**, and apply advanced regression techniques and compare their effectiveness on such problems.

# In[35]:


# load packages
import pandas as pd # data frames
import numpy as np # arrays and computing
import seaborn as sns # statistical data visualization
import matplotlib

import matplotlib.pyplot as plt # plots

get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


# load data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.shape


# In[37]:


#check the data
train.columns
train.head()


# In[38]:


#descriptive statistics summary
train['SalePrice'].describe()


# In[39]:


#histogram
sns.distplot(train['SalePrice']);


# ## Some observations from the histogram:
# 1. Not a normal distribution
# 2. Left-skewed distribution
# 3. Has a clear peak

# In[40]:


all_data = pd.concat((train.loc[:,'MSSubClass':'SaleCondition'],
                      test.loc[:,'MSSubClass':'SaleCondition']))


# In[41]:


all_data = pd.get_dummies(all_data)
all_data.shape


# In[42]:


#filling NA's with the mean of the column:
all_data = all_data.fillna(all_data.mean())


# In[43]:


#creating matrices for sklearn:
X_train = all_data[:train.shape[0]]
X_test = all_data[train.shape[0]:]
y = np.log1p(train.SalePrice)


# In[44]:


from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV, LinearRegression, ElasticNetCV
from sklearn.model_selection import cross_val_score
import xgboost as xgb

def rmse_cv(model):
    rmse= np.sqrt(-cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv = 5))
    return(rmse)


# In[45]:


model_ridge = Ridge()


# In[46]:


alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha = alpha)).mean() 
            for alpha in alphas]


# In[47]:


cv_ridge = pd.Series(cv_ridge, index = alphas)
cv_ridge.plot(title = "Validation")
plt.xlabel("alpha")
plt.ylabel("rmse")


# In[48]:


cv_ridge.min()


# In[49]:


# train model
model_ridge = Ridge(alpha=10).fit(X_train, y)
rmse_ridge = rmse_cv(model_ridge).mean()
rmse_ridge


# In[50]:


# Linear regression
model_LinearRegr = LinearRegression()
model_LinearRegr.fit(X_train, y)
rmse_LinearRegr = rmse_cv(model_LinearRegr).mean()
rmse_LinearRegr


# In[51]:


# RidgeCV
model_RidgeCV = RidgeCV()
model_RidgeCV.fit(X_train, y)
rmse_RidgeCV = rmse_cv(model_RidgeCV).mean()
rmse_RidgeCV


# In[52]:


# ElasticNetCV 
model_EN = ElasticNetCV()
model_EN.fit(X_train, y)
rmse_EN = rmse_cv(model_EN).mean()
rmse_EN


# In[53]:


# LassoCV 
model_LassoCV= LassoCV()
model_LassoCV.fit(X_train, y)
rmse_LassoCV = rmse_cv(model_LassoCV).mean()
rmse_LassoCV


# In[54]:


# xgboost
dtrain = xgb.DMatrix(X_train, label = y)
dtest = xgb.DMatrix(X_test)

params = {"max_depth":2, "eta":0.1}
model = xgb.cv(params, dtrain,  num_boost_round=500, early_stopping_rounds=100)


# In[55]:


model_xgb = xgb.XGBRegressor(n_estimators=360, max_depth=2, learning_rate=0.1) #the params were tuned using xgb.cv
model_xgb.fit(X_train, y)


# In[56]:


rmse_xgb = model.values[-1,0]
rmse_xgb


# In[57]:


rmse_dict = {'data':[rmse_ridge, rmse_RidgeCV, rmse_LinearRegr, rmse_EN,rmse_LassoCV,rmse_xgb]}
rmse_df = pd.DataFrame(data = rmse_dict, index = ['Ridge','RidgeCV','Linear regression','ElasticNetCV','LassoCV','xgboost'])
rmse_df.plot.bar(legend = False, title = 'RMSE')


# In[58]:


# predict
ridge_preds = model_ridge.predict(X_test)


# In[59]:


# prepare data for submission
# solution = pd.DataFrame({"id":test.Id, "SalePrice":ridge_preds})
solution = pd.DataFrame({"id":test.Id})
solution = solution.assign(SalePrice = ridge_preds)
solution.head()


# In[60]:


# save file
solution.to_csv("xgboost_pred.csv", index = False)


# # So xgBoost is definitely a great algorithm for this kind of prediction.
# 
# but it can sometimes result in overfitted models.
