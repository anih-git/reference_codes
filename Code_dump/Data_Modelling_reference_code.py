# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 16:29:02 2019

@author: anirban.hati
"""
import pandas as pd
%cd "D:\test_datasets"
cname  = ['age', 'workclass' ,'fnlwgt', 'education', 'education-num', 'marital-status', 
          'occupation', 'relationship', 'race', 'sex', 'capital-gain','capital-loss',
          'hours-per-week', 'native-country', 'Target']
tdata = pd.read_csv("adult.data", names = cname)

#---------::: Start of Chi-Sq Test of Dependence :::-----------

contab = pd.crosstab(tdata['workclass'], tdata['education'])

from scipy import stats
chir = stats.chi2_contingency(contab) # All of the expected frequencies are greater than 5

# Chi-square tests using the Bonferroni-adjusted p-value
import researchpy as rp
ohec_race = pd.get_dummies(tdata['race'])
ohec_relationship = pd.get_dummies(tdata['relationship'])
#ohec.drop(["?"], axis= 1, inplace= True)

for race in ohec_race:
    for rel in ohec_relationship:
        nl = "\n"
        crosstab = pd.crosstab(ohec_race[f"{race}"], ohec_relationship[f"{rel}"])
    print(crosstab, nl)
    chi2, p, dof, expected = stats.chi2_contingency(crosstab)
    print(f"Chi2 value= {chi2}{nl}p-value= {p}{nl}Degrees of freedom= {dof}{nl}")
    
    
#---------::: End of Chi-Sq Test of Dependence :::-----------    
#---------::: Start of ANOVA :::-----------
    
a = 10
print(f"kdsnfkdn : {a}")    
    
#---------::: End of ANOVA :::-----------
import pandas as pd
import scipy.stats as stats
import researchpy as rp
import statsmodels.api as sm
from statsmodels.formula.api import ols    

results = ols('libido ~ C(dose)', data=df).fit()
results.summary()
    
#---------::: Start of Test/Train Split :::-----------
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#---------::: End of Test/Train Split :::-----------


#---------::: Start of K-means ::: ----------------
import pandas as pd

df = pd.DataFrame({
    'x': [12, 20, 28, 18, 29, 33, 24, 45, 45, 52, 51, 52, 55, 53, 55, 61, 64, 69, 72],
    'y': [39, 36, 30, 52, 54, 46, 55, 59, 63, 70, 66, 63, 58, 23, 14, 8, 19, 7, 24]
})
    
from sklearn.cluster import KMeans
aa = df['x'].values()
kmeans = KMeans(n_clusters=2, random_state=0).fit(df.x.values.reshape(-1, 1))
df['cls'] = kmeans.labels_

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df.x.values.reshape(-1, 1))
    sse[k] = kmeans.inertia_ 

import matplotlib.pyplot as plt  
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()),'bx-')
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()

#---------::: End of K-means ::: ----------------


#---------::: Start of Regression Model(OLS) :::-----------
import statsmodels.api as sm # import statsmodels 

X = qdataf.drop(columns = ['Apr18_Jun18-TRX_EF']) ## X usually means our input variables (or independent variables)
y = qdataf['Apr18_Jun18-TRX_EF'] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()

#---------::: End of Regression Model(OLS) :::-----------

#---------::: Start of Random Forrest Regression :::-----------
nbad3 = pd.read_csv("nbad3 - Copy.csv")
nbad3.drop(columns = ['Jan18_Mar18-TRX_EF'], inplace = True)

aa = nbad3.describe().transpose()

x = nbad3.drop(columns = ['Apr18_Jun18-TRX_EF']).values
y = nbad3['Apr18_Jun18-TRX_EF'].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestRegressor # Get the model

# Check default Parameters
from pprint import pprint
rf = RandomForestRegressor(random_state = 42)
pprint(rf.get_params())

# Random HP grid
from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

# Random search training
# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                               n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(x_train, y_train)

# view the best parameters from fitting the random search
rf_random.best_params_

"""
rf_random.best_params_
Out[30]: 
{'n_estimators': 1600,
 'min_samples_split': 10,
 'min_samples_leaf': 2,
 'max_features': 'auto',
 'max_depth': None,
 'bootstrap': True}
"""

#Evaluate Random Search
def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    
    return accuracy

# Best model
best_random = rf_random.best_estimator_
random_accuracy = evaluate(best_random, x_test, y_test)

# Rsq value
scre_test = best_random.score(x_train, y_train)

# Grid Search based on the best values provided by random search
from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(train_features, train_labels)
grid_search.best_params_

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, test_features, test_labels)

# ----RF Model evalution----
# View Best parameters
column = call_vls04.drop(columns = ["TOTAL_CALLS"]).columns
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = column, 
                                   columns=['importance']).sort_values('importance', ascending=False)

#Evaluting the algorithm
from sklearn import metrics
y_pred = model.predict(y_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
scre_test = best_random.score(x_train, y_train)



#---------::: End of Random Forrest Regression :::-------------


#----::: Start of Partial Dependence plot ::: --------
from pdpbox import pdp
column = nbad3.drop(columns = ["Apr18_Jun18-TRX_EF"]).columns
pdp_reg = pdp.pdp_isolate(model=rf, dataset=nbad3, model_features=column,
                          feature='Apr18_Jun18-Sample',num_grid_points = 100)
#fig, axes = pdp.pdp_plot(pdp_reg, 'Sample', plot_lines=True, frac_to_plot=100)
fig, axes = pdp.pdp_plot(pdp_reg, 'Apr18_Jun18-Sample')

# Custom PDP plot---------------------------
nbad4 = nbad3[(nbad3['AprJun18RdmByJanMar18TRX']>0) | (nbad3['Apr18_Jun18-PDE']>0) | (nbad3['Apr18_Jun18-Sample']>0) ]    
nbad5 = nbad4.drop(columns = ["Apr18_Jun18-TRX_EF"])


def custom_PDP(dframe, feat_name, crange):
    
    #crange = [x * .1 for x in range(r_start,r_fin,steps)]
    
    df = pd.DataFrame(columns = [feat_name,'Pred_Trx'])
    
    for n in  tqdm(crange):
        nbad6 = dframe.copy(deep=True)
        nbad6[feat_name] = n
        ptrx = np.mean(rf.predict(nbad6.values))
        df = df.append({feat_name:n,'Pred_Trx':ptrx}, ignore_index = True) 
    
    return df

crange = [x * .1 for x in range(0,101,1)]
df = custom_PDP(nbad5, 'Apr18_Jun18-Sample', crange)

#----::: End of Partial Dependence plot ::: --------