# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:31:56 2019

@author: anirban.hati
This is a reference script for Data analytics
"""
#---------::: Start of Install useful Libraries :::-------
import pandas as pd # for dataframe/tables
import os           # for directory 
import numpy as np  # for numeric calculations
#---------::: End of Install useful Libraries :::-------

#---------::: Start of Data Loading ::::------------ 
os.getcwd()               # get current directory
os.chdir("D:\Python_cd")  # change current directory  
%cd "D:\Python_cd"        # change current directory 
##--- From Text file ---
nbad = pd.read_csv("ChannelAffinity_Combineddata_Ver1.0.txt",sep='\t') 

nbad = pd.read_csv("MTpharma_start_to_Oct18.txt", encoding = "ISO-8859-1")

##--- From Excel File ---
nbad = pd.read_excel ('Data Request  For Galderma _Oracea_Soolantra_DLP_06172019.xlsx', sheet_name='Data _ Oracea_Soolantra')


##---  From SQL ----

import pyodbc 
import pandas as pd
from tqdm import tqdm

dbnames = ['Abbott_Juven', 'Mylan_Symfilo', 'Nutent_Volt', 'Promius_Trianex',
           'Promius_Zembrace']

pcdata = pd.DataFrame()
for dbs in tqdm(range(len(dbnames))):
    cnxn = pyodbc.connect("Driver={SQL Server};"
                        "Server=datapipeline.database.windows.net;"
                        "Database="+ str(dbnames[dbs])+";"
                        "uid=pc_db_read;pwd=Welcome123")
    df = pd.read_sql_query(("SELECT convert(date, delivered_time) as Del_Date,  \
                            datepart(HOUR, delivered_time) as Del_Hour, email_subject, \
                            delivered_count, opened_count, unique_opened_count , '"+str(dbnames[dbs]) + "'as Campaign \
                            from standard.tbl_email_data where source_name = 'Click_dimension';"), cnxn)
    cnxn.close()
    pcdata = pcdata.append(df)
    
    
# Convert date from string to date times
data['date'] = data['date'].apply(dateutil.parser.parse, dayfirst=True)

#---------::: End of Data Loading ::::------------

#---------::: Start of Package Instalation ::::------------
import pip
from pip._internal.utils.misc import get_installed_distributions

installed_pkgs = [pkg.key for pkg in get_installed_distributions()]
#---------::: End of Package Instalation ::::------------

#---------:::  Start of Data Manipulation :::-----------
##--- Groupby
# Get the sum of the durations per month
data.groupby('month')['duration'].sum()
# What is the sum of durations, for calls only, to each network
data[data['item'] == 'call'].groupby('network')['duration'].sum()
# produces Pandas Series
data.groupby('month')['duration'].sum() 
# Produces Pandas DataFrame
data.groupby('month')[['duration']].sum()
# Group the data frame by month and item and extract a number of stats from each group
data.groupby(
   ['month', 'item'], as_index=False
).agg(
    {
         'duration':'sum',    # Sum duration per group
         'network_type': 'count',  # get the count of networks
         'date': 'first'  # get the first date per group
    }
)

# Define the aggregation procedure outside of the groupby operation
aggregations = {
    'duration':'sum',
    'date': lambda x: max(x) - 1
}
data.groupby('month').agg(aggregations)

# Group the data frame by month and item and extract a number of stats from each group
data.groupby(
    ['month', 'item']
).agg(
    {
        # find the min, max, and sum of the duration column
        'duration': [min, max, sum],
         # find the number of network type entries
        'network_type': "count",
        # min, first, and number of unique dates per group
        'date': [min, 'first', 'nunique']
    }
)

# Renaming grouped statistics from groupby operations
grouped = data.groupby('month').agg("duration": [min, max, mean]) 
# Using ravel, and a string join, we can create better names for the columns:
grouped.columns = ["_".join(x) for x in grouped.columns.ravel()]

# Pandas String subsetting
tdata['tstring'] = tdata['education'].str[3:5]
#---------::: End of Data Manipulation :::-----------

#---------::: Start of Class grouping based on count ::::------------
tdata = pd.DataFrame({'Code': ['PD','PD','PA','PA','NPS','PA','US','FP','D','D',	
                               'D',	'PA','D','D','PA','IM','D','D','D','IM','NPS']})

aa = tdata['Code'].value_counts()[:3].index.tolist()
tdata['Coden'] = np.where(tdata['Code'].isin(aa), tdata['Code'], 'Other')
#---------::: End of Class grouping based on count ::::------------

#---------::: Start of Data Summary ::::------------ 

##-------Pandas Profiling
import pandas_profiling as pp
aa = pp.ProfileReport(afdata)
#profile = df.profile_report(title='Pandas Profiling Report')
aa.to_file(outputfile="output.html")


aa = nbad3.describe().transpose() # ::: Get data summary

# For Categorical Columns
cat_colmn = nbad3.select_dtypes(include='object').columns

Freq_Tab = pd.DataFrame(columns = ['CAT', 'FREQ','Colname'])

for i in cat_colmn:
    cc = pd.DataFrame(data = nbad3.groupby(i).size()).reset_index(level = 0) 
    cc.columns = ['CAT', 'FREQ']
    cc['Colname'] = i
    
    Freq_Tab = Freq_Tab.append(cc, ignore_index=True)


#---------::: End of Data Summary ::::------------ 


#------- ::: Start of Different Percentile values :::--------
# Check different percentile values
def get_percentile(data_frame, feature_list, per_points):
    dframe = data_frame.copy(deep=True)
    feat_df = pd.DataFrame(feature_list, columns = ['feature'])
    for pp in per_points:
        cname = 'PP_'+ str(pp)
        pp_df = pd.DataFrame(columns = [cname])
        for features in feature_list:
            pp_df = pp_df.append({cname: dframe[features].quantile(pp)}, ignore_index=True)
        
        feat_df =  pd.concat([feat_df, pp_df], axis = 1)       
    return feat_df  


pplist = ['Apr18_Jun18-Sample', 'Jan18_Mar18-Sample', 'Jan18_Mar18-PDE', 'Apr18_Jun18-PDE',
          'Jan18_Mar18-TRX_EF', 'Apr18_Jun18-TRX_EF', 'Jan18_Mar18-TRX_Mkt',
          'AprJun18RdmByJanMar18TRX', 'JanMar18RdmByOctDec17TRX']

pp_dframe = get_percentile(nbad3, pplist, [0.75, 0.9,0.95,0.99,0.995])
#------- ::: End of Different Percentile values :::--------

#------::: Start of Outlier Analysis :::-------
def outlier_analysis(dframe, columns):
    """
    This is to analyse outliers
    Arguments: df-Data Source; columns - input columns
    """
    df = dframe.copy(deep=True)
    # Using IQR
    boxplot_df = pd.DataFrame(columns = ['Column', 'Q1', 'Q3', 'IQR', 
                                   'Upper_Whisker', 'Lower_Whisker'])
    for i in columns:
        Q1 = df[i].quantile(0.25)
        Q3 = df[i].quantile(0.75)
        IQR = Q3 - Q1
        UWH = Q3 + 1.5*IQR
        LWH = Q1 - 1.5*IQR
        boxplot_df = boxplot_df.append({'Column': i, 'Q1':Q1, 'Q3': Q3, 'IQR': IQR, 
                                   'Upper_Whisker':UWH, 'Lower_Whisker': LWH}, ignore_index=True)
        
        nc = i +'_IQR' # New column name
        df.loc[df[i] < LWH, nc] = 'BLW'
        df.loc[df[i] > UWH, nc] = 'AUW'
        df.loc[(df[i] <= UWH) & (df[i] >= LWH), nc] = 'SV'
        
        #Using Z-score(The value should be between -3 to 3)
        ncz = i + '_zsc' #New column 
        df[ncz] = np.abs(stats.zscore(df[i]))
        

    return df, boxplot_df

numcolmn = list(nbad2.select_dtypes(exclude=['object']))

ul_df, boxdf = outlier_analysis(nbad2, numcolmn)
 


# Using Zscore-The value should be between -3 to 3
from scipy import stats
import numpy as np
z = np.abs(stats.zscore(nbad2['Prd_TRX_UNIT_JUN_2018']))

#------::: End of Outlier Analysis :::------------------
#------::: Start of Histogram & Correlation Plot :::---------
# Plot Histogram
import matplotlib.pyplot as plt
nbad1.hist(figsize = (12,10))
plt.show()
    

#Correlation plot
import seaborn as sb
cor_mat = nbad1.corr()

def cor_heatmap(df):
    """ 
    Param: df - Datafame
    O/P: Heatmap
    """   
    cor_mat = df.corr() # correlation matrix
    # Plot figsize
    fig, ax = plt.subplots(figsize = (15,15))
    #Generate Color Map, Red & Blue
    colormap = sb.diverging_palette(220, 10, as_cmap=True)
    #Generate Heat Map, allow annotations and place floats in map
    sb.heatmap(cor_mat, cmap=colormap, annot=True, fmt=".2f")
    #Apply xticks
    plt.xticks(range(len(cor_mat.columns)), cor_mat.columns);
    #Apply yticks
    plt.yticks(range(len(cor_mat.columns)), cor_mat.columns)
    #show plot
    plt.show()

cor_heatmap(nbad1)
#------::: End of Histogram & Correlation Plot :::---------
nbad.Subject Line.unique
bb = nbad.groupby(['Email Name', 'Subject Line']).size()

#-----::: Start of Variable Standardisation ::::---------------
# standardizing the input feature
from sklearn.preprocessing import StandardScaler
def get_std_table(tdaf, target):
    # Get numerical columns in seperate table
    catcolmn = list(tdaf.select_dtypes(include=['object']))
    catcolmn.append(target)
    tdaf_num = tdaf.drop(columns = catcolmn)
    tdaf_cat = tdaf[catcolmn]
    sc = StandardScaler()
    tdaf_num_std = sc.fit_transform(tdaf_num)
    tdaf_num_std = pd.DataFrame(data = tdaf_num_std, columns = tdaf_num.columns)
    
    return pd.concat([tdaf_num_std,tdaf_cat],axis=1)


tdaf_ss = get_std_table(tdaf, 'Time-Day-Category')

# Min-Max Scaling
from sklearn.preprocessing import MinMaxScaler
def get_mm_table(tdaf, target = 'NR'):
    # Get numerical columns in seperate table
    catcolmn = list(tdaf.select_dtypes(include=['object']))
    if target != 'NR':
        catcolmn.append(target)
    tdaf_num = tdaf.drop(columns = catcolmn)
    tdaf_cat = tdaf[catcolmn]
    mm = MinMaxScaler()
    tdaf_num_nor = mm.fit_transform(tdaf_num)
    tdaf_num_nor = pd.DataFrame(data = tdaf_num_nor, columns = tdaf_num.columns)
    
    return pd.concat([tdaf_num_nor,tdaf_cat],axis=1)


X_mm = get_mm_table(X)


#---------::: End of Variable Standardisation ::::---------------

#------::: Start of 1-Hot encoding for categorical variables :::--------
def onehotenc(df, colname):
    for i in colname:
        if(df[i].dtype == np.dtype('object')):
            ohec = pd.get_dummies(df[i], prefix = i)
            df = pd.concat([df,ohec], axis = 1)
            df.drop([i], axis =1, inplace = True)
            
    return df        
            
            
catcolmn = list(tdaf.select_dtypes(include=['object']))          
print('There were {} columns before OHEC: '.format(tdaf.shape[1]))
tdaf2 =  onehotenc(tdaf, catcolmn)           
print('There are {} columns after OHEC: '.format(tdaf2.shape[1]))
#------::: End of 1-Hot encoding for categorical variables :::--------

#------::: Start of Imbalance Class Check::::---------------
# Check for Imbalance classes
print("Number transactions X_train dataset: ", x_train.shape)
print("Number transactions y_train dataset: ", y_train.shape)
print("Number transactions X_test dataset: ", x_test.shape)
print("Number transactions y_test dataset: ", y_test.shape)

from imblearn.over_sampling import SMOTE
print("Before OverSampling, counts of label '2': {}".format(sum(y_train==2)))
print("Before OverSampling, counts of label '3': {} \n".format(sum(y_train==3)))

sm = SMOTE(random_state=2)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(x_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))

print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
print("After OverSampling, counts of label '3': {}".format(sum(y_train_res==3)))
#------::: End of Imbalance Class Check::::---------------


#------::: Start of VIF Check::::---------------
from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif(df):
    """"
    df: Dataframe with independent continuous features
    """
    vif_columns = list(df.columns)
    variables = list(range(df.shape[1]))
    vif = [variance_inflation_factor(df.iloc[:, variables].values, ix)
                   for ix in range(df.iloc[:, variables].shape[1])]
    
    vif_df = pd.DataFrame.from_dict({'Features': vif_columns, 'VIF': vif})
    return vif_df
    
vifdf =  get_vif(X)   

    
#---------::: End of VIF Check::::-----------------

#-------------::: Start of Correlation with Significance :::--------------
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.graphics.api as smg
import pandas as pd
from scipy.stats import pearsonr


cdata = pd.read_excel("Xiidra_Regression_data.xlsx, 'new_correlation_data'")
cdata = cdata.drop(columns = ["Year-Month"])

def calculate_pvalues(df):
    #df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            tdf = pd.DataFrame({"r" : df[r], "c": df[c]})
            tdf1 = tdf.dropna()
            pvalues[r][c] = str(round(pearsonr(tdf1["r"], tdf1["c"])[0], 4)) + "_" + str(round(pearsonr(tdf1["r"], tdf1["c"])[1], 4))
    return pvalues

aa = calculate_pvalues(cdata)

aa.to_csv("pear.csv")

#-------------::: End of Correlation with Significance :::--------------


