import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew
from scipy.special import boxcox1p   
from scipy.special import inv_boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_score

#-----------------------LOAD DATA----------------------------

train = pd.read_csv('train.csv', delimiter=',')
test = pd.read_csv('test.csv', delimiter=',')


#--------------------MISSING VALUES--------------------------

#Drop rows with more than 80% NaN
NAperRow = train.isnull().mean(axis=1)  # mean of NaN per row
threshold = 0.8
train_clean = train[NAperRow <= threshold].copy()

#Stack train and test datasets to process missing values
train_test = pd.concat((train_clean.loc[:, 'MSSubClass':'SaleCondition'], 
                       test.loc[:, 'MSSubClass':'SaleCondition'])).reset_index(drop=True) 

#Id and SalePrice variables won't be handled in the missing-values process
train_test = train_test[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
       'LotShape', 'LandContour', 'Utilities', 'LotConfig',
       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
       'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
       'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
       'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
       'SaleCondition'
       #'Fence','Alley', 'Id', 'SalePrice'
                          ]]

#Missing values imputation 'MSZoning'
train_test['MSZoning'] = train_test.groupby('Neighborhood')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

#Missing values imputation 'MiscFeature'
train_test.loc[train_test['MiscVal'] == 0, 'MiscFeature'] = 'NA'
train_test['MiscFeature'] = train_test['MiscFeature'].fillna('Othr')

#Missing values imputation 'LotFrontage'
train_test['LotFrontage'] = train_test.groupby('Neighborhood')['LotFrontage'] \
                                      .transform(lambda x: x.fillna(x.mean()))
                                      
#Missing values imputation 'MasVnrType'
train_test.loc[train_test['MasVnrArea'].isna(),['MasVnrType', 'MasVnrArea']] = ['None', 0]
train_test.loc[train_test['MasVnrType'].isna(),'MasVnrType'] = 'BrkFace'

#Missing values imputation 'BsmtQual, BsmtCond, BsmtExposure, BsmtFinType1, BsmtFinType2'
train_test.loc[train_test['BsmtCond'].isna(), 
               ['BsmtQual','BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']] = 'NA'
train_test.loc[train_test['BsmtCond'] == 'NA', 
                ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0

train_test.loc[train_test['BsmtExposure'].isna(),'BsmtExposure'] = 'No' # Id=948,1487,2348

train_test.loc[train_test['TotalBsmtSF'].isna(), 
                 ['TotalBsmtSF','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'BsmtFullBath', 'BsmtHalfBath']] = 0

train_test['BsmtQual'] = train_test['BsmtQual'].fillna(train_test['BsmtCond'])
moda = train_test.loc[train_test['BsmtFinType1'] == 'GLQ']['BsmtFinType2'].mode()[0]
train_test['BsmtFinType2'] = train_test['BsmtFinType2'].fillna(moda)

#Missing values imputation 'Electrical, Utilities, Functional, KitchenQual, SaleType'
for x in ('Electrical', 'Utilities', 'Functional', 'KitchenQual', 'SaleType'):
    train_test[x] = train_test[x].fillna(train_test[x].mode()[0])

#Missing values imputation 'Exterior1st, Exterior2nd'
Ext1_counts = train_test['Exterior1st'].value_counts().sort_values(ascending=False)

train_test.loc[train_test['Exterior1st'].isna(), ['Exterior1st', 'Exterior2nd']] \
         = [Ext1_counts.index[0], Ext1_counts.index[1]]     

 
#Missing values imputation 'FireplaceQu'
train_test.loc[train_test['Fireplaces'] == 0, 'FireplaceQu'] = 'NA'
 
 
#Missing values imputation 'GarageType, GarageYrBlt, GarageFinish, GarageQual, GarageCond'
train_test.loc[train_test['GarageCars'] == 0, ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']] = 'NA'

train_test['GarageYrBlt'].fillna(train_test['YearBuilt'], inplace=True)
for x in ('GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond'):
    train_test[x] = train_test[x].fillna(train_test[x].mode()[0])
 
 
#Missing values imputation 'PoolQC'
train_test.loc[train_test['PoolArea'] == 0, 'PoolQC'] = 'NA'
train_test['PoolQC'] = train_test['PoolQC'].fillna('TA')


#Check years
train_test.loc[train_test['YearBuilt'] > train_test['YearRemodAdd'], 'YearRemodAdd'] = train_test['YearBuilt']
train_test.loc[train_test['YearBuilt'] > train_test['YrSold'], 'YrSold'] = train_test['YearRemodAdd']
train_test.loc[train_test['YearRemodAdd'] > train_test['YrSold'], 'YrSold'] = train_test['YearRemodAdd']

#Split datasets
train_sel = train_test.head(len(train_clean)).copy() 
test_sel = train_test.tail(len(test)).copy()
test_sel.reset_index(drop=True, inplace=True)
train_sel['SalePrice'] = train_clean['SalePrice']


#--------------VARIABLE TRANSFORMATION----------------------

var = 'MSSubClass'
train_sel[var] = train_sel[var].astype(str)
test_sel[var] = test_sel[var].astype(str)

var = 'MoSold'
mapping = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul',
          8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
train_sel[var] = train_sel[var].replace(mapping)
test_sel[var] = test_sel[var].replace(mapping)

var = 'PavedDrive'
encoder = OrdinalEncoder(categories=[["N", "P", "Y"]])
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'Electrical'
mapping = {'Mix':'Othr', 'FuseP':'Othr', 'FuseF':'Othr'}
train_sel[var] = train_sel[var].replace(mapping)
test_sel[var] = test_sel[var].replace(mapping)

var = 'FireplaceQu'
encoder = OrdinalEncoder(categories=[["Po","NA", "Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'CentralAir'
encoder = OrdinalEncoder(categories=[["N", "Y"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'HeatingQC'
encoder = OrdinalEncoder(categories=[["Po","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'KitchenQual'
encoder = OrdinalEncoder(categories=[["Po","Fa","TA","Gd","Ex"]])
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'ExterQual'
encoder = OrdinalEncoder(categories=[["Po","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'ExterCond'
encoder = OrdinalEncoder(categories=[["Po","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'GarageFinish'
encoder = OrdinalEncoder(categories=[["NA","Unf","RFn","Fin"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'GarageQual'
encoder = OrdinalEncoder(categories=[["NA","Po","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'GarageCond'
encoder = OrdinalEncoder(categories=[["NA","Po","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'BsmtQual'
encoder = OrdinalEncoder(categories=[["NA","Po","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'BsmtCond'
encoder = OrdinalEncoder(categories=[["Po","NA","Fa","TA","Gd","Ex"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'BsmtExposure'
encoder = OrdinalEncoder(categories=[["NA","No", "Mn","Av","Gd"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'BsmtFinType1'
encoder = OrdinalEncoder(categories=[["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

var = 'BsmtFinType2'
encoder = OrdinalEncoder(categories=[["NA","Unf","LwQ","Rec","BLQ","ALQ","GLQ"]]) 
encoder.fit(train_sel[[var]])
train_sel[var+'-encoded'] = encoder.transform(train_sel[[var]])
encoder.fit(test_sel[[var]])
test_sel[var+'-encoded'] = encoder.transform(test_sel[[var]])

#----------------FEATURE ENGINEERING----------------
train_sel['TotalSF'] = (train_sel['TotalBsmtSF'] 
                       + train_sel['1stFlrSF'] 
                       + train_sel['2ndFlrSF'])
test_sel['TotalSF'] = (test_sel['TotalBsmtSF'] 
                       + test_sel['1stFlrSF'] 
                       + test_sel['2ndFlrSF'])

train_sel['TotalFinishSF'] = (train_sel['BsmtFinSF1'] 
                                 + train_sel['BsmtFinSF2'] 
                                 + train_sel['1stFlrSF'] 
                                 + train_sel['2ndFlrSF'])
test_sel['TotalFinishSF'] = (test_sel['BsmtFinSF1'] 
                                 + test_sel['BsmtFinSF2'] 
                                 + test_sel['1stFlrSF'] 
                                 + test_sel['2ndFlrSF'])

train_sel['has2ndFloor'] = 0
train_sel.loc[train_sel['2ndFlrSF'] != 0, 'has2ndFloor'] = 1
test_sel['has2ndFloor'] = 0
test_sel.loc[test_sel['2ndFlrSF'] != 0, 'has2ndFloor'] = 1

train_sel['hasBsmt'] = 0
train_sel.loc[train_sel['TotalBsmtSF'] != 0, 'hasBsmt'] = 1
test_sel['hasBsmt'] = 0
test_sel.loc[test_sel['TotalBsmtSF'] != 0, 'hasBsmt'] = 1

train_sel['hasShed'] = 0
train_sel.loc[train_sel['MiscFeature'] == 'Shed', 'hasShed'] = 1
test_sel['hasShed'] = 0
test_sel.loc[test_sel['MiscFeature'] == 'Shed', 'hasShed'] = 1

train_sel['hasGarage'] = train_sel['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
test_sel['hasGarage'] = test_sel['GarageArea'].apply(lambda x: 1 if x > 0 else 0)

train_sel['hasFireplace'] = train_sel['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
test_sel['hasFireplace'] = test_sel['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)

train_sel['hasPool'] = train_sel['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
test_sel['hasPool'] = test_sel['PoolArea'].apply(lambda x: 1 if x > 0 else 0)

train_sel['TotalBaths'] = train_sel['FullBath'] \
                        + 0.5 * train_sel['HalfBath'] \
                        + train_sel['BsmtFullBath'] \
                        + 0.5 * train_sel['BsmtHalfBath']
test_sel['TotalBaths'] = test_sel['FullBath'] \
                        + 0.5 * test_sel['HalfBath'] \
                        + test_sel['BsmtFullBath'] \
                        + 0.5 * test_sel['BsmtHalfBath']
                        
train_sel['TotalPorchSF'] = train_sel['ScreenPorch'] \
                            + train_sel['3SsnPorch'] \
                            + train_sel['EnclosedPorch'] \
                            + train_sel['OpenPorchSF'] \
                            + train_sel['WoodDeckSF']
test_sel['TotalPorchSF'] = test_sel['ScreenPorch'] \
                            + test_sel['3SsnPorch'] \
                            + test_sel['EnclosedPorch'] \
                            + test_sel['OpenPorchSF'] \
                            + test_sel['WoodDeckSF']
                            
train_sel['Age'] = train_sel['YrSold'] - train_sel['YearRemodAdd']
test_sel['Age'] = test_sel['YrSold'] - test_sel['YearRemodAdd']

train_sel['isRemodeled'] = 0
train_sel.loc[train_sel['YearRemodAdd'] > train_sel['YearBuilt'], 'isRemodeled'] = 1
test_sel['isRemodeled'] = 0
test_sel.loc[test_sel['YearRemodAdd'] > test_sel['YearBuilt'], 'isRemodeled'] = 1

train_sel['isNew'] = 0
train_sel.loc[train_sel['YearBuilt'] == train_sel['YrSold'], 'isNew'] = 1
test_sel['isNew'] = 0
test_sel.loc[test_sel['YearBuilt'] == test_sel['YrSold'], 'isNew'] = 1

#Feature selection
train_sel = train_sel[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
                       'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
                       'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 
                       'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 
                       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 
                       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual','BsmtCond', 
                       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 
                       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
                       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 
                       'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu','hasFireplace',
                       'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 
                       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition', 
                       'TotalPorchSF', 'TotalBaths', 'Age', 'hasPool', 'TotalSF', 'hasGarage',
                       'isRemodeled', 'hasShed', 'isNew', 'has2ndFloor', 'hasBsmt', 'SalePrice',
                       'LowQualFinSF',  'Fireplaces',
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','YearRemodAdd'
    #'Condition2', 'YrSold', 'MiscVal','MiscFeature',
    #'RoofMatl', 'Heating', 'KitchenAbvGr', 'PoolArea', 'PoolQC', 'MoSold','BedroomAbvGr',
    # 'Alley', 'Fence','Street','Utilities','Functional'
                          ]]
test_sel = test_sel[['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea',
                       'LotShape', 'LandContour', 'LotConfig', 'LandSlope',
                       'Neighborhood', 'Condition1', 'BldgType', 'HouseStyle', 
                       'OverallQual', 'OverallCond', 'YearBuilt', 'RoofStyle', 
                       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 
                       'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual','BsmtCond', 
                       'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 
                       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'HeatingQC',
                       'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 
                       'KitchenQual', 'TotRmsAbvGrd', 'FireplaceQu','hasFireplace',
                       'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 
                       'GarageQual', 'GarageCond', 'PavedDrive', 'SaleType', 'SaleCondition', 
                       'TotalPorchSF', 'TotalBaths', 'Age', 'hasPool', 'TotalSF', 'hasGarage',
                       'isRemodeled', 'hasShed', 'isNew', 'has2ndFloor', 'hasBsmt', 
                       'LowQualFinSF', 'Fireplaces',
                       'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                       'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath','YearRemodAdd'
                     # 'Alley', 'Fence','Street','Utilities','MiscFeature','Functional',
                     #'Condition2', 'YrSold', 'MiscVal','MiscFeature',BedroomAbvGr',
                      # 'RoofMatl', 'Heating', 'KitchenAbvGr', 'PoolArea', 'PoolQC', 'MoSold'
                          ]]                               

#-----------------------------OUTLIERS-------------------------------

train_sel = train_sel.drop(train_sel[train_sel['GrLivArea'] > 4500].index) 
train_sel = train_sel.drop(train_sel[train_sel['LotFrontage'] > 300].index) 
train_sel = train_sel.drop(train_sel[train_sel['LotArea'] > 100000].index) 
train_sel = train_sel.reset_index(drop=True)

#-----------------------SKEWNESS-----------------------------------------

#Numerical and categorical variables
num_vars = train_sel.dtypes[train_sel.dtypes != "object"].index
categ_vars = train_sel.dtypes[train_sel.dtypes == "object"].index

#Box-Cox transformation to SalePrice
lambda_value = boxcox_normmax(train_sel['SalePrice'] + 1)
train_sel['SalePrice'] = boxcox1p(train_sel['SalePrice'], lambda_value)

#exclude SalePrice
num_vars = [col for col in num_vars if col not in ['SalePrice']] 

skewness = train_sel[num_vars].apply(lambda x: abs(skew(x)))
skewed_num_vars = skewness[skewness > 0.75].index

#Box-Cox transformation
lam = 0.15
for var in skewed_num_vars:
    train_sel[var] = boxcox1p(train_sel[var], lam)
    test_sel[var] = boxcox1p(test_sel[var], lam)
    
#Add again SalePrice
num_vars = list(num_vars)
num_vars.append('SalePrice')
num_vars = pd.Index(num_vars)


#----------------------DUMMIES FOR CATEGORICAL-------------

categ_vars = train_sel.dtypes[train_sel.dtypes == "object"].index
train_dummy_vars = pd.get_dummies(train_sel[categ_vars])
test_dummy_vars = pd.get_dummies(test_sel[categ_vars])

#Drop dummies with all ones or zeros
sum_of_dummies = train_dummy_vars.sum()
all_ones = train_sel.shape[0]
cols_to_drop = sum_of_dummies[(sum_of_dummies < 1) | (sum_of_dummies == all_ones)].index
train_dummy_vars = train_dummy_vars.drop(columns=cols_to_drop)

#Same for test but checking if exist
cols_to_drop = [col_name for col_name in cols_to_drop if col_name in test_dummy_vars.columns]
test_dummy_vars.drop(columns=cols_to_drop, inplace=True)
cols_to_drop = [col_name for col_name in test_dummy_vars.columns if col_name not in train_dummy_vars.columns]
test_dummy_vars.drop(columns=cols_to_drop, inplace=True)

train_final = pd.concat([train_sel, train_dummy_vars], axis=1)
train_final = train_sel.drop(categ_vars, axis=1)
test_final = pd.concat([test_sel, test_dummy_vars], axis=1)
test_final = test_sel.drop(categ_vars, axis=1)


#------------------------RIDGE MODEL--------------------------------

y = train_final['SalePrice'].copy()             # real values SalePrice
x = train_final.drop(columns=['SalePrice']).copy()   # selected explanatory variables

#Cross-fold validation
n_folds = 10
scorer = make_scorer(mean_squared_error, greater_is_better = False)

def rmse_cv(model):
    k_folds = KFold(n_folds, shuffle=True, random_state=0).get_n_splits(x)
    rmse= (np.sqrt(-cross_val_score(model, x, y, scoring=scorer, cv = k_folds))).mean()
    return(rmse)

#Tuning alpha
ridgeCV = RidgeCV(alphas = [0.01, 0.03, 0.06, 0.1, 0.3, 0.6, 1, 3, 6, 10, 30, 60])
ridgeCV.fit(x, y)
alpha = ridgeCV.alpha_
ridgeCV = RidgeCV(alphas = [alpha * .6, alpha * .65, alpha * .7, alpha * .75, 
                          alpha * .8, alpha * .85, alpha * .9, alpha * .95, 
                          alpha, alpha * 1.05, alpha * 1.1, alpha * 1.15, 
                          alpha * 1.25, alpha * 1.3, alpha * 1.35, alpha * 1.4], 
                  cv = 10)
ridgeCV.fit(x, y)


#-------------------------PREDICTION---------------------

x_test = test_final.copy()
y_test = ridgeCV.predict(x_test)
y_test = np.trunc(inv_boxcox1p(y_test, lambda_value) - 1).astype(int)


#--------------------------SUBMISSION----------------------------------

submission = pd.concat([test['Id'], pd.Series(y_test, name='SalePrice')], axis=1)
submission.to_csv('mysubmission.csv', index=False)
