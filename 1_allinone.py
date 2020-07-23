###explorer:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def describe(df):

    dfinfo = pd.DataFrame(columns=['dtype','count','mean','std','min','median','max','dupe','null','zero','unique','freq','top','upout','lowout'])

    for col in df.columns:
        if df[col].dtypes != object:
            dfinfo.loc[col,'dtype'] = df[col].dtypes
            dfinfo.loc[col,'count'] =  ('%.0f' % df[col].count())  
            dfinfo.loc[col,'mean'] = df[col].mean()
            dfinfo.loc[col,'std'] = df[col].std()
            dfinfo.loc[col,'min'] = df[col].min()
            dfinfo.loc[col,'median'] = df[col].median()
            dfinfo.loc[col,'max'] = df[col].max()
            dfinfo.loc[col,'dupe'] = ('%.0f' % len(df[df.duplicated()]))
            dfinfo.loc[col,'null'] = ('%.0f' % df[col].isnull().sum())
            dfinfo.loc[col,'zero'] = ('{:.3%}'.format(df[df[col] == 0][col].count() / len(df[col])))
            dfinfo.loc[col,'unique'] = ('%.0f' % df[col].nunique())
            dfinfo.loc[col,'freq'] = ('%.0f' % df[col].value_counts().sort_values(ascending=False).values[0])
            dfinfo.loc[col,'top'] = df[col].value_counts().sort_values(ascending=False).index[0]

            Q1, Q3 = df[col].quantile(q=0.25), df[col].quantile(q=0.75)
            lowOutliers, upOutliers = Q1-1.5*(Q3-Q1), Q3+1.5*(Q3-Q1)
            dfinfo.loc[col, 'upout'] = ('{:.3%}'.format(df[df[col]>upOutliers][col].count()/len(df)))
            dfinfo.loc[col, 'lowout'] = ('{:.3%}'.format(df[df[col]<lowOutliers][col].count()/len(df)))
        
        elif df[col].dtypes == object:
            dfinfo.loc[col,'dtype'] = df[col].dtypes
            dfinfo.loc[col,'count'] =  ('%.0f' % df[col].count())  
            dfinfo.loc[col,'dupe'] = ('%.0f' % len(df[df.duplicated()]))
            dfinfo.loc[col,'null'] = df[col].isnull().sum()
            dfinfo.loc[col,'zero'] = ('{:.3%}'.format(df[df[col] == 0][col].count() / len(df[col])))
            dfinfo.loc[col,'unique'] = df[col].nunique()
            dfinfo.loc[col,'freq'] = df[col].value_counts().sort_values(ascending=False).values[0]
            dfinfo.loc[col,'top'] = df[col].value_counts().sort_values(ascending=False).index[0]

    return dfinfo


def unique(df, dtype = 'all', value = 100):

    ## dtype = 'all', 'obj'
    ## value = 100

    if dtype == 'all': 
        for col in df.columns:
            if (len(df[col].unique()) <= value):
                print(f'{col} : {df[col].unique()}')
                print(80*'-')
    
    elif dtype == 'obj':
        for col in df.columns:
            if (df[col].dtype == object) and (len(df[col].unique()) <= value):
                print(f'{col} : {df[col].unique()}')
                print(80*'-')
    return 




#def winsorized(df):
##upout_ or lowout_ winsorize

#def quantile(df):
##25,50,75

#def splitby(df):
##directly proportional



###visualizer:

def nullbar(df):
## null value percent
    for col in df.columns:
        plt.barh(col, len(df[df[col].notna()])/len(df), color = '#348ABD')
        plt.barh(col, len(df[df[col].isna()])/len(df), left = len(df[df[col].notna()])/len(df), color = '#E24A33')
    
    return plt.show()

#def outlier(df):
##minmax > boxplot

#def bartext(df):
##bar height annotate

#def multiscatter(df):
##rows and columns

#def stackbar(df):
##stacked bar plot

#def mapplot(df):
##folium map

#def stdbar(col):
##col based std and bar plot



###learner

#def regframe(X,y):
##rsq,mae,mse,rmse,mape

#def predplot(X,y):
##multi pred plot

#def confusion(X,y):
##confusion matrix, precision, recall, fscore

#def parameter(param):
##best params

#def modelframe(X,y):
##accuracy, precision, recall, f1-score, auc

#def modelcurve(X,y):
##roc and recall/precision

#def modelcoef(X,y):
##coefficient bar plot


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from statsmodels.tools.eval_measures import mse, rmse
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import statsmodels.api as sm


def perfstats(col, Y):
    pf = pd.DataFrame(columns=['model', 'rsq', 'rsq_adj', 'f_value', 'aic', 'bic', 'mae', 'mse', 'rmse', 'mape'])
    pd.options.display.float_format = '{:.3f}'.format
    for num,X in enumerate(col,1): 
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        
        standardscaler = StandardScaler()
        x_train = standardscaler.fit_transform(x_train)
        x_test = standardscaler.transform(x_test)
        
        x_train = sm.add_constant(x_train)
        results = sm.OLS(y_train, x_train).fit()
        x_test = sm.add_constant(x_test)
        y_pred = results.predict(x_test)
        pf.loc[num] = ('model_'+str(num) , results.rsquared, results.rsquared_adj, results.fvalue, results.aic, results.bic, 
                       mean_absolute_error(y_test, y_pred), mse(y_test, y_pred), rmse(y_test, y_pred), (np.mean(np.abs((y_test - y_pred) / y_test)) * 100))
    return pf

def predplts(col, Y):
    if(len(col) % 3) == 0:
        row = int(len(col) / 3)
    elif (len(col) % 3) != 0:
        row = int((len(col) // 3) +1)
    for num,X in enumerate(col,1): 
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        
        standardscaler = StandardScaler()
        x_train = standardscaler.fit_transform(x_train)
        x_test = standardscaler.transform(x_test)
        
        results = LinearRegression().fit(x_train, y_train)
        y_pred = results.predict(x_test)
        
        plt.subplot(row, 3, num)
        sns.scatterplot(x=y_test, y=y_pred)
        sns.lineplot(x=y_test, y=y_test, label='ytest')
        plt.ylabel("predict")
        plt.title('model_'+str(num))
        plt.tight_layout()
    return 

def regstats(col, Y):  #'linear',
    
    pf = pd.DataFrame(columns=['model', 'rsq_train', 'rsq_test', 'subt_rsq', 'mae_test', 'mse_test', 'rmse_test', 'mape_test']) 
    pd.options.display.float_format = '{:.3f}'.format
    
    
    for num,X in enumerate(col,1): 
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
        
        standardscaler = StandardScaler()
        x_train = standardscaler.fit_transform(x_train)
        x_test = standardscaler.transform(x_test)
        
        
        results = LinearRegression().fit(x_train, y_train)
        y_pred = results.predict(x_test)
        
        pf.loc[num] = ('model_'+str(num) ,
                       results.score(x_train, y_train),
                       results.score(x_test, y_test),
                       results.score(x_train, y_train) - results.score(x_test, y_test),
                       mean_absolute_error(y_test, y_pred), 
                       mse(y_test, y_pred), 
                       rmse(y_test, y_pred), 
                       (np.mean(np.abs((y_test - y_pred) / y_test)) * 100))
    return pf

def coefplts(X2,Y2):
    x_train, x_test, y_train, y_test = train_test_split(X2, Y2, test_size = 0.2, random_state = 42)
    
    standardscaler = StandardScaler()
    x_train = standardscaler.fit_transform(x_train)
    x_test = standardscaler.transform(x_test)
    
    model1 = LinearRegression().fit(x_train, y_train)
    dfCoef = pd.DataFrame([model1.coef_[0:]], columns=X2.columns, index=['Linear']).T
    
    ax = dfCoef.plot.bar(rot=0, figsize=(15, 4), width=0.9)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    for i, p in enumerate(ax.patches):
        ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2, p.get_height()),
                         ha='center', va='center', rotation=90, xytext=(0, 20), textcoords='offset points')
    plt.ylim(-4, 14)
    plt.tight_layout()
    plt.show()
    return
