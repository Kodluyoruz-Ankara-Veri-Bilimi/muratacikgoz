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
