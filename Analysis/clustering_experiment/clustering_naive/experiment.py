import numpy as np
import pandas as pd

from sklearn.cluster import *
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import normalize


def score_clustering_algo(algo, variable_subset, imputation_method, measures, df):
    try:
        subset_df = df[variable_subset]
        
        imputed_df = preprocess(imputation_method=imputation_method, df=subset_df)

        labels = algo.fit_predict(imputed_df)
        scores = np.zeros((len(measures)))
        for measure_idx, measure in enumerate(measures):
            if len(np.unique(labels)) > 1:
                score = measure(X=imputed_df, labels=labels)
                scores[measure_idx] = score
            else:
                scores[measure_idx] = -1
        return (scores, algo)
    except Exception as e:
        print(e)



def preprocess(imputation_method, df, random_seed=0):

    #num_nan = df.isna().sum().sum()
    #num_non_nan = df.count().sum()
    #print("{} NaN values out of {} total elements in df".format(num_nan, num_non_nan))
    columns = df.columns

    string_df = df.select_dtypes('object')
    numerical_df = df.select_dtypes('number')
    
    # applicable to all columns
    if imputation_method == 'iterative':
        df =  IterativeImputer(
            estimator=BayesianRidge(),
            random_state=random_seed,
            max_iter=20,
        ).fit_transform(df)
        df = normalize(df)
        return df
    elif imputation_method == 'KNN':
        df = normalize(df)
        df = KNNImputer(
            n_neighbors=5,
            weights='distance'
        ).fit_transform(df)
        return df
    elif imputation_method == 'mode':
        df.apply(lambda series: series.fillna(series.mode(), inplate = True))
        df = normalize(df)
        return df

    # apply method to numerical and most frequent on string columns
    elif imputation_method == 'mean':
        numerical_df.apply(lambda series: series.fillna(series.mean(), inplace = True))
    elif imputation_method == 'median':
        numerical_df.apply(lambda series: series.fillna(series.median(), inplace = True))    
    numerical_df = normalize(numerical_df)

    string_df.apply(lambda series: series.fillna(series.mode(), inplate = True))
    df = np.concatenate((numerical_df, string_df), axis=1)

    df = pd.DataFrame(df, columns=columns)

    return df