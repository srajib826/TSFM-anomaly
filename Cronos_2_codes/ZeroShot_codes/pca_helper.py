import sys
sys.path.append('/home/rajib/TSB-AD')
import numpy as np
from TSB_AD.models.PCA import PCA
import pandas as pd
from TSB_AD.evaluation.metrics import get_metrics
import glob
from tqdm import tqdm  
import os
import warnings
warnings.filterwarnings("ignore") 


def split_df(df, split_ratio):
    split_index = int(len(df) * split_ratio)
    return df[:split_index], df[split_index:]

def process_file(file, split_ratio):
    df = pd.read_csv(file).dropna()
    X_train, X_test = split_df(df, split_ratio)
    labels = X_test['is_anomaly'].astype(int).to_numpy()
    X_test.drop(columns=['is_anomaly','timestamp'], inplace=True)
    X_train.drop(columns=['is_anomaly','timestamp'], inplace=True)
    return X_train, X_test, labels


def anomaly_PCA(file, split_ratio):
    X_train, X_test, labels = process_file(file, split_ratio)

    detector = PCA(
    slidingWindow=10,
    sub=True,
    n_components=10,
    n_selected_components=10, #Number of selected principal components
    contamination=0.05,
    copy=True,
    whiten=False,
    svd_solver='auto',
    tol=0.0,
    iterated_power='auto',
    random_state=42,
    weighted=True,
    standardization=True,
    zero_pruning=True,
    normalize=True
    )
    
    detector.fit(X_train)
    anomaly_score= detector.decision_function(X_test)
    evaluation_result = get_metrics(anomaly_score, labels)
    return anomaly_score,evaluation_result