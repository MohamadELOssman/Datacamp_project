import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np
import missingno as msno
import seaborn as sb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans


problem_title = 'In hospital mortality prediction'

_prediction_label_names = [0, 1]

Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)

workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name='acc'),
    rw.score_types.RMSE(name='rmse'),
    rw.score_types.F1Above(name='f1_score')
]

# score_type_1 = rw.score_types.ClassificationError(name='err', precision=3)
# score_type_2 = rw.score_types.MARE(name='mare', precision=3)
# score_types = [
#     # The official score combines the two scores with weights 2/3 and 1/3.
#     rw.score_types.Combined(
#         name='combined', score_types=[score_type_1, score_type_2],
#         weights=[2. / 3, 1. / 3], precision=3),
# ]

def get_cv(X, y):
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    return cv.split(X, y)
    
_target_column_name = 'outcome'

def preprocess_data(df):
    for column_name in df.columns:
        column_name_without_spaces=column_name.strip()
        if column_name_without_spaces != column_name :
            df.columns = df.columns.str.replace(column_name, column_name_without_spaces)


    list_to_drop=['group','ID']
    df = df.drop(list_to_drop, axis=1)

    """## **Preprocessing**

    ### **Dropping columns of the dataset with high NULL value count**
    """

    threshold=0.23
    list_to_drop = []

    for column in df.columns:
        if df[column].isnull().sum() >= (threshold)*df.shape[0]:
            list_to_drop.append(df[column].name)

    df = df.drop(list_to_drop, axis=1)

    """### **Split features between binary and numeric categories**"""

    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    numAttr=[]
    binaryAttr=[]
    for key in dtypes:
        uniqueValues = df[key].dropna().unique()
        if(key=="outcome"):
            binaryAttr.append(key)
        elif((dtypes[key] == 'int64' or dtypes[key] == 'float64') and len(uniqueValues)==2):
            binaryAttr.append(key)
        else:
            numAttr.append(key)

    """### **Identifying outliers**

    """

    df.isnull().values.sum()

    """#### Standard Deviation Method"""

    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()


    i=0

    for key in dtypes:
        if(key in numAttr ):
            i+=1
            
            data_mean, data_std = np.mean(df[key]), np.std(df[key])
            cut_off = data_std * 4
            lower, upper = data_mean - cut_off, data_mean + cut_off
            outliers = [x for x in df[key] if x < lower or x > upper]
            outliers_removed = [x for x in df[key] if x > lower and x < upper]

    """#### Interquartile Range Method"""

    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    tot_out_indices=[]
    i=0
    for key in dtypes:
        if(key in numAttr):
            i+=1
            q5, q95 = np.percentile(df[key].dropna(), 2), np.percentile(df[key].dropna(), 98)
            iqr = q95 - q5
            # calculate the outlier cutoff
            cut_off = iqr * 1.5
            lower, upper = q95 - cut_off, q5 + cut_off
            outliers = [x for x in df[key] if (x < lower or x > upper) and (x!=None)]
            out_indices = [ind for ind in range(df.shape[0]) if df.loc[ind,key] < lower or df.loc[ind,key] > upper]
            tot_out_indices.extend(out_indices)
            outliers_removed = [x for x in df[key] if x > lower and x < upper]

    tot_out_indices = np.unique(tot_out_indices)

    df.drop(tot_out_indices, axis=0,inplace=True)

    """### **Correlation Study**

    """

    uncorrelatedAtt={}
    min_corr=0.26
    for att in df.columns:
        uncorrelatedAtt[att]=True

    max_corr=0.75
    corratt={}

    """#### **Univariate Correlation between numerical-input numerical-output**

    ##### **Pearson method for linear relation**
    """

    a=np.abs(df[numAttr].corr())

    a=np.abs(df[numAttr].corr())
    corratt={}
    imtnt_att={}

    for att in numAttr:
        k = a.loc[att].sort_values(ascending=False)
        imtnt_att[att] = []
        if k[1]>min_corr:
            uncorrelatedAtt[att] = False
        for p,b in k.items():
            if b<1:
                imtnt_att[att].append((p,b))
                if( (p,att) not in corratt.keys() and b>max_corr):
                    corratt[(att,p)] = b

    df_corr = df[numAttr]
    corr = df_corr.corr()

    """##### **Spearman method for nonlinear relation**"""

    a=np.abs(df[numAttr].corr(method='spearman'))

    for att in numAttr:
        k= a.loc[att].sort_values(ascending=False)[1:6]
        if k[0]>min_corr:
            uncorrelatedAtt[att] = False
        k = k[k>max_corr]

        if not k.empty :
            for p,b in k.items():
                if( (p,att) not in corratt.keys()):
                    corratt[(att,p)] = b

    df_corr = df[numAttr]
    corr = df_corr.corr()

    """#### **Univariate Correlation between categorical-input categorical-output**"""

    a=np.abs(df[binaryAttr].corr())

    for att in binaryAttr:
        k= a.loc[att].sort_values(ascending=False)[1:6]
        if k[0]>min_corr:
            uncorrelatedAtt[att] = False
        k = k[k>max_corr]
        if not k.empty :
            for p,b in k.items():
                if( (p,att) not in corratt.keys()):
                    corratt[(att,p)] = b

    df_corr = df[binaryAttr]
    corr = df_corr.corr()

    """#### **Univariate Correlation between numerical-input categorical-output**"""

    corr = df.corr(method='kendall')
    corr = np.abs(corr.loc[binaryAttr,numAttr])

    dic={}
    for att in numAttr:
        for batt in binaryAttr:
            k= corr.loc[batt,att]
            if k<min_corr and batt=='classification':
               dic[(batt,att)] = k

        for att in numAttr:
            for batt in binaryAttr:
                k= corr.loc[batt,att]
                if k>max_corr:
                    corratt[(batt,att)] = k

    corr = df.corr(method='kendall')

    corr = corr.loc[binaryAttr,numAttr]

    uncorrelatedAttr = pd.DataFrame.from_dict(uncorrelatedAtt,orient='index').drop('outcome')
    uncorrelatedAt = uncorrelatedAttr[uncorrelatedAttr[0]==True].index
    for att in uncorrelatedAt:
        if att in binaryAttr:
            if corr.loc[att].sort_values(ascending=False)[0] > min_corr:
                uncorrelatedAt.drop(att)
        else:
            if np.abs(corr[att].sort_values(ascending=False)[0]) > min_corr:
                uncorrelatedAt.drop(att)

    """### **Feature Extraction**

    We have studied in the section above the correlation between attributes. Using correlaion we can reduce nombre of variables by:

    - remove the atrributes havn't any correlation with others
    - leave one of correlated attributes that have a high correlation and remove others

    #### **Uncorrelated attribute**
    """

    df.drop(uncorrelatedAt,axis=1,inplace=True)

    for att in uncorrelatedAt:
        if att in numAttr:
            numAttr.remove(att)
        elif att in binaryAttr:
            binaryAttr.remove(att)

    """Using pearson correlation we can see that we have a linear relation between:
    - PT and INR
    - hematocrit and RBC
    - MCH and MCV
    - Neutrophils and Lymphocyte

    Thus we can this relation to get the missing values.
    """

    p1_1 = df[['PT','INR']].loc[df['PT'].isnull()].dropna(subset=['INR'], how='all') # missing values in PT can be prredicted using values of INR
    p1_2 = df[['PT','INR']].loc[df['INR'].isnull()].dropna(subset=['PT'], how='all') # missing values in INR can be prredicted using values of PT
    p2_1 = df[['hematocrit','RBC']].loc[df['hematocrit'].isnull()].dropna(subset=['RBC'], how='all') # missing values in hematocrit can be prredicted using values of RBC
    p2_2 = df[['hematocrit','RBC']].loc[df['RBC'].isnull()].dropna(subset=['hematocrit'], how='all') # missing values in RBC can be prredicted using values of hematocrit
    p3_1 = df[['MCH','MCV']].loc[df['MCH'].isnull()].dropna(subset=['MCV'], how='all') # missing values in MCH can be prredicted using values of MCV
    p3_2 = df[['MCH','MCV']].loc[df['MCV'].isnull()].dropna(subset=['MCH'], how='all') # missing values in MCV can be prredicted using values of MCH
    p4_1 = df[['Neutrophils','Lymphocyte']].loc[df['Neutrophils'].isnull()].dropna(subset=['Lymphocyte'], how='all') # missing values in Neutrophils can be prredicted using values of Lymphocyte
    p4_2 = df[['Neutrophils','Lymphocyte']].loc[df['Lymphocyte'].isnull()].dropna(subset=['Neutrophils'], how='all') # missing values in Lymphocyte can be prredicted using values of Neutrophils

    """### **Missing values**

    #### Representing missing values
    """

    """#### Applying different methods to handle missing values problem

    ##### 1. Delete Rows with Missing Values:
    """

    df_DM = df.dropna()

    """**Using this method the data becomes much smaller with 158 rows, so it's better to use another approach to handle our concern.**

    ##### 2. Impute missing values with Mean/Median/Mode (Numerical and categorical values):
    Using this method could efficient when number of missing values is small, but having a large number could affect our prediction; hence we will impute missing values with mean/median/mode when the number is less than 10.
    """

    threshold=0.05
    dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
    objAttr=[]
    binaryObjects=[]
    naryObjects=[]
    for key in dtypes:
        if(key in numAttr and df.isnull().sum()[key] < threshold*df.shape[0]):
            df[key] = df[key].replace(np.NaN,  df[key].mean())  # fill missing values with the mean or median
        elif(key in binaryAttr and df.isnull().sum()[key] < threshold*df.shape[0]):
            df[key] = df[key].fillna(df[key].mode()[0])  # fill categorical missing values with the most frequent category

    """##### 3. For the remaining missing values we will use the impute method from the sklearn library."""

    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df),columns = df.columns)

    """### **Features Scaling**

    #### 1. Standardization
    """

    sc=StandardScaler()
    df_S = df.copy()
    dtypes = df_S.dtypes.apply(lambda x: x.name).to_dict()
    df_S[numAttr] = sc.fit_transform(df_S[numAttr])

    df_S.mean()
    df=df_S.copy()

    """#### 2. Min-Max Scaling:"""

    sc = MinMaxScaler()
    df_M = df.copy()

    dtypes = df_M.dtypes.apply(lambda x: x.name).to_dict()
    df_M[numAttr] = sc.fit_transform(df_M[numAttr])

    """ ###  **Data Balancing : SMOTE Method**

    """

    df_balanced_skl=df_S

    labels = df_balanced_skl.outcome.astype(int)
    features = df_balanced_skl.drop('outcome', axis=1)
    # setting up testing and training sets
    sm = SMOTE(random_state=42)
    features, labels = sm.fit_resample(features, labels)
    
    features_train_sm, features_test_sm, labels_train_sm, labels_test_sm = train_test_split(features, labels, test_size=0.25, random_state=42)
    
    return features_train_sm, labels_train_sm, features_test_sm, labels_test_sm

class ReadData:
    preprocessed_data = None
    def __init__(self):
        pass
    
    @staticmethod
    def read_data(path, f_name):
        if ReadData.preprocessed_data is None:
            data = pd.read_csv(os.path.join(path, 'data', f_name))
            X_train, y_train, X_test, y_test = preprocess_data(data) #
            ReadData.preprocessed_data = (X_train.to_numpy(), y_train.to_numpy()), (X_test.to_numpy(), y_test.to_numpy())
        return ReadData.preprocessed_data    

def get_train_data(path='.'):
    f_name = 'data.csv'
    return ReadData.read_data(path, f_name)[0]


def get_test_data(path='.'):
    f_name = 'data.csv'
    return ReadData.read_data(path, f_name)[1]

