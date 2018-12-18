# --------------
import pandas as pd
from sklearn import preprocessing

#path : File path

# Code starts here


# read the dataset
dataset=pd.read_csv(path)

# look at the first five columns
dataset.head()

# Check if there's any column which is not useful and remove it like the column id
dataset.drop(columns='Id',inplace=True)

# check the statistical description
dataset.describe()



# --------------
# We will visualize all the attributes using Violin Plot - a combination of box and density plots
import seaborn as sns
from matplotlib import pyplot as plt

#names of all the attributes 
cols=dataset.columns
#number of attributes (exclude target)
size=dataset.iloc[:,0:54].shape

#x-axis has target attribute to distinguish between classes
x=dataset['Cover_Type'].to_string()

#y-axis shows values of an attribute
y=dataset.iloc[:,0:54]

#Plot violin for all attributes
ax = sns.violinplot(size,data=dataset)





# --------------
import numpy
threshold = 0.5

# no. of features considered after ignoring categorical variables

num_features = 10

# create a subset of dataframe with only 'num_features'
subset_train=dataset.iloc[:,0:10]
cols=subset_train.columns

#Calculate the pearson co-efficient for all possible combinations
data_corr=subset_train.corr(method='pearson')
sns.heatmap(data_corr)


# Set the threshold and search for pairs which are having correlation level above threshold
corr_var_list=data_corr[(data_corr > threshold) & (data_corr > threshold) & (data_corr !=1)]
corr_var_list.dropna(how='all',inplace=True)
print(corr_var_list)
# Sort the list showing higher ones first 
def get_reduntant_pairs(df):
    pairs_to_drop=set()
    cols=df.columns
    for i in range(0,df.shape[1]):
        for j in range(0,i+1):
            pairs_to_drop.add((cols[i],cols[j]))
        return pairs_to_drop

def get_top_abs_correlations(df):
    au_corr=df.abs().unstack()
    labels_to_drop=get_reduntant_pairs(df)
    au_corr=au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:]

s_corr_list=get_top_abs_correlations(data_corr)
print(s_corr_list)

#Print correlations and column names




# --------------
#Import libraries 
from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# Identify the unnecessary columns and remove it 
dataset.drop(columns=['Soil_Type7', 'Soil_Type15'], inplace=True)

X=dataset.iloc[:,:-1]
y=dataset.iloc[:,-1]

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

num_feat_cols = ['Elevation', 'Aspect', 'Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways','Hillshade_9am','Hillshade_Noon', 'Hillshade_3pm','Horizontal_Distance_To_Fire_Points']

cat_feat_cols = list(set(X_train.columns) - set(num_feat_cols))
scaler=StandardScaler()

X_train_temp=X_train[num_feat_cols].copy()

X_test_temp=X_test[num_feat_cols].copy()

X_train_temp[num_feat_cols]=scaler.fit_transform(X_train_temp[num_feat_cols])

X_test_temp[num_feat_cols]=scaler.fit_transform(X_test_temp[num_feat_cols])


X_train1=pd.concat([X_train_temp,X_train.loc[:,cat_feat_cols]],axis=1)

print(X_train1.head())

X_test1=pd.concat([X_test_temp,X_test.loc[:,cat_feat_cols]],axis=1)

print(X_test1.head())

scaled_features_train_df=X_train1
#Standardized
#Apply transform only for non-categorical data
scaled_features_test_df=X_test1

#Concatenate non-categorical data and categorical



# --------------
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import f_classif


# Write your solution here:
skb=SelectPercentile(f_classif,percentile=20)
predictors=skb.fit_transform(X_train1,y_train)

scores=list(predictors)
top_k_index  = skb.get_support(indices=True)
top_k_predictors= predictors[top_k_index]
print(top_k_predictors)
print(top_k_index)
print(scores)



# --------------
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score

clf = OneVsRestClassifier(LogisticRegression())
clf1=OneVsRestClassifier(LogisticRegression())
model_fit_all_features =clf.fit(X_train , y_train)
predictions_all_features=clf.predict(X_test)

score_all_features= accuracy_score(y_test,predictions_all_features )

print(scaled_features_train_df.columns[skb.get_support()])

X_new = scaled_features_train_df.loc[:,skb.get_support()]
X_test_new=scaled_features_test_df.loc[:,skb.get_support()]

model_fit_top_features  =clf1.fit(X_new , y_train)
predictions_top_features=clf1.predict(X_test_new)

score_top_features= accuracy_score(y_test,predictions_top_features )




