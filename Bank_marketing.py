#!/usr/bin/env python
# coding: utf-8

# # Predict the Success of Bank Telemarketing

# In[42]:


#import library
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
get_ipython().run_line_magic('matplotlib', 'inline')


# In[821]:


df = pd.read_csv("/Users/kyle/Google Drive (NTU)/Python Learning/Bank Marketing Data/bank-additional/bank-additional-full.csv",';') 


# In[822]:


df.head()


# In[ ]:





# In[823]:


a = 1
for i in range(1,800):
    a = i*50
    print(i)
    print(df['month'][a])


# In[396]:


df


# In[224]:


range(len(df))


# In[796]:


df2 = df.copy()


# In[797]:


df2 = df.copy()
day_passed = 0
for i in range(1,len(df)):
    if df2.loc[i,'day_of_week']!= df2.loc[i-1,'day_of_week']:
        day_passed = day_passed + 1
        df2.loc[i,'date'] = day_passed
    else:
        df2.loc[i,'date'] = day_passed
    


# In[798]:


df2.loc[0,'date']=0
df2['date']=df2['date']+1
pd.DataFrame({'date':df2['date'],'day':df2['day_of_week'],'month': df2['month']})


# There in total 486 transaction days in the dataset

# In[382]:


a = pd.array(df2.groupby('date').count().age)
ax = range(0,486)
dayplot = pd.DataFrame({'day' : ax, 'count' : a})


# In[385]:


dayplot.loc[dayplot['count'] ==822]


# The telemarketing amounts for this campaign has decreased after 1 year.
# How about the success rate?

# In[397]:


df2


# In[855]:


df2['Y'] = 0
df2.Y.loc[df2['y']=='yes']=1


# In[393]:


b = pd.array(df2.groupby('date').Y.sum())
dayplot['success'] = b
dayplot['success_rate'] = dayplot['success']/dayplot['count'] 


# In[465]:


dayplot


# In[419]:


np.convolve(dayplot['success'], np.ones((5,))/5, mode='valid')


# In[420]:


mov5day = pd.DataFrame()
mov5day['count'] = np.convolve(dayplot['count'], np.ones((5,))/5, mode='valid')
mov5day['success'] = np.convolve(dayplot['success'], np.ones((5,))/5, mode='valid')
mov5day['success_rate'] = np.convolve(dayplot['success_rate'], np.ones((5,))/5, mode='valid')


# In[455]:


rows = np.shape(mov5day)[0] #number of rows
columns = np.shape(mov5day)[1] #number of columns
l = range(rows)[0::5] #indexes of each third element including the first element

new_matrix = pd.DataFrame() #Your new matrix

for i in range(len(l)):
    new_matrix[i] = mov5day.loc[l[i]] #addin


# In[457]:


df1_t = new_matrix.T


# In[740]:


df1_t['week'] = df1_t.index+1


# ### Exploratory Data Analysis

# In[739]:


fig, (ax1, ax2, ax3) = plt.subplots(nrows = 1, ncols = 3, figsize = (15,6))
sns.countplot(df['contact'], ax = ax1)
ax1.set_xlabel('Contact', fontsize = 10)
ax1.set_ylabel('Count', fontsize = 10)
ax1.set_title('Contact Counts')
ax1.tick_params(labelsize=10)

sns.countplot(df['month'], ax = ax2, order = ['may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec','mar', 'apr'])
ax2.set_xlabel('Months', fontsize = 10)
ax2.set_ylabel('')
ax2.set_title('Months Counts')
ax2.tick_params(labelsize=10)

sns.countplot(df['day_of_week'], ax = ax3)
ax3.set_xlabel('Day of Week', fontsize = 10)
ax3.set_ylabel('')
ax3.set_title('Day of Week Counts')
ax3.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.25)


# In[772]:


fig, (ax3, ax4) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,6))
sns.lineplot(ax = ax3,x = 'day', y = 'count',data = dayplot)
ax3.set_xlabel('Daily Call', fontsize = 10)
ax3.set_ylabel('')
ax3.set_title('Daily Fluctuation')
ax3.tick_params(labelsize=10)

sns.lineplot(ax = ax4,x = 'week', y = 'count',data = df1_t)
ax4.set_xlabel('Weekly Call', fontsize = 10)
ax4.set_ylabel('')
ax4.set_title('Weekly Fluctuation')
ax4.tick_params(labelsize=10)


# In[777]:


fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2, figsize = (15,6))
sns.lineplot(ax = ax1,x = 'day', y = 'success_rate',data = dayplot)
ax1.set_xlabel('Daily Call', fontsize = 10)
ax1.set_ylabel('')
ax1.set_ylim(0,1)
ax1.set_title('Daily Fluctuation')
ax1.tick_params(labelsize=10)

sns.lineplot(ax = ax2,x = 'week', y = 'success_rate',data = df1_t)
ax2.set_xlabel('Weekly Call', fontsize = 10)
ax2.set_ylabel('')
ax2.set_ylim(0,1)
ax2.set_title('Weekly Fluctuation')
ax2.tick_params(labelsize=10)


# In[743]:


df1_t.plot( x = 'week',y = 'success_rate')


# After We apply 5 day moving average, the fluctuation still exist. This imply lower frequency temporal factor such as 'month' influenced the success rate. Perhaps clients had more willingness to save money during certain months?

# In[386]:


dayplot.plot(x = 'day', y = 'count')


# In[466]:


df1_t.plot(y = 'count')


# Total call per days decrease as time went by, many the campaign was not the priority as new campaign launched.

# In[48]:


df.describe(include=['O'])


# In[26]:


df.shape


# In[52]:


df.info()


# In[53]:


pd.crosstab(index = df["y"],
                              columns="count")


# Relatively imbalanced dataset with only 10% y

# In[54]:


df.describe()


# 1. Previous shows:75% clients clients have not been contacted by bank before: First contact with bank
# 2. Most contact are finished with 5 mins
# 3. pdays stands for "Recency"(last contact), and from 1. we can know most of them would be N/A (denoted as 999)

# In[729]:


pd.crosstab(df['contact'],columns='count')


# In[737]:


contact_s = df2.groupby('contact').Y.sum()
contact_c = df2.groupby('contact').Y.count()
contact_d = pd.DataFrame({'count' : contact_c, 'success' : contact_s})
contact_d['success_rate'] = contact_d['success']/contact_d['count']


# In[738]:


contact_d


# In[736]:


sns.set(style="darkgrid")
grid = sns.FacetGrid(df, col='y',height = 5, row='contact', aspect=1.6,margin_titles=True)
grid.map(plt.hist, 'duration',color="tomato",  bins=np.linspace(0, 1000, 13))
grid.add_legend();


# In[500]:


sns.set(style="darkgrid")

g = sns.FacetGrid(df, row="y", col="contact", margin_titles=True)
bins = np.linspace(0, 1000, 13)
g.map(plt.hist, "duration", color="steelblue", bins=bins)


# While talking on cell phone lead to higher probability of success rate

# In[480]:


job_s = df2.groupby('job').Y.sum()
job_c = df2.groupby('job').Y.count()
job_d = pd.DataFrame({'count' : job_c, 'success' : job_s})
job_d['success_rate'] = job_d['success']/job_d['count']


# In[673]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(20, 8))
# Load the example car crash dataset
# Plot the total crashes
sns.set_color_codes("pastel")
g = sns.barplot(y="count", x=job_d.index, data=job_d,
            label="Total", color="b",order = job_d.sort_values('count',ascending=False).index)

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
g = sns.barplot(y="success", x=job_d.index, data=job_d,
            label="Success", color="b",order = job_d.sort_values('count',ascending=False).index)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
for i, v in enumerate(job_d.sort_values('count',ascending=False)['success_rate']):
    g.text( i-0.1, job_d.sort_values('count',ascending=False)['success'][i]+100, '{0:.0%}'.format(round(v,2)), color='black', fontweight='bold')


# Add a legend and informative axis label
ax.legend(ncol=2, loc="center right", frameon=True)
ax.set(ylim=(0, 11000), ylabel="",
    title="Total Call and Success in each Occupations")
sns.despine(left=True, bottom=True)


# Seems that students and retired workers are more likely to subscribe than other occupation

# In[501]:


# What kind of jobs clients this bank have, if you cross jobs with default, loan or housing, there is no relation
fig, ax = plt.subplots()
fig.set_size_inches(20, 8)
sns.countplot(x = 'job', data = df,order = df['job'].value_counts().index)
ax.set_xlabel('Job', fontsize=15)
ax.set_ylabel('Count', fontsize=15)
ax.set_title('Job Count Distribution ', fontsize=15)
ax.tick_params(labelsize=15)
sns.despine()


# In[679]:


month_s = df2.groupby('month').Y.sum()
month_c = df2.groupby('month').Y.count()
month_d = pd.DataFrame({'count' : month_c, 'success' : month_s})
month_d['success_rate'] = month_d['success']/month_d['count']


# In[709]:


label_m = ["may","jun","jul","aug","sep","oct","nov","dec","mar","apr"]
month_d = month_d.reindex(label_m)


# In[711]:


month_d


# In[712]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(15, 8))
# Load the example car crash dataset
# Plot the total crashes
sns.set_color_codes("pastel")
g = sns.barplot(y="count", x=month_d.index, data=month_d,
            label="Total", color="darkseagreen",order=label_m)

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
g = sns.barplot(y="success", x=month_d.index, data=month_d,
            label="Success", color="darkgreen",order=label_m)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
for i, v in enumerate(month_d['success_rate']):
    g.text( i-0.1, month_d.loc[label_m[i]]['success']+100, '{0:.0%}'.format(round(v,2)), color='black')


# Add a legend and informative axis label
ax.legend(ncol=2, loc="center right", frameon=True)
ax.set(ylim=(0, 11000), ylabel="",
    title="Total Call and Success in each Month")
sns.despine(left=True, bottom=True)


# In[674]:


week_s = df2.groupby('day_of_week').Y.sum()
week_c = df2.groupby('day_of_week').Y.count()
week_d = pd.DataFrame({'count' : week_c, 'success' : week_s})
week_d['success_rate'] = week_d['success']/week_d['count']


# In[713]:


label_w = ['mon','tue','wed','thu','fri']
week_d = week_d.reindex(label_w)


# In[714]:


week_d


# In[715]:


sns.set(style="whitegrid")

# Initialize the matplotlib figure
f, ax = plt.subplots(figsize=(20, 8))
# Load the example car crash dataset
# Plot the total crashes
sns.set_color_codes("pastel")
g = sns.barplot(y="count", x=week_d.index, data=week_d,
            label="Total", color="b",order = label_w)

# Plot the crashes where alcohol was involved
sns.set_color_codes("muted")
g = sns.barplot(y="success", x=week_d.index, data=week_d,
            label="Success", color="b",order=label_w)
g.set_xticklabels(g.get_xticklabels(), rotation=45)
for i, v in enumerate(week_d['success_rate']):
    g.text( i-0.1, week_d.loc[label_w[i]]['success']+100, '{0:.0%}'.format(round(v,2)), color='black', fontweight='bold')


# Add a legend and informative axis label
ax.legend(ncol=2, loc="center right", frameon=True)
ax.set(ylim=(0, 11000), ylabel="",
    title="Total Call and Success in each Weekday")
sns.despine(left=True, bottom=True)


# In[779]:


sns.set(style="ticks")

sns.pairplot(df2, hue="y", palette="Set1")
plt.show()


# In[784]:


ho_s = df2.groupby('housing').Y.sum()
ho_c = df2.groupby('housing').Y.count()
ho_d = pd.DataFrame({'count' : ho_c, 'success' : ho_s})
ho_d['success_rate'] = ho_d['success']/ho_d['count']
ho_d


# In[786]:


def_s = df2.groupby('default').Y.sum()
def_c = df2.groupby('default').Y.count()
def_d = pd.DataFrame({'count' : def_c, 'success' : def_s})
def_d['success_rate'] = def_d['success']/def_d['count']
def_d


# In[785]:


lo_s = df2.groupby('loan').Y.sum()
lo_c = df2.groupby('loan').Y.count()
lo_d = pd.DataFrame({'count' : lo_c, 'success' : lo_s})
lo_d['success_rate'] = lo_d['success']/lo_d['count']
lo_d


# In[781]:


df["housing"].value_counts()/len(df)


# In[782]:


df["loan"].value_counts()/len(df)


# In[ ]:


# grid = sns.FacetGrid(train_df, col='Embarked')
grid = sns.FacetGrid(df2, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Y', 'Sex', palette='deep')
grid.add_legend()


# In[ ]:





# ### Model Preparation

# In[800]:


df3 = df2.copy()


# In[825]:


df3.head()


# In[804]:


df3 = pd.get_dummies(data = df3, columns = ['job'] , prefix = ['job'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['marital'] , prefix = ['marital'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['education'], prefix = ['education'], drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['default'] , prefix = ['default'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['housing'] , prefix = ['housing'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['loan'] , prefix = ['loan'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['contact'] , prefix = ['contact'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['month'] , prefix = ['month'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['day_of_week'] , prefix = ['day_of_week'] , drop_first = True)
df3 = pd.get_dummies(data = df3, columns = ['poutcome'] , prefix = ['poutcome'] , drop_first = True)


# In[788]:


numeric_data = df
label = LabelEncoder()
dicts = {}

X = [
                   'age', 'job', 'marital',
                   'education', 'default', 'housing',
                   'loan','contact',
                   'month','day_of_week','duration', 'campaign',
                   'pdays','previous',
                   'poutcome', 'emp.var.rate',
                   'cons.price.idx', 'cons.conf.idx',
                   'euribor3m', 'nr.employed'
]

fields = X
fields.append('y')

for f in fields:
    label.fit(df[f].drop_duplicates())
    dicts[f] = list(label.classes_)
    numeric_data[f] = label.transform(df[f])    

target = numeric_data['y']
numeric_data = numeric_data.drop(['y'], axis=1)     

# Looking for most valuable columns in our dataset
# k-value affect auc final score and roc curve
numeric_data_best = SelectKBest(f_classif, k=7).fit_transform(numeric_data, target)

#looking for null data
df.isnull().sum()


# In[809]:


df3.head()


# In[839]:





# In[818]:


np.where(df2['date']==200)


# 30429 make it a whole year for training from May-2008 to April-2009

# In[841]:


df_train = df3.iloc[0:30429]


# In[842]:


df_test = df3.iloc[30429:]


# In[860]:


df_test.y


# In[895]:


X_Train = df_train.drop("y", axis=1)
Y_Train = df_train["y"]
X_predict  = df_test.copy().drop('y', axis=1)
Y_predict = df_test['y']
X_Train.shape, Y_Train.shape, X_predict.shape, Y_predict.shape


# In[862]:


X_train, X_test, y_train, y_test = train_test_split(X_Train,Y_Train,test_size = 0.2,random_state = 123)


# ### Model and Comparison

# In[870]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


# #### Logistics Regression

# In[864]:


# Create the hyperparameter grid
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space, 'penalty': ['l1', 'l2']}

# Instantiate the logistic regression classifier: logreg
logreg = LogisticRegression()

# Create train and test sets

# Instantiate the GridSearchCV object: logreg_cv
logreg_cv = GridSearchCV(logreg,param_grid,cv = 5)

# Fit it to the training data
logreg_cv.fit(X_train,y_train)


# In[876]:


y_pred = logreg_cv.predict(X_test)

# Compute and print metrics
print("Accuracy: {}".format(logreg_cv.score(X_test, y_test)))
print(classification_report(y_test, y_pred))


# Print the optimal parameters and best score
print("Tuned Logistic Regression Parameter: {}".format(logreg_cv.best_params_))
print("Tuned Logistic Regression Accuracy: {}".format(logreg_cv.best_score_))


# In[896]:


logreg = LogisticRegression(**logreg_cv.best_params_)
logreg.fit(X_Train,Y_Train)
logreg_Y_pred = logreg.predict(X_predict)
logreg_Y_pred


# In[878]:


from sklearn.feature_selection import f_regression
logreg = LogisticRegression(**logreg_cv.best_params_)

# Fit it to the training data
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print(logreg.score(X_test,y_test))
#Coefficient report
coeff_df = pd.DataFrame(df_train.columns.delete(0))
coeff_df.columns = ['Feature']
coeff_df["Correlation"] = pd.Series(logreg.coef_[0])
coeff_df["P-value"] = pd.Series(f_regression(X_train, y_train)[1])
coeff_df.sort_values(by='P-value', ascending=True)


# #### Decision Tree

# In[867]:


from scipy.stats import randint


# In[879]:


# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree,param_dist , cv=5)

# Fit it to the data
tree_cv.fit(X_train,y_train)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
print("Test Accuracy is {}".format(tree_cv.score(X_test,y_test)))
acc_decision_tree = tree_cv.score(X_test,y_test)


# In[897]:


tree = DecisionTreeClassifier(**tree_cv.best_params_)
tree.fit(X_Train,Y_Train)
tree_Y_pred = tree.predict(X_predict)
tree_Y_pred


# In[904]:


import graphviz 
dot_data = tree.export_graphviz(tree, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("X_Train") 


# In[ ]:


dot_data = tree.export_graphviz(clf, out_file=None, 
                      feature_names=X_Train.feature_names,  
                      class_names=X_Train.target_names,  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 


# #### KNN

# In[901]:


#Tuning parameter K
# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 25)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train,y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)


# In[902]:


plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()


# In[905]:


n = 10
knn = KNeighborsClassifier(n_neighbors= n)


# In[906]:


# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Predict the labels of the test data: y_pred
y_pred = knn.predict(X_test)

# Generate the confusion matrix and classification report
print("Accuracy: {}".format(knn.score(X_test, y_test)))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("KNN Neighbors = {}".format(n))
acc_knn = knn.score(X_test, y_test)


# In[907]:


knn.fit(X_Train,Y_Train)
knn_Y_pred = knn.predict(X_predict)
knn_Y_pred


# In[ ]:





# In[16]:


# comparing best model
model_lr = LogisticRegression(penalty='l1', tol=0.01) 
model_dt = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
model_svc = svm.SVC() 
model_svc = SVC(kernel='rbf', random_state=0)
model_bnn = MLPClassifier()

ROCtrainTRN, ROCtestTRN, ROCtrainTRG, ROCtestTRG = model_selection.train_test_split(numeric_data_best, target, test_size=0.3) 

results = {}
kfold = 10

results['LogisticRegression_best_params'] = model_selection.cross_val_score(model_lr, numeric_data_best, target, cv = kfold).mean()
results['DecisionTree_best_params'] = model_selection.cross_val_score(model_dt, numeric_data_best, target, cv = kfold).mean()
results['SVC_best_params'] = model_selection.cross_val_score(model_svc, numeric_data_best, target, cv = kfold).mean()
results['NN_best_params'] = model_selection.cross_val_score(model_bnn, numeric_data_best, target, cv = kfold).mean()

results['LogisticRegression_all_params'] = model_selection.cross_val_score(model_lr, numeric_data, target, cv = kfold).mean()
results['DecisionTree_all_params'] = model_selection.cross_val_score(model_dt, numeric_data, target, cv = kfold).mean()
results['SVC_all_params'] = model_selection.cross_val_score(model_svc, numeric_data, target, cv = kfold).mean()
results['NN_all_params'] = model_selection.cross_val_score(model_bnn, numeric_data, target, cv = kfold).mean()    


# In[17]:


# ROC with all parameters
roc_train_all, roc_test_all, roc_train_all_class, roc_test_all_class = model_selection.train_test_split(numeric_data, target, test_size=0.25) 
roc_train_best, roc_test_best, roc_train_best_class, roc_test_best_class = model_selection.train_test_split(numeric_data_best, target, test_size=0.25) 

models = [
    {
        'label' : 'SVC_all_params',
        'model': model_svc,
        'roc_train': roc_train_all,
        'roc_test': roc_test_all,
        'roc_train_class': roc_train_all_class,        
        'roc_test_class': roc_test_all_class,        
    },        
    {
        'label' : 'LogisticRegression_all_params',
        'model': model_lr,
        'roc_train': roc_train_all,
        'roc_test': roc_test_all,
        'roc_train_class': roc_train_all_class,        
        'roc_test_class': roc_test_all_class,        
    },
    {
        'label' : 'DecisionTree_all_params',
        'model': model_dt,
        'roc_train': roc_train_all,
        'roc_test': roc_test_all,
        'roc_train_class': roc_train_all_class,        
        'roc_test_class': roc_test_all_class,        
    },
    {
        'label' : 'NN_all_params',
        'model': model_bnn,
        'roc_train': roc_train_all,
        'roc_test': roc_test_all,
        'roc_train_class': roc_train_all_class,        
        'roc_test_class': roc_test_all_class,        
    }
]


plt.clf()
plt.figure(figsize=(8,6))

for m in models:
    m['model'].probability = True
    probas = m['model'].fit(m['roc_train'], m['roc_train_class']).predict_proba(m['roc_test'])
    fpr, tpr, thresholds = roc_curve(m['roc_test_class'], probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()


# In[18]:


# ROC with best parameters
roc_train_all, roc_test_all, roc_train_all_class, roc_test_all_class = model_selection.train_test_split(numeric_data, target, test_size=0.25) 
roc_train_best, roc_test_best, roc_train_best_class, roc_test_best_class = model_selection.train_test_split(numeric_data_best, target, test_size=0.25) 

models = [
    {
        'label' : 'SVC_best_params',
        'model': model_svc,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },        
    {
        'label' : 'LogisticRegression_best_params',
        'model': model_lr,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },
    {
        'label' : 'DecisionTree_best_params',
        'model': model_dt,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    },
    {
        'label' : 'NN_best_params',
        'model': model_bnn,
        'roc_train': roc_train_best,
        'roc_test': roc_test_best,
        'roc_train_class': roc_train_best_class,        
        'roc_test_class': roc_test_best_class,        
    }
]


plt.clf()
plt.figure(figsize=(8,6))

for m in models:
    m['model'].probability = True
    probas = m['model'].fit(m['roc_train'], m['roc_train_class']).predict_proba(m['roc_test'])
    fpr, tpr, thresholds = roc_curve(m['roc_test_class'], probas[:, 1])
    roc_auc  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (m['label'], roc_auc))


plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc=0, fontsize='small')
plt.show()


# In[19]:


xgb_params = {
    'eta': 0.05,
    'max_depth': 8,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(numeric_data,target,feature_names = numeric_data.columns.values)
model = xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=100)

fig,ax=plt.subplots(figsize = (13,19))
xgb.plot_importance(model, max_num_features=50, height=0.5, ax=ax)
plt.show()


# In[ ]:




