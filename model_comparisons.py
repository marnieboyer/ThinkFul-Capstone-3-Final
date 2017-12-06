
# coding: utf-8

# In[2]:

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import ensemble
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.decomposition import PCA


# # Set up Test/Train for Clustering

# In[3]:

## IMPORT latest dataset:

data = pd.read_csv('all_features5.csv',index_col = None)
data = data.drop('Unnamed: 0',axis = 1)
data.shape


# In[4]:

data_clean = data.dropna(axis=0, how='any')
#data_clean = data

data_clean.shape


# In[5]:

#X = data_clean.iloc[:,145:171]
X = data_clean

X = X.drop(['filenum','filename','classified_shape'] , axis = 1)

Y = data_clean['classified_shape']
X.shape


# In[6]:

SMALL_X = data_clean.drop(['0','1','2','3','4','5','6','7','8','9','10','11',	'12',	'13',	'14',	'15',	'16','17',
                             '18',	'19',	'20',	'21',	'22',	'23',	'24','25',	'26',	'27',	'28',	'29',
                             '30',	'31',	'32',	'33',	'34',	'35',	'36',	'37',	'38',	'39',	'40',	'41',
                             '42',	'43',	'44',	'45',	'46',	'47',	'48',	'49',	'50',	'51',	'52',	'53',
                             '54',	'55',	'56',	'57',	'58',	'59',	'60',	'61',	'62',	'63',	'64',	'65',
                             '66',	'67',	'68',	'69',	'70',	'71',	'72',	'73',	'74',	'75',	'76',	'77',
                             '78',	'79',	'80',	'81',	'82',	'83',	'84',	'85',	'86',	'87',	'88',	'89',
                             '90',	'91',	'92',	'93',	'94',	'95',	'96',	'97',	'98',	'99',	'100',	'101',
                             '102',	'103',	'104',	'105',	'106',	'107',	'108',	'109',	'110',	'111',	'112',	'113',
                             '114',	'115',	'116',	'117',	'118',	'119',	'120',	'121',	'122',	'123',	'124',	'125',
                             '126',	'127',	'128',	'129',	'130',	'131',	'132',	'133',	'134',	'135',	'136',	'137',
                             '138',	'139',	'140',	'141',	'142',	'143'
                             ,'A1','A2','A3','A4','A5','A6','A7','A8'
                             ,'A9','A10','A11','A12','A13','A14','A15'
                            ,'Height','Width'
                            ,'MJ_width','Jaw_width'
                            #'H_W_Ratio','J_F_Ratio','MJ_J_width'
                               ],axis = 1)
corrmat = SMALL_X.corr()

# Set up the matplotlib figure.
f, ax = plt.subplots(figsize=(10, 10))

# Draw the heatmap using seaborn
sns.heatmap(corrmat,vmin= -1, vmax=1, square=True)
plt.show()


# # Supervised Learning

# In[7]:

# Standardize features by removing the mean and scaling to unit variance

scaler = StandardScaler()  
scaler.fit(X)  

X = scaler.transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X,Y,
    test_size=0.25,
    random_state=None)


# ### Use PCA for dimension reduction

# In[8]:

n_components = 18
pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X)


print(
    'The percentage of total variance in the dataset explained by each',
    'component from Sklearn PCA.\n',
    pca.explained_variance_ratio_
)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)


# In[9]:

# #Remove PCA 

X_train_pca = X_train
X_test_pca = X_test


# ## Neural Network (MLP)

# In[10]:

mlp = MLPClassifier(hidden_layer_sizes=(60,10,60,30), solver='sgd',shuffle  = True, 
                    learning_rate_init=0.01, max_iter = 1000,warm_start  = False)
#reducing the learning rate init allowed the MLP to converge 
mlp.fit(X_train_pca, Y_train)
mlp.score(X_train_pca, Y_train)


# In[11]:

print(mlp.score(X_train_pca,Y_train))
mlp_score = mlp.score(X_test_pca,Y_test)
print(mlp_score)

y_pred = mlp.predict(X_test_pca)
 
mlp_crosstab = pd.crosstab(Y_test, y_pred, margins=True)
mlp_crosstab


# In[12]:

from sklearn.model_selection import cross_val_score
cross_val_score(mlp, X, Y, cv=5)


# In[13]:

print(classification_report(Y_test,y_pred))


# In[14]:

# Get the RECALL for each shape and overall
correct_list =[]
shape_list = []
for i in mlp_crosstab.index[0:5]:
    correct = (mlp_crosstab.at[i,i]/mlp_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    shape_list.append(i)
    correct_list.append(correct)

shape_list.append('Overall')
correct_list.append(round(mlp_score,2)*100)
results_df = pd.DataFrame()
results_df['shape']= shape_list
results_df['MLP']=correct_list


# ## KNN Classifier

# In[56]:

#neigh = KNeighborsClassifier(n_neighbors=9,weights='distance')

#determined 9 was best through experimentation, weighting by distance led to overfitting

nn = []
score = []
cv_scores = []
neighbors = range(1,30)
for n in neighbors:
    neigh = KNeighborsClassifier(n_neighbors=n) 
    neigh.fit(X_train_pca, Y_train) 
    sc = neigh.score(X_test_pca,Y_test)
    scores = cross_val_score(neigh, X_train, Y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())
    nn.append(n)
    score.append(sc)


# In[63]:

plt.plot(nn,cv_scores)
plt.title('Cross-validation scores by n')
plt.ylabel('Cross-validation score')
plt.xlabel('n')
plt.grid()
plt.show()


# In[64]:

# changing to misclassification error
MSE = [1 - x for x in cv_scores]

# determining best k
optimal_k = neighbors[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d" % optimal_k)

# plot misclassification error vs k
plt.plot(neighbors, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.grid()
plt.show()


# In[16]:

neigh = KNeighborsClassifier(n_neighbors=n) 
print(neigh.score(X_train_pca,Y_train))
print(neigh.score(X_test_pca,Y_test))
y_pred = neigh.predict(X_test_pca)

KNN_crosstab = pd.crosstab(Y_test, y_pred,margins = True) 
KNN_crosstab


# In[17]:

print(cross_val_score(neigh, X, Y, cv=5))
print(classification_report(Y_test,y_pred))


# In[18]:

correct_list =[]
for i in KNN_crosstab.index[0:5]:
    correct = (KNN_crosstab.at[i,i]/KNN_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(neigh.score(X_test_pca,Y_test),2)*100)
results_df['KNN']=correct_list


# ### Random Forest Classifier

# In[233]:


clf = RandomForestClassifier(max_depth=None, random_state=5,n_estimators=90,max_features='sqrt',
                             min_samples_leaf=5,min_samples_split=15,criterion='entropy', bootstrap=True)

#min_samples_leaf - lower, way overfit because it allows leaf size to be 1;
    #A smaller leaf makes the model more prone to capturing noise in train data.
    # At default (1), there was significant overfitting; as I increased min_samples_leaf, 
    # the scores for both train and test decreased, but for training, there was more decline, reducing overfitting.
#random state - so my #s don't change
#n_estimators (The number of trees in the forest.) - higher # takes longer but makes predictions stronger and more stable.
#criterion did not make a difference, entropy slightly better and more stable with CV; documentation says there is little difference
#max depth - The maximum depth of the tree. As None, nodes are expanded until all leaves are pure
            #or until all leaves contain less than min_samples_split samples
    # I set min_samples_split to be 15 (default is 2) to try to reduce noise from small sample size. 
    # At 2, the model was significantly overfit; at 15, less so.
# I toggled many other parameters but found little difference in performance as I changed them.

clf.fit(X_train_pca, Y_train)


# In[275]:

from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
param_grid = { 
    'n_estimators': [50,150, 250, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
        'min_samples_leaf': [1,5,10,20,25],
        'min_samples_split': [2,5,10],
    'max_depth': [None,5,10,15,20,25],
    "criterion"         : ["gini", "entropy"],
     "bootstrap": [True, False]
}

random_search = RandomizedSearchCV(estimator=clf, param_distributions=param_grid, cv= 5, n_iter = 50)
random_search.fit(X_train_pca, Y_train)
print(random_search.best_estimator_)


# In[276]:

print(random_search.score(X_train_pca,Y_train))
print(random_search.score(X_test_pca,Y_test))

y_pred = random_search.predict(X_test_pca)

rfc_crosstab = pd.crosstab(Y_test, y_pred,margins = True) 
rfc_crosstab


# In[ ]:

print(cross_val_score(random_search, X, Y, cv=5))
print(cross_val_score(random_search, X, Y, cv=5).mean())


# In[ ]:

print(classification_report(Y_test,y_pred))


# In[22]:

correct_list =[]
for i in rfc_crosstab.index[0:5]:
    correct = (rfc_crosstab.at[i,i]/rfc_crosstab.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(clf.score(X_test_pca,Y_test),2)*100)
results_df['Random_Forest']=correct_list


# ### Gradient Boosting

# In[23]:

# GB is by far the slowest model to run


# In[252]:

# We'll make 500 iterations, use 2-deep trees, and set our loss function.
params = {'n_estimators': 500,
          'max_depth': 10,
          'loss': 'deviance',
          'min_samples_leaf': 20}

# max depth (The maximum depth of a tree, Used to control over-fitting as higher depth will allow model to 
# learn relations very specific to a particular sample) at 2 works better than 10 or 20
# increasing min_samples_leaf helped accuracy (default is 1), performed best at 20 (15 and 25 worse)
# Initialize and fit the model.
gb = ensemble.GradientBoostingClassifier(**params)
gb.fit(X_train_pca, Y_train)


# In[254]:

print(gb.score(X_train_pca,Y_train))
print(gb.score(X_test_pca,Y_test))
print(cross_val_score(gb, X, Y, cv=5))
print(cross_val_score(gb, X, Y, cv=5).mean())


# In[25]:

predict_train = gb.predict(X_train_pca)
predict_test = gb.predict(X_test_pca)


# Accuracy tables.
table_train = pd.crosstab(Y_train, predict_train, margins=True)
table_test = pd.crosstab(Y_test, predict_test, margins=True)

print(gb.score(X_train_pca,Y_train))
print(gb.score(X_test_pca,Y_test))
table_test


# In[26]:

print(cross_val_score(gb, X, Y, cv=5))
print(classification_report(Y_test,predict_test))


# In[27]:

correct_list =[]
for i in table_test.index[0:5]:
    correct = (table_test.at[i,i]/table_test.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(gb.score(X_test_pca,Y_test),2)*100)
results_df['Gradient_Boosting']=correct_list


# ## Linear Discriminant Analysis

# In[28]:


lda = LinearDiscriminantAnalysis(n_components = 10)
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_pca, Y_train)


# In[29]:

#print(lda.score(X_train_pca, Y_train))
#print(lda.score(X_test_pca, Y_test))

predict_test = lda.predict(X_test_pca)
table_test = pd.crosstab(Y_test, predict_test, margins=True)
table_test


# In[30]:

print(cross_val_score(lda, X, Y, cv=5))
print(classification_report(Y_test,predict_test))


# In[31]:

correct_list =[]
for i in table_test.index[0:5]:
    correct = (table_test.at[i,i]/table_test.at[i,'All'])
    correct = round(correct,2)* 100
    correct_list.append(correct)

correct_list.append(round(lda.score(X_test_pca,Y_test),2)*100)
results_df['LDA']=correct_list
results_df


# In[32]:

import matplotlib.pyplot as plt

def model_graph():
    ind = np.arange(6)  # the x locations for the groups
    width = 0.15       # the width of the bars

    fig, ax = plt.subplots(figsize=(15, 7))
    al = 0.6
    rects1 = ax.bar(ind, results_df['MLP'], width, color='blue',alpha= al,tick_label = results_df['shape'])
    rects2 = ax.bar(ind + width, results_df['KNN'], width, color='green',alpha= al)
    rects3 = ax.bar(ind + width*2, results_df['Random_Forest'], width, color='pink',alpha= al)
    rects4 = ax.bar(ind + width*3, results_df['Gradient_Boosting'], width, color='orange',alpha= al)
    rects5 = ax.bar(ind + width*4, results_df['LDA'], width, color='purple',alpha= al)

    ax.legend(results_df.iloc[0:0,1:7],loc=0)
    plt.ylabel('Accuracy')
    plt.show()
    
model_graph()


# The neural network outperformed the other models for overall performance and for four out of the five shapes.

# In[33]:

results_df


# In[ ]:




# In[ ]:



