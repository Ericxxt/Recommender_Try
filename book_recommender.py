import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix
from sklearn.metrics import roc_auc_score
import time 
from lightfm.evaluation import auc_score
import pickle
import re
import seaborn as sns

books=pd.read_csv('./BX-CSV-Dump/BX-Books.csv',sep=';',error_bad_lines=False,encoding="latin-1")
books.columns=['ISBN','bookTitle','bookAuthor','yearOfPublication','publisher','imageUrlS','imageUrlM','imageUrlL']
users=pd.read_csv('./BX-CSV-Dump/BX-Users.csv',sep=";",error_bad_lines=False,encoding="latin-1")
users.columns=['userID','Location','Age']
ratings=pd.read_csv('./BX-CSV-Dump/BX-Book-Ratings.csv',sep=";",error_bad_lines=False,encoding="latin-1")
ratings.columns=['userID','ISBN','bookRating']
# print(books.head())
# print(ratings.head())
# print(users.head())
# as we don't need the image information of books,we should drop it 
books.drop(['imageUrlS','imageUrlM','imageUrlL'],axis=1,inplace=True)

# print(books.dtypes)
# column width to display full text of columns.
# pd.set_option('display.max_colwidth',-1) 感觉这行没啥用 ，因为本来就能显示全部
# we can check all the unique year and delete the unformal years
# print(books.yearOfPublication.unique())
# print(books.head())
# 之前纠结这里为什么还有显示图片路径，现在发现是这几条数据有问题
# print(books.loc[books.yearOfPublication == 'DK Publishing Inc', :])
# now rectify the data
books.loc[books.ISBN == '0789466953','yearOfPublication']= 2000
books.loc[books.ISBN == '0789466953','bookAuthor']= "James Buckley"
books.loc[books.ISBN == '0789466953','publisher']= "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bootTitle']= "DK Readers: Creating the X-Men, How Comic Books Come to Life(Level 4: Proficient Readers)"

books.loc[books.ISBN == '078946697X','yearOfPublication']= 2000
books.loc[books.ISBN == '078946697X','bookAuthor']= "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher']= "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bootTitle']= "DK Readers: Creating the X-Men, How it All Began(Level 4: Proficient Readers)"
# print(books.loc[books.yearOfPublication == 'Gallimard', :])
books.loc[books.ISBN == '2070426769','yearOfPublication']= 2003
books.loc[books.ISBN == '2070426769','bookAuthor']= "Jean-Marie Gustave"
books.loc[books.ISBN == '2070426769','publisher']= "Gallimard"
books.loc[books.ISBN == '2070426769','bootTitle']= "Peuple du ciel, suivi de 'les Bergers"
# print(books.yearOfPublication.unique())
#  we have already rectify the wrong publication year books
books.yearOfPublication=pd.to_numeric(books.yearOfPublication,errors='coerce')
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication']=np.NaN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()),inplace=True)
books.loc[(books.ISBN == '193169656X'), 'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'), 'publisher'] = 'other'
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
# df.mean()等价于df.mean(0)。把轴向数据求平均，得到每列数据的平均值。
users.Age = users.Age.fillna(users.Age.mean())
users.Age =users.Age.astype(np.int32)

ratings_new=ratings[ratings.ISBN.isin(books.ISBN)]
ratings=ratings[ratings.userID.isin(users.userID)]
ratings_explicit = ratings_new[ratings_new.bookRating != 0]
#  show the ratings distribution
# sns.countplot(data=ratings_explicit, x='bookRating')
# plt.show()
def informed_train_test(rating_df,train_ratio):
    split_cut=np.int(np.round(rating_df.shape[0]*train_ratio))
    train_df=rating_df.iloc[0:split_cut]
    test_df=rating_df.iloc[split_cut::]
    test_df=test_df[(test_df['userID'].isin(train_df['userID']))& (test_df['ISBN'].isin(train_df['ISBN']))]
    id_cols= ['userID','ISBN']
    trans_cat_train=dict()
    trans_cat_test=dict()
    for k in id_cols:
        cate_enc=preprocessing.LabelEncoder()
        trans_cat_train[k]=cate_enc.fit_transform(train_df[k].values)
        trans_cat_test[k]=cate_enc.fit_transform(test_df[k].values)
    # encoding ratings
    cate_enc=preprocessing.LabelEncoder()
    ratings=dict()
    ratings['train']=cate_enc.fit_transform(train_df.bookRating)
    ratings['test']=cate_enc.transform(test_df.bookRating)
    n_users=len(np.unique(trans_cat_train['userID']))
    n_items=len(np.unique(trans_cat_train['ISBN']))
    train=coo_matrix((ratings['train'],(trans_cat_train['userID'],trans_cat_train['ISBN'])),shape=(n_users,n_items))
    test=coo_matrix((ratings['test'],(trans_cat_test['userID'],trans_cat_test['ISBN'])),shape=(n_users,n_items))
    return train,test,train_df

# train and test is coo matrix, and train_df is 3 cols with userID, ISBN, ratings
train,test,train_df=informed_train_test(ratings_explicit,0.8)
start_time=time.time()
model=LightFM(loss='warp')
model.fit(train,epochs=12,num_threads=2)

auc_train=auc_score(model,train).mean()
auc_test=auc_score(model,test).mean()
print("-----RUN TIME: {} mins ----".format((time.time()-start_time)/60))
print("Train AUC Score: {} ".format(auc_train))
print("Test AUC Score: {} ".format(auc_test))


