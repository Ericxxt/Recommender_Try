
import numpy as np
from lightfm.datasets import fetch_movielens
from lightfm import LightFM
# fetch data and format it
data = fetch_movielens(min_rating=4.0)  # only collect the movies with a rating of 4 or higher
 

 # the structure of data
 #first: train
# first: test
# first: item_features
# first: item_feature_labels
# first: item_labels
# print training and testing data

# print(data['train'])

# for single_data in data['train']:
    # print("first:",single_data[0],"second",data[1])
    # print("first:",)
# print(repr(data['train']))
# print(repr(data['test']))
'''repr()函数将对象转化为供解释器读取的形式'''
#  下面是例子，相当于加了括号
# >>> dict = {'runoob': 'runoob.com', 'google': 'google.com'};
# >>> repr(dict)
# "{'google': 'google.com', 'runoob': 'runoob.com'}"
# m1=data['train'].tocsr()
# [rows,cols]=m1.shape
# for i in range(5):
#     for j in range(5):
#         print("m1:",m1[i,j])
 
# create model
model = LightFM(loss='warp')  # warp = weighted approximate-rank pairwise
'''
warp helps us create recommendations for each user by looking at the existing user rating pairs
and predicting rankings for each, it uses the gradient descent algorithm to iteratively find the
weights that improve our prediction over time. This takes into account both the user's past rating
history content based and similar user ratings collaborative, it's a hybrid system.
WARP is an implicit feedback model: all interactions in the training matrix are treated as positive 
signals, and products that users did not interact with they implicitly do not like. The goal of the 
model is to score these implicit positives highly while assigning low scores to implicit negatives.
'''
# train model
# get an interactional matrix
model.fit(data['train'], epochs=30, num_threads=2)

# only want to print nonzero elements
# for row,col, value in zip(data['train'].row,data['train'].col,data['train'].data):
#     print("({0},{1}) {2}".format(row,col,value))

# print(data['train'].todense())
'''
parameters: the data set we want to train it on,
            the number of epochs we want to run the training for,
            the number of threads we want to run this on
Model training is accomplished via SGD (stochastic gradient descent). This means that for every pass through 
the data — an epoch — the model learns to fit the data more and more closely. We’ll run it for 10 epochs in 
this example. We can also run it on multiple cores, so we’ll set that to 2. (The dataset in this example is 
too small for that to make a difference, but it will matter on bigger datasets.)
'''
 
def sample_recommendation(model, data, user_ids):
    # our model, our data and a list of user ids(these are users we want to generate recommendations for)
 
    # number of users and movies in training data
    n_users, n_items = data['train'].shape
 
    # generate recommendation for each user we input
    '''
    iterate through every user id that we input and say that we want the list of known positives for each line
    if M considers ratings that are 5 positive and ratings that are 4 or below negative to make the problem binary 
    much simplers
    '''
    # for i in data['item_labels']:
    #     print(i)
    for user_id in user_ids:
 
        # movies they already like
        known_positives = data['item_labels'][data['train'].tocsr()[user_id].indices]
        # for result in known_positives:
        #     print(result)
        '''
        data['item_labels']的类型是  <class 'numpy.ndarray'>
        data['train']的类型是  <class 'scipy.sparse.coo.coo_matrix'> 即 坐标形式的一种稀疏矩阵
            # tocsr() 的作用是  Return a copy of this matrix in Compressed Sparse Row format
            # coo_matrix.tocsr() 将把coo_matrix转化为csr_matrix，所以，
        data['train'].tocsr()的类型是 <class 'scipy.sparse.csr.csr_matrix'> 即 压缩的行稀疏矩阵
        data['train'].tocsr()[user_id] 的类型也是 <class 'scipy.sparse.csr.csr_matrix'>
        data['train'].tocsr()[user_id].indices 的类型是 <class 'numpy.ndarray'>
            # indices属性的作用是返回	CSR format index array of the matrix
            
        总之，data['train'].tocsr()[2].indices 获取  user_id=2 的观众打分为5的电影索引数组
        data['item_labels'][...]  根据索引数组，输出对应的电影名称
        
        '''
        # movies our model predicts they will like
        scores = model.predict(user_id, np.arange(n_items))
        '''np.arange()用于创建等差数组，返回一个array对象'''
        # rank them in order of most liked to least
        top_items = data['item_labels'][np.argsort(-scores)]
        '''np.argsort(x)返回数组值从小到大的索引值,np.argsort(-x)按降序排列'''
 
        # print out the results
        print("User %s" % user_id)
        print("      Known positives:")
 
        for x in known_positives[:3]:
            print("         %s" % x)
        print("      Recommended:")
        for x in top_items[:3]:
            print("         %s" % x)
 
 
sample_recommendation(model, data, [3,25,450])
# sample_recommendation(model, data, [3])