'''
Created on Apr 10, 2018
LR on Page_blocks
@author: Hanzhe Hu
'''

import numpy as np
import os
import time
from SMOTE import Smote
from SMOTE import edit_k_neighbors

#get data set
train_set=np.loadtxt('assign2_dataset/page_blocks_train_feature.txt')
train_label=np.loadtxt('assign2_dataset/page_blocks_train_label.txt')
test_set=np.loadtxt('assign2_dataset/page_blocks_test_feature.txt')
test_label=np.loadtxt('assign2_dataset/page_blocks_test_label.txt')

#normalization
def standardize(X):
    m, n = X.shape
    # 归一化每一个特征
    for j in range(n):
        features = X[:,j]
        meanVal = features.mean(axis=0)
        std = features.std(axis=0)
        if std != 0:
            X[:, j] = (features-meanVal)/std
        else:
            X[:, j] = 0
    return X
train_set=standardize(train_set)
test_set=standardize(test_set)
#add bias
train_set=np.column_stack((np.ones((train_set.shape[0],1)),train_set))   #(4935,11)
test_set=np.column_stack((np.ones((test_set.shape[0],1)),test_set))      #(538,11)
one_trainset=[]
two_trainset=[]
three_trainset=[]
four_trainset=[]
five_trainset=[]
#divide training set into five pieces
for i in range(train_set.shape[0]):
    if(train_label[i]==1):
        one_trainset.append(train_set[i])        #(4431,11)
    if(train_label[i]==2):
        two_trainset.append(train_set[i])        #(292,11)
    if(train_label[i]==3):
        three_trainset.append(train_set[i])      #(25,11)
    if(train_label[i]==4):
        four_trainset.append(train_set[i])       #(84,11)
    if(train_label[i]==5):
        five_trainset.append(train_set[i])       #(103,11)
#merge features and labels into one matrix
one_trainset_all=np.column_stack((one_trainset,np.ones((len(one_trainset),1))))        #(4431,12)
two_trainset_all=np.column_stack((two_trainset,2*np.ones((len(two_trainset),1))))      #(292,12)
three_trainset_all=np.column_stack((three_trainset,3*np.ones((len(three_trainset),1))))  #(25,12)
four_trainset_all=np.column_stack((four_trainset,4*np.ones((len(four_trainset),1))))    #(84,12)
five_trainset_all=np.column_stack((five_trainset,5*np.ones((len(five_trainset),1))))    #(103,12)


#theta=np.zeros((train_set.shape[1],1))
def sigmoid(x):
    return 1.0/(1+np.exp(-x))
def model(X, theta):
    return sigmoid(np.dot(X,theta.T))
#compute cost
def cost(X, y, theta):
    #left = np.multiply(-y, np.log(model(X, theta)))
    #right = np.multiply(1 - y, np.log(1 - model(X, theta))
    #return np.sum(left - right) / (len(X))
    left=-np.multiply(y,np.dot(X,theta.T))
    right=np.log(1+np.exp(np.dot(X,theta.T)))
    #regularization
    reg=0
    for i in range(len(theta)):
        reg=reg+theta[0][i]**2
    a=np.sum(left+right)
    value=a+0.5*reg
    return value
def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta)- y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(term) / len(X)

    return grad
#shuffle data
def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y
#set stop strategy
STOP_ITER = 0
STOP_COST = 1
STOP_GRAD = 2
def stopCriterion(type, value, threshold):
    #three ways
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold
#gradient descent

def descent(data, theta, batchSize, stopType, thresh, alpha):

    init_time = time.time()
    n=len(data)
    i = 0 # number of evaluations
    k = 0 # batch

    X, y = shuffleData(data)
    grad = np.zeros(theta.shape) # gradient
    costs = [cost(X, y, theta)] # cost
    print('begin optimizing classifier ')

    print(theta)
    iteration=0
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize
        if k >= n:
            k = 0
            X, y = shuffleData(data) #shuffle again
        #if(iteration==40000):
        #    alpha=alpha/10
        #alpha=alpha/(1+0.00001*iteration)
        theta = theta - alpha*grad # update the parameters
        iteration=iteration+1
        #print(cost(X,y,theta))
        #print('Iteration of',iteration,'time: ',time.time()-init_time,' cost: ',cost(X,y,theta))
        costs.append(cost(X, y, theta)) # compute new cost
        i += 1
#different stop strategy
        if stopType == STOP_ITER:       value = i
        elif stopType == STOP_COST:     value = costs
        elif stopType == STOP_GRAD:     value = grad
        if stopCriterion(stopType, value, thresh): break

    return theta, i-1, costs, grad, time.time() - init_time
def runexp(data, theta, batchSize, stopType, thresh, alpha):
    #import pdb; pdb.set_trace();
    theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
    name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
    name += " data - learning rate: {} - ".format(alpha)
    if batchSize==len(data): strDescType = "Gradient"
    elif batchSize==1:  strDescType = "Stochastic"
    else: strDescType = "Mini-batch ({})".format(batchSize)
    name += strDescType + " descent - Stop: "
    if stopType == STOP_ITER: strStop = "{} iterations".format(thresh)
    elif stopType == STOP_COST: strStop = "costs change < {}".format(thresh)
    else: strStop = "gradient norm < {}".format(thresh)
    name += strStop
    print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
        name, theta, iter, costs[-1], dur))
    #fig, ax = plt.subplots(figsize=(12,4))
    #ax.plot(np.arange(len(costs)), costs, 'r')
    #ax.set_xlabel('Iterations')
    #ax.set_ylabel('Cost')
    #ax.set_title(name.upper() + ' - Error vs. Iteration')
    return theta
#train 5 classifiers
def trainclassifiers():
    theta_classifiers=[]
    for i in range(1,6):
        if(i==1):
            theta=np.random.randn(1,11)
            rest_set=np.row_stack((two_trainset_all,three_trainset_all,four_trainset_all,five_trainset_all))
            one_trainset_all_ed=edit_k_neighbors(one_trainset_all,5)
            N=len(one_trainset_all_ed)/len(rest_set)*100
            s=Smote(rest_set,N)
            restset=s.over_sampling()
            #reset labels
            for i in range(len(one_trainset_all_ed)):
                one_trainset_all_ed[i][11]=1
            for i in range(len(restset)):
                restset[i][11]=0
            data=np.row_stack((one_trainset_all_ed,restset))
            #theta1=runexp(data, theta, batchSize, stopType, thresh, alpha)
            theta1=runexp(data,theta,200,STOP_COST, thresh=0.000005, alpha=0.05)
            theta_classifiers.append(theta1)
        if(i==2):
            theta=np.random.randn(1,11)
            rest_set=np.row_stack((one_trainset_all,three_trainset_all,four_trainset_all,five_trainset_all))
            #reset labels
            for i in range(len(rest_set)):
                rest_set[i][11]=0
            rest_set_ed=edit_k_neighbors(rest_set,5)
            N=len(rest_set_ed)/len(two_trainset)*100
            s=Smote(two_trainset_all,N)
            twotrainset_all=s.over_sampling()
            #reset labels
            for i in range(len(twotrainset_all)):
                twotrainset_all[i][11]=1

            data=np.row_stack((twotrainset_all,rest_set_ed))
            #theta2=runexp(data, theta, batchSize, stopType, thresh, alpha)
            theta2=runexp(data,theta,100,STOP_ITER, thresh=30000,alpha=0.05)
            theta_classifiers.append(theta2)
        if(i==3):
            theta=np.random.randn(1,11)
            rest_set=np.row_stack((two_trainset_all,one_trainset_all,four_trainset_all,five_trainset_all))
            #reset labels
            for i in range(len(rest_set)):
                rest_set[i][11]=0
            rest_set_ed=edit_k_neighbors(rest_set,5)
            N=len(rest_set_ed)/len(three_trainset)*100
            s=Smote(three_trainset_all,N)
            threetrainset_all=s.over_sampling()
            #reset labels
            for i in range(len(threetrainset_all)):
                threetrainset_all[i][11]=1

            data=np.row_stack((threetrainset_all,rest_set_ed))
            #theta3=runexp(data, theta, batchSize, stopType, thresh, alpha)
            theta3=runexp(data,theta,200,STOP_COST, thresh=0.000001, alpha=0.05)
            theta_classifiers.append(theta3)
        if(i==4):
            theta=np.random.randn(1,11)
            rest_set=np.row_stack((two_trainset_all,three_trainset_all,one_trainset_all,five_trainset_all))
            #reset labels
            for i in range(len(rest_set)):
                rest_set[i][11]=0
            rest_set_ed=edit_k_neighbors(rest_set,5)
            N=len(rest_set_ed)/len(four_trainset)*100
            s=Smote(four_trainset_all,N)
            fourtrainset_all=s.over_sampling()
            #reset labels
            for i in range(len(fourtrainset_all)):
                fourtrainset_all[i][11]=1

            data=np.row_stack((fourtrainset_all,rest_set_ed))
            #theta4=runexp(data, theta, batchSize, stopType, thresh, alpha)
            theta4=runexp(data,theta,100,STOP_ITER, thresh=30000, alpha=0.05)
            theta_classifiers.append(theta4)
        if(i==5):
            theta=np.random.randn(1,11)
            rest_set=np.row_stack((two_trainset_all,three_trainset_all,four_trainset_all,one_trainset_all))
            #reset labels
            for i in range(len(rest_set)):
                rest_set[i][11]=0
            rest_set_ed=edit_k_neighbors(rest_set,5)
            N=len(rest_set_ed)/len(five_trainset)*100
            s=Smote(five_trainset_all,N)
            fivetrainset_all=s.over_sampling()
            #reset labels
            for i in range(len(fivetrainset_all)):
                fivetrainset_all[i][11]=1

            data=np.row_stack((fivetrainset_all,rest_set_ed))
            #theta5=runexp(data, theta, batchSize, stopType, thresh, alpha)
            theta5=runexp(data,theta,100,STOP_COST, thresh=0.000001, alpha=0.05)
            theta_classifiers.append(theta5)
    return theta_classifiers

#theta_classifiers(5,11)

def testing(theta_classifiers):
    #n1,n2,n3,n4,n5 nums of labels

    result_tmp=np.zeros((5,len(test_set)))     #(5,538)
    for i in range(5):
        #feed each test example to five classifiers
        for j in range(len(test_set)):
            result_tmp[i][j]=sigmoid(np.dot(test_set[j],theta_classifiers[i].reshape(11,1)))

    #get the final classified result
    result=np.zeros(len(test_set))
    for i in range(len(test_set)):
        for j in range(1,6):
            if(result_tmp[j-1][i]==max(result_tmp[:,i])):
                result[i]=j

    #compute accuracy
    correctnum=0
    TP1=TP2=TP3=TP4=TP5=TN1=TN2=TN3=TN4=TN5=0
    FP1=FP2=FP3=FP4=FP5=FN1=FN2=FN3=FN4=FN5=0
    TF1=TF2=TF3=TF4=TF5=0
    for i in range(len(test_set)):
        if(result[i]==test_label[i]):
            correctnum=correctnum+1
            if(result[i]==1):
                TP1=TP1+1
            if(result[i]==2):
                TP2=TP2+1
            if(result[i]==3):
                TP3=TP3+1
            if(result[i]==4):
                TP4=TP4+1
            if(result[i]==5):
                TP5=TP5+1
        if(result[i]==1):
            TF1=TF1+1
        if(result[i]==2):
            TF2=TF2+1
        if(result[i]==3):
            TF3=TF3+1
        if(result[i]==4):
            TF4=TF4+1
        if(result[i]==5):
            TF5=TF5+1
    R1=TP1/482*100
    R2=TP2/37*100
    R3=TP3/3*100
    R4=TP4/4*100
    R5=TP5/12*100
    P1=TP1/TF1*100
    P2=TP2/TF2*100
    P3=TP3/TF3*100
    P4=TP4/TF4*100
    P5=TP5/TF5*100
    accuracy=correctnum/len(test_set)*100

    print('The Precision of the first class: %.2f %%' %P1,end='   ')
    print('The Recall of the first class: %.2f %%' %R1)
    print('The Precision of the second class: %.2f %%' %P2,end='   ')
    print('The Recall of the second class: %.2f %%' %R2)
    print('The Precision of the third class: %.2f %%' %P3,end='   ')
    print('The Recall of the third class: %.2f %%' %R3)
    print('The Precision of the fourth class: %.2f %%' %P4,end='   ')
    print('The Recall of the fourth class: %.2f %%' %R4),
    print('The Precision of the fifth class: %.2f %%'%P5,end='   ')
    print('The Recall of the fifth class: %.2f %%' %R5)
    print('The Accuracy of this classifier is: %.2f %%'  %accuracy)

def run():
    theta_classifiers=trainclassifiers()
    testing(theta_classifiers)
    print("The final theta of the five classifiers:")
    print(theta_classifiers)



if __name__ == '__main__':
    run()
