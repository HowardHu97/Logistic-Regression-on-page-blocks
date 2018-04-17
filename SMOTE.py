'''
Created on Apr 10, 2018
SMOTE
@author: Hanzhe Hu
'''
import random
import numpy as np
import operator

#kNN
def get_k_neighbors(index,dataset,k):
    datasize=dataset.shape[0]
    diffmat=np.tile(index,(datasize,1))-dataset
    diffmat=diffmat**2
    distance=diffmat.sum(axis=1) #sum of each row
    distance=distance**0.5
    sortdiffidx=distance.argsort()
    neighbors=[]
    for i in range(1,k+1):
        neighbors.append(dataset[sortdiffidx[i]])
    neighbors=np.array(neighbors)
    return neighbors     #(k,shape[1])

#Edited Nearest Neighbors(ENN)
def edit_k_neighbors(dataset,k):
    for i in range(len(dataset)):
        nnarray=get_k_neighbors(i,dataset,k)
        diffnum=0
        for j in range(len(nnarray)):
            if(dataset[i][11]!=nnarray[j][11]):
                diffnum=diffnum+1
        if(diffnum>k/2):
            dataset=np.delete(dataset,i,axis=0)
    return dataset
#oversampling
class Smote:
    def __init__(self,samples,N=10,k=5):
        self.n_samples,self.n_attrs=samples.shape
        self.N=int(N/100)
        self.k=k
        self.samples=samples
        self.newindex=0
        self.synthetic=np.zeros((self.samples.shape[0]*self.N,self.samples.shape[1]))

    def over_sampling(self):
        #N=self.N
        self.synthetic = np.zeros((self.samples.shape[0]*self.N,self.samples.shape[1]))
        #neighbors=NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        for i in range(len(self.samples)):
            #nnarray=neighbors.kneighbors(self.samples[i].reshape(1,-1),return_distance=False)[0]
            #print nnarray
            nnarray=get_k_neighbors(i,self.samples,self.k)
            self._populate(self.N,i,nnarray)
        return self.synthetic


    # for each minority class samples,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self,N,i,nnarray):
        for j in range(N):
            nn=random.randint(0,self.k-1)
            dif=nnarray[nn]-self.samples[i]
            gap=random.random()
            self.synthetic[self.newindex]=self.samples[i]+gap*dif
            self.newindex+=1
#a=np.array([[1,2,3],[4,5,6],[2,3,1],[2,1,2],[2,3,4],[2,3,4]])
#s=Smote(a,N=600)
#print(s.over_sampling())
