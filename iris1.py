import numpy as np
import pandas
import matplotlib.pyplot as plt


ds=pandas.read_csv('E:/115cs0231/iris1.csv')

df['class'],class_names = pd.factorize(df['class'])
X=df.iloc[:,:-1]
#print (X)
Y=df.iloc[:,-1]   
print(Y.unique())#

#dic={'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}

msk = np.random.rand(len(df)) < 0.8

#print(Y)
X_train = X[msk]
X_test = X[~msk]
Y_train=Y[msk]
Y_test=Y[~msk]
#print X_test
#print (Y_test)

x_test=np.array(X_test,dtype=float)
y_test=np.array(Y_test)

########array######
X_train=np.array(X_train)
#print X

Y_train=np.array(np.array([Y_train]).T)
#print (Y_train)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def der_sig(x):
	return x*(1-x)

epoch=5000 ###no_of_times
lr=0.1     ###learning_rate
ip_=X_train.shape[1]  ####Input neuron size
print (ip_      )

hl=3     #####no of neurons in hidden layer perceptron
op=1     #####no of neurons in output layer 

w1=np.random.uniform(size=(ip_,hl))   ####weight of input layer
b1=np.random.uniform(size=(1,hl))    ###bias of ip layer
w2=np.random.uniform(size=(hl,op))    ####weight of hidden layer
b2=np.random.uniform(size=(1,op))     ###bias of hidden layr

#print (b2)


def forward_prop(X_train):
  
    hlip1=np.dot(X_train,w1)
    
    #print(hlip1.shape)
    hlip=hlip1+b1
    hla=sigmoid(hlip)
    opip1=np.dot(hla,w2)
    opip=opip1+b2
    opt=sigmoid(opip)
    cache=hla,opt,w2,w1,b2,b1
    return cache
def back_prop(Y_train,cache):
    ##backpropagation####
    hla, opt,w2,w1,b2,b1=cache
    e=Y_train-opt
    slope_op=der_sig(opt)
    slope_hl=der_sig(hla)

    d_op=e*slope_op
    #print d_op.shape
    
    e_hl=d_op.dot(w2.T)

    d_hl=e_hl*slope_hl

    w2+=hla.T.dot(d_op)*lr
    w1+=X_train.T.dot(d_hl)*lr

    b2+=np.sum(d_op,axis=0,keepdims=True)*lr
    b1+=np.sum(d_hl,axis=0,keepdims=True)*lr

#print(w1)
for i in range (10000):
    cache=forward_prop(X_train)
    back_prop(Y_train, cache)

print(b2)
#print (opt*2)

_,y_pred,_,_,_,_=forward_prop(x_test)

print(y_pred*2)
print (y_test)
df1=pd.DataFrame(y_pred*2,y_test)
df1.to_csv('E:/115cs0231/iris2.csv')

plt.figure()
