import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def to_one_hot(Y):
    n_col=np.amax(Y)+1
    binarized=np.zeros((len(Y),n_col))
    for i in range(len(Y)):
        binarized[i,Y[i]]=1
    return binarized


def from_one_hot(Y):
    arr=np.zeros((len(Y),1))
    for i in range(len(Y)):
        l=Y[i]
        maxi=0
        maxj=0
        for j in range(len(l)):
            if(l[j]>maxi):
                maxi=l[j]
                if (maxi>maxj):
                    arr[i]=j
                #arr[i]=j+1
    return arr
 
    
def normalize(X,axis=1,order=2):
    l2=np.atleast_1d(np.linalg.norm(X,order,axis))
    l2[l2==0]=1
    return X/np.expand_dims(l2,axis)
    
    
df=pd.read_csv('E:/115cs0231/iris1.csv')

df['class'],class_names = pd.factorize(df['class'])
X=df.iloc[:,:-1]
#print (X)

X=normalize(X.as_matrix())
Y=df.iloc[:,-1]   
#print(Y)
print(Y.unique())#
Y=Y.as_matrix()



Y=Y.flatten()

Y=to_one_hot(Y)
#print(Y)
#dic={'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}
#print(Y.shape)
msk = np.random.rand(len(df)) < 0.8

#print(Y)
X_train = X[msk]
X_test = X[~msk]
Y_train=Y[msk]
Y_test=Y[~msk]


#print (Y_train)
#print (Y_train.shape)
#print X_test
print (Y_test)

x_test=np.array(X_test,dtype=float)
y_test=np.array(Y_test)

########array######
X_train=np.array(X_train)
#print X

#Y_train=np.array([Y_train]).T
#print (Y_train.shape)

def sigmoid(x):
	return 1/(1+np.exp(-x))

def der_sig(x):
	return x*(1-x)

epoch=10000 ###no_of_times
lr=0.1     ###learning_rate
ip_=X_train.shape[1]  ####Input neuron size
print (ip_      )

hl=5     #####no of neurons in hidden layer perceptron
op=3     #####no of neurons in output layer 

w1=2*np.random.uniform(size=(ip_,hl)) -1  ####weight of input layer
#b1=2*np.random.uniform(size=(1,hl))  -1  ###bias of ip layer
w2=2*np.random.uniform(size=(hl,op))-1    ####weight of hidden layer
#b2=2*np.random.uniform(size=(1,op)) -1    ###bias of hidden layr

#print (b2)

parameter=w1,w2
def forward_prop(X_train,parameter):
    w1,w2=parameter
    layer0=np.dot(X_train,w1)#+b1
    
    #print(layer1.shape)
    #hlip=layer1+b1
    layer1=sigmoid(layer0)
    opip1=np.dot(layer1,w2)#+b2
    layer2=sigmoid(opip1)
    cache=layer1,layer2,w2,w1#,b2,b1
    return cache
def back_prop(Y_train,cache):
    ##backpropagation####
    layer1, layer2,w2,w1=cache  #,b2,b1=cache
    e=Y_train-layer2
    #print (e.shape)
    slope_op=der_sig(layer2)
    slope_hl=der_sig(layer1)

    d_op=e*slope_op
    #print (d_op.shape)
    
    e_hl=d_op.dot(w2.T)

    d_hl=e_hl*slope_hl

    w2+=layer1.T.dot(d_op)*lr
    w1+=X_train.T.dot(d_hl)*lr

    #b2+=np.sum(d_op,axis=0,keepdims=True)*lr
    #b1+=np.sum(d_hl,axis=0,keepdims=True)*lr
    
    error=np.mean(np.abs(e))
    accuracy=(1-error)*100
    return error,accuracy,w1,w2

#print(w1)
errors=[]
for i in range (epoch):
    cache=forward_prop(X_train,parameter)
    error,accuracy,w1u,w2u=back_prop(Y_train, cache)
    parameter=w1u,w2u
    errors.append(error)
print('Training accuracy',accuracy)
#print (layer2*2)

plt.plot(errors)
plt.show()

_,y_pred,_,_=forward_prop(x_test,parameter)

print(y_pred)
#print (y_test)




print (from_one_hot(y_pred))
df1=pd.DataFrame(from_one_hot(y_pred),from_one_hot(y_test))
df1.to_csv('E:/115cs0231/iris2.csv')


test_error=from_one_hot(y_pred)-from_one_hot(y_test)
error=np.mean(np.abs(test_error))
accuracy=(1-error)*100


print ('Testing accuracy', accuracy)