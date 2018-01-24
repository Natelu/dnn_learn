'''
第一个dnn Demo，多多学习
'''
import numpy as np
from matplotlib import pyplot as plt
import h5py
import dnn_utils_v2
import dnn_app_utils_v2
import testCases_v2

#initialize Parameters

def initialize_parameters(n_x,n_h,n_p):
    np.random.seed(1)
    W1=np.random.randn(n_h,n_x)
    b1=np.zeros((n_h,1))
    W2=np.random.randn(n_p,n_h)
    b2=np.zeros((n_p,1))

    #Ensure the shape of Parameters
    assert (W1.shape==(n_h,n_x))
    assert(b1.shape==(n_h,1))
    assert (W2.shape==(n_p,n_h))
    assert (b2.shape==(n_p,1))

    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2}
    return parameters

def initialize_dnn(layer_dim):
    '''
    :param layer_dim:网络参数列表，元素表示该层神经元数目 
    :return: 
    '''
    np.random.seed(3)
    parameters={}
    L=len(layer_dim)

    for i in range(1,L):
        parameters['W'+str(i)]=np.random.randn(layer_dim[i],layer_dim[i-1])*0.01
        parameters['b'+str(i)]=np.zeros((layer_dim[i],1))
        #Ensutr the shape of every parameters
        assert (parameters['W'+str(i)].shape==(layer_dim[i], layer_dim[i-1]))
        assert (parameters['b'+str(i)]).shape==(layer_dim[i],1)

    return parameters

def liner_forward(A,W,b):
    Z1=np.dot(W,A)+b
    assert (Z1.shape== (W.shape[0],A.shape[1]))
    cache=(A,W,b)
    return Z1,cache

def liner_activate_forward(A_prev,W,b,activation):
    '''
    :param A_prev: 前一层的输出值
    :param W: 本层的权重
    :param b: 本层偏置项
    :param activation: 激活函数类型 sifmod ReLu
    :return: 
    '''
    # Z=np.dot(W,A_prev)+b
    if activation=='sigmod':
        Z,liner_cache=liner_forward(A_prev,W,b)
        A,actication_cache=dnn_utils_v2.sigmoid(Z)
    elif activation=='relu':
        Z,liner_cache=liner_forward(A_prev,W,b)
        A,actication_cache=dnn_utils_v2.relu(Z)
    assert (A.shape==(W.shape[0],A_prev.shape[1]))
    cache=(liner_cache,actication_cache)    #本层的输入 权重w+b; 本层的z
    return A,cache

# A_prev,W,b=testCases_v2.linear_activation_forward_test_case()
# A,liner_cache=liner_activate_forward(A_prev,W,b,activation='relu')
# print("sigmod A is :"+str(A))

def L_model_forward(X,parameters):
    '''
    :param X: train input
    :param parameters: 
    :return: 
    '''
    caches=[]
    A=X
    L=len(parameters)//2

    for l in range(1,L):
        A_prev=A
        A,cache=liner_activate_forward(A_prev,parameters['W'+str(l)],parameters['b'+str(l)],activation='relu')
        caches.append(cache)
    AL,cache=liner_activate_forward(A,parameters['W'+str(L)],parameters['b'+str(L)],activation='sigmod')
    caches.append(cache)
    assert (AL.shape==(1,X.shape[1]))
    return AL,caches

def cal_loss(AL,Y):
    m=Y.shape[0]
    cost=-np.sum(Y*np.log(AL)+(1-Y)*np.log(1-AL))/m
    cost=np.squeeze(cost)
    assert (cost.shape==())
    return cost

#计算一层的反向传播
def linear_backward(dz,cache):

    A_prev,W,b=cache
    m=A_prev.shape[1]
    dW=np.dot(dz,A_prev.T)/m
    dB=np.sum(dz,axis=1).reshape(dz.shape[0],1)/m
    dA_prev=np.dot(W.T,dz)

    assert (dA_prev.shape==A_prev.shape)
    assert (dW.shape==W.shape)
    assert (dB.shape==b.shape)

    return dA_prev,dW,dB

# 计算每层输出的导数
def linear_activation_backward(dA,cache,activation):
    '''
    :param dA: 
    :param cache: 
    :param activation: 
    :return: 
    '''
    linear_cache,activation_cache=cache
    if activation=="sigmod":
        dZ=dnn_utils_v2.sigmoid_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)
    elif activation== "relu":
        dZ=dnn_utils_v2.relu_backward(dA,activation_cache)
        dA_prev,dW,db=linear_backward(dZ,linear_cache)

    # if activation == "relu":
    #     ### START CODE HERE ### (≈ 2 lines of code)
    #     dZ =dnn_utils_v2.relu_backward(dA, activation_cache)
    #     dA_prev, dW, db = linear_backward(dZ, linear_cache)
    #     ### END CODE HERE ###
    #
    # elif activation == "sigmoid":
    #     ### START CODE HERE ### (≈ 2 lines of code)
    #     dZ = dnn_utils_v2.sigmoid_backward(dA, activation_cache)
    #     dA_prev, dW, db = linear_backward(dZ, linear_cache)
    #     ### END CODE HERE ###
    return dA_prev,dW,db

#L-layers L 层网络的后向传递
def L_model_backward(AL,Y,caches):
    '''
    :param AL: 
    :param Y: 
    :param caches: 
    :return: 
    '''
    grads={}
    L=len(caches)
    m=AL.shape[1]
    Y=Y.reshape(AL.shape)   #保持输出层与标签列表 维数相同

    dAL=-np.divide(Y,AL)-np.divide(1-Y,1-AL)
    # L sigmod,others relu
    current_cache=caches[L-1]
    grads["dA"+str(L-1)],grads["dW"+str(L)],grads["db"+str(L)]=linear_activation_backward(dAL,current_cache,"sigmod")

    #计算除最后一层sigmoid层外，其他relu层
    for l in reversed(range(L-1)):
        current_cache=caches[l]
        dA_prev,dW,db=linear_activation_backward(grads["dA"+str(l+1)],current_cache,"relu")
        grads["dA"+str(l)]=dA_prev
        grads["dW"+str(l+1)]=dW
        grads["db"+str(l+1)]=db

    return grads

#更新网络权值
def update_parameters(parameters,grades,learning_rate):
    L=len(parameters)//2  #网络层数
    for l in range(L):
        parameters["W"+str(l+1)]=parameters["W"+str(l+1)]-learning_rate*grades["dW"+str(l+1)]
        parameters["b"+str(l+1)]=parameters["b"+str(l+1)]-learning_rate*grades["db"+str(l+1)]
    return parameters