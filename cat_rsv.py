import numpy as np
import os
import time
import h5py
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
from dnn_utils_v2 import *
from dnn_app_utils_v2 import *
# from dnn_demo import *
import pickle
#matplotlib inline
# print(plt.rcParams.keys())
plt.rcParams['figure.figsize']=(5.0,4.0)
plt.rcParams['image.interpolation']='nearest'
plt.rcParams['image.cmap']='gray'

np.random.seed(1)

def load_data():
    train_f=h5py.File('dataset'+os.sep+'train_catvnoncat.h5','r')
    test_f=h5py.File('dataset'+os.sep+'test_catvnoncat.h5','r')
    return train_f['train_set_x'][:],train_f['train_set_y'][:],test_f['test_set_x'][:],test_f['test_set_y'][:],train_f['list_classes'][:]


# print(len(train_x_ori))
# print(train_y[7])
# print(classes[train_y[7]])
# print(train_f.keys())
# index=7
# plt.imshow(train_x_ori[index])
# plt.show()
# print("train set has {} samples".format(train_x_ori.shape[0]))
# print("each image size is {}".format(train_x_ori.shape[1]))
# print("test_ori_x shape is {}".format(test_x_ori.shape))
# m_train =train_x_ori.shape[0]
# num_px=train_x_ori.shape[1]
# m_test=test_x_ori.shape[0]

#reshape the train and test examples
def reshapeIma(train_x_ori,test_x_ori):
    train_x_flatten=train_x_ori.reshape(train_x_ori.shape[0],-1).T
    test_x_flatten =test_x_ori.reshape(test_x_ori.shape[0],-1).T
    #standardize data have feature value between 0 and 1
    train_x=train_x_flatten/255
    test_x=test_x_flatten/255
    print("变换图像向量 及 标准化完成...")
    return train_x,test_x



def two_layer_model(X,Y,layer_dim,learning_rate=0.0075,num_iteration=3000,print_cost=False):
    '''
    :param X: 
    :param Y: 
    :param layer_dim: 
    :param learning_rate: 
    :param num_iteration: 
    :param print_cost: 
    :return: 
    '''
    np.random.seed(1)
    grads={}
    costs=[]
    m=X.shape[1]    #the number of examples
    (n_x,n_h,n_y)=layer_dim

    #initialize parameters
    parameters=initialize_parameters(layer_dim[0],layer_dim[1],layer_dim[2])

    #get parameters detial
    W1=parameters['W1']
    W2=parameters['W2']
    b1=parameters['b1']
    b2=parameters['b2']

    #loop gradient descent
    for i in range(0,num_iteration):
        #计算前向传播
        A1,cache1=linear_activation_forward(X,W1,b1,'relu')
        A2,cache2=linear_activation_forward(A1,W2,b2,'sigmoid')

        #计算损失函数
        cost=compute_cost(A2,Y)

        #初始化后向传播
        dA2=-(np.divide(Y,A2)-np.divide(1-Y,1-A2))

        dA1,dW2,db2=linear_activation_backward(dA2,cache2,'sigmoid')
        dA0,dW1,db1=linear_activation_backward(dA1,cache1,'relu')

        grads['dW1']=dW1
        grads['dW2']=dW2
        grads['db1']=db1
        grads['db2']=db2

        #更新权重
        parameters=update_parameters(parameters,grads,learning_rate)
        #取新权值
        W1 = parameters['W1']
        W2 = parameters['W2']
        b1 = parameters['b1']
        b2 = parameters['b2']

        if print_cost and i % 100==0:
            print("cost after itertation {} is {}".format(i,np.squeeze(cost)))
        if print_cost and i%100==0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('itertations(per tens)')
    plt.title('learning rate = '+str(learning_rate))
    plt.show()
    return parameters

def predict_y(train_x,train_y,parameters):
    #计算前向传播
    A1, cache1 = linear_activation_forward(train_x,parameters['W1'], parameters['b1'], 'relu')
    A2,cache2=linear_activation_forward(A1,parameters['W2'], parameters['b2'], 'sigmoid')
    # A2.reshape(train_y.shape)
    for i in range(len(A2[0])):
        if A2[0][i]>0.5:
            A2[0][i]=1
        else:
            A2[0][i] = 0
    print("Auccary is :{}".format(np.sum(train_y==A2[0])/len(train_y)))
n_x=12288
n_h=7
n_y=1



if __name__=='__main__':
    train_x_ori, train_y, test_x_ori, test_y, classes = load_data()  # 四个数组
    train_x, test_x = reshapeIma(train_x_ori, test_x_ori)
    #parameters = two_layer_model(train_x, train_y, layer_dim=(n_x, n_h, n_y), num_iteration=2500, print_cost=True)
    #保存网络参数
    # file_parameters = open('parameters.pkl', 'wb')
    # pickle.dump(parameters, file_parameters)
    # file_parameters.close()
    #读取网络参数
    f=open('parameters.pkl','rb')
    paras=pickle.load(f)
    f.close()
    # print(paras['W1'].shape)
    #预测测试集
    predict_y(test_x,test_y,paras)