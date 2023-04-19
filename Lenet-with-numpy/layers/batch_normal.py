import numpy as np
from layers import parameter

#每個通道計算平均、標準差，並學習兩個參數alpha、beta。
class BN():
    def __init__(self, channel, moving_decay=0.9, is_train=True):
        """
        :param input_shape:输入需要归一化的数据：[N,C,H,W]
        :param is_train: 当前是否为训练状态
        """
        self.alpha = parameter.parameter(np.ones((channel,1,1)))
        self.beta = parameter.parameter(np.zeros((channel,1,1)))
        self.is_train = is_train
        self.eps 

        self.moving_mean = np.zeros((channel,1,1))
        self.moving_var = np.zeros((channel,1,1))
        self.moving_decay = moving_decay


    def forward(self, x, is_train=True):
        """
        :param x:输入的feature map：[N,C,H,W]
        :return: batch_normalization的结果[N,C,H,W]
        """
        self.is_train = is_train
        N, C, H, W = x.shape
        self.x = x
        if N <= 4:
            print("batch size较小，BN层不能准确估计数据集的均值和方差，不建议使用BN层")
            return x

        if self.is_train:       
            self.mean = np.mean(x, axis=(0,2,3))[:,np.newaxis, np.newaxis]
            self.var = np.var(x, axis=(0,2,3))[:,np.newaxis, np.newaxis]

            # 計算移動平均和標準差
            if (np.sum(self.moving_mean)==0) and (np.sum(self.moving_var)==0):
                self.moving_mean = self.mean
                self.moving_var = self.var
            else:
                self.moving_mean = self.moving_mean * self.moving_decay + (1-self.moving_decay) * self.mean
                self.moving_var = self.moving_var * self.moving_decay + (1-self.moving_decay) * self.var

            self.y = (x - self.mean) / np.sqrt(self.var + self.eps)
            return  self.alpha.data * self.y + self.beta.data
        else:   
            self.y = (x - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
            return  self.alpha.data * self.y + self.beta.data


    def backward(self, eta, lr):
        """
        :param eta:
        :param lr:学习率
        :return:
        """
        # 計算alpha和beta的梯度
        N, _, H, W = eta.shape
        alpha_grad = np.sum(eta * self.y, axis=(0,2,3))
        beta_grad = np.sum(eta, axis=(0,2,3))

        # 返回上一層
        yx_grad = (eta * self.alpha.data)
        ymean_grad = (-1.0 / np.sqrt(self.var +self.eps)) * yx_grad
        ymean_grad = np.sum(ymean_grad, axis=(2,3))[:,:,np.newaxis,np.newaxis] / (H*W)
        yvar_grad = -0.5*yx_grad*(self.x - self.mean) / (self.var+self.eps)**(3.0/2)
        yvar_grad = 2 * (self.x-self.mean) * np.sum(yvar_grad,axis=(2,3))[:,:,np.newaxis,np.newaxis] / (H*W)
        result = yx_grad*(1 / np.sqrt(self.var +self.eps)) + ymean_grad + yvar_grad


        self.alpha.data -= lr * alpha_grad[:,np.newaxis, np.newaxis] / N
        self.beta.data -=  lr * beta_grad[:, np.newaxis, np.newaxis] / N

        return result



