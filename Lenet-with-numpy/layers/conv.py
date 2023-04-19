import numpy as np
from functools import reduce
from layers.parameter import parameter


class conv():
    def __init__(self, filter_shape, stride=1, padding='SAME', bias=True, requires_grad=True):
        """
        :param filter_shape:元组（O, C, K, K）
        :param stride: 步长
        :param padding: 填充方式:{"SAME", "VALID"}
        :param bias:是否有偏置
        :param requires_grad:是否计算梯度
        """
        self.weight = parameter(np.random.randn(*filter_shape) * (2 / reduce(lambda x,y: x*y, filter_shape[1:])**0.5))  # #“Xavier初始化
        self.stride = stride
        self.padding = padding
        self.requires_grad = requires_grad
        self.output_channel = filter_shape[0]   # 输出通道數
        self.input_channel = filter_shape[1]    # 输入通道數
        self.filter_size = filter_shape[2]  # 捲積大小
        if bias:
            self.bias = parameter(np.random.randn(self.output_channel))
        else:
            self.bias = None


    def forward(self, input):
        """
        :param input:feature map维度：[N,C,H,W]
        :return:卷积结果result：[N,O,output_H, output_W]
        """
        # 是否padding
        if self.padding == "VALID":
            self.x = input

        elif self.padding == "SAME":
            p = self.filter_size // 2
            self.x = np.pad(input, ((0,0), (0,0), (p,p), (p,p)), "constant")


        x_fit = (self.x.shape[2]-self.filter_size)%self.stride
        y_fit = (self.x.shape[3]-self.filter_size)%self.stride


        if (self.stride>1):
            if x_fit != 0:
                self.x = self.x[:,:,0:self.x.shape[2]-x_fit,:]
            if y_fit != 0:
                self.x = self.x[:,:,:,0:self.x.shape[3]-y_fit]


        #捲積計算
        N, C, H, W = self.x.shape
        output_H, output_W = (H-self.filter_size)//self.stride+1, (W-self.filter_size)//self.stride+1
        result = np.zeros((N, self.output_channel, output_H, output_W))

        for n in range(N):
            for o in range(self.output_channel):
                for i in range(0, H-self.filter_size+1, self.stride):
                    for j in range(0, W-self.filter_size+1, self.stride):
                        result[n,o,i,j] = np.sum(self.x[n, :, i:i+self.filter_size, j:j+self.filter_size]
                                                 * self.weight.data[o,:,:,:])\
                                          + (self.bias.data[o] if self.bias else 0)

        return result


    def backward(self,eta, lr):
        """
        :param eta:上一层返回的梯度[N,O,output_H, output_W]
        :return:本层的梯度result
        说明：对于某一层conv层进行求导时分为两个部分：1、本层梯度反向传播到上一层；2、本层内求导，分别对 W,b
        """

        if self.stride>1:
            N, O, output_H, output_W = eta.shape[:]
            inserted_H, inserted_W = output_H + (self.stride-1)*(output_H-1), output_W + (self.stride-1)*(output_W-1)
            insert_eta = np.zeros((N, O, inserted_H, inserted_W))
            insert_eta[:,:,::self.stride, ::self.stride] = eta[:]
            eta = insert_eta

        # 求微分
        N, _, H, W = eta.shape
        self.b_grad = eta.sum(axis=(0,2,3))
        self.W_grad = np.zeros(self.weight.data.shape)      # 形状[O, C, K, K]
        for i in range(self.filter_size):
            for j in range(self.filter_size):
                self.W_grad[:,:,i,j] = np.tensordot(eta, self.x[:,:,i:i+H,j:j+W], ([0,2,3], [0,2,3]))
        # 權重更新
        self.weight.data -= lr * self.W_grad / N
        self.bias.data -= lr * self.b_grad / N

        # 第二步边缘填充
        if self.padding == "VALID":
            p = self.filter_size - 1
            pad_eta = np.lib.pad(eta, ((0,0),(0,0),(p,p),(p,p)), "constant", constant_values=0)
            eta = pad_eta
        if self.padding == "SAME":
            p = self.filter_size // 2
            pad_eta = np.lib.pad(eta, ((0,0),(0,0),(p,p),(p,p)), "constant", constant_values=0)
            eta = pad_eta


        # 梯度傳遞
        weight_flip = np.flip(self.weight.data, (2,3))  # 旋轉180度
        weight_flip_swap = np.swapaxes(weight_flip, 0, 1)  # 交换输入、输出通道的顺序[C,O,H,W]
        N, O, H, W = eta.shape
        output_H, output_W = (H-self.filter_size)//self.stride+1, (W-self.filter_size)//self.stride+1
        self.weight.grad = np.zeros((N, weight_flip_swap.shape[0], output_H, output_W))

        for n in range(N):
            for c in range(weight_flip_swap.shape[0]):
                for i in range(0, H-self.filter_size+1, self.stride):
                    for j in range(0, W-self.filter_size+1, self.stride):
                        self.weight.grad[n,c,i,j] = np.sum(eta[n, :, i:i+self.filter_size, j:j+self.filter_size]
                                                 * weight_flip_swap[c,:,:,:])

        return self.weight.grad

