import numpy as np
from functools import reduce
from layers.parameter import parameter

class fc():
    def __init__(self, input_num, output_num, bias=True, requires_grad=True):

        self.input_num = input_num          
        self.output_num = output_num        
        self.requires_grad = requires_grad
        self.weight = parameter(np.random.randn(self.input_num, self.output_num) * (2/self.input_num**0.5))
        if bias:
            self.bias = parameter(np.random.randn(self.output_num))
        else:
            self.bias = None


    def forward(self, input):

        self.input_shape = input.shape    # 记录输入数据的形状
        if input.ndim > 2:
            N, C, H, W = input.shape
            self.x = input.reshape((N, -1))
        elif input.ndim == 2:
            self.x = input
        else:
            print("fc.forward的输入數據維度存在問題")
        result = np.dot(self.x, self.weight.data)
        if self.bias is not None:
            result = result + self.bias.data
        return result


    def backward(self, eta, lr):
        N, _ = eta.shape
        # 計算gredient
        next_eta = np.dot(eta, self.weight.data.T)
        self.weight.grad = np.reshape(next_eta, self.input_shape)


        x = self.x.repeat(self.output_num, axis=0).reshape((N, self.output_num, -1))
        self.W_grad = x * eta.reshape((N, -1, 1))
        self.W_grad = np.sum(self.W_grad, axis=0) / N
        self.b_grad = np.sum(eta, axis=0) / N

        # 權重更新
        self.weight.data -= lr * self.W_grad.T
        self.bias.data -= lr * self.b_grad

        return self.weight.grad

if __name__ == "__main__":
    #檢查是否正確
    input = np.array([[[[1,2,1], [2,3,1],[1,2,2]]]])
    fc_layer = fc(9, 2, bias=True, requires_grad=True)
    out = fc_layer.forward(input)
    print(out)
    W_grad, b_grad = fc_layer.backward(np.array([[0.5,0.2]]))
    print(W_grad)
    print(b_grad)


