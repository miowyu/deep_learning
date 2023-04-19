import numpy as np
from layers import conv_fast
from layers import pooling
from layers import activate
from layers import fc
from layers import loss
from layers import batch_normal

class nn1L():
    def __init__(self):
        
        self.fc1 = fc.fc(32*32*3, 50, bias=True, requires_grad=True)
        self.relu1 = activate.Relu()
        self.softmax = loss.softmax()

    def forward(self, imgs, labels, is_train=True):
        """
        :param imgs:输入的图片：[N,C,H,W]
        :param labels:
        :return:
        """

        x = self.fc1.forward(imgs)
        x = self.relu1.forward(x)

        loss = self.softmax.calculate_loss(x, labels)
        prediction = self.softmax.prediction_func(x)
        return loss, prediction


    def backward(self, lr):
        """
        :param lr:学习率
        :return:
        """
        eta = self.softmax.gradient()

        eta = self.relu1.backward(eta)
        eta = self.fc1.backward(eta, lr)


