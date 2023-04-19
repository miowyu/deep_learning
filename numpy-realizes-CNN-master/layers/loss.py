import numpy as np

class softmax():

    def calculate_loss(self, x, label):

        N, _ = x.shape
        self.label = np.zeros_like(x)
        for i in range(self.label.shape[0]):
            self.label[i, label[i]] = 1

        self.x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])   # 所有label計算時，每個類別先減去最大值以避免數值過大
        sum_x = np.sum(self.x, axis=1)[:, np.newaxis]
        self.prediction = self.x / sum_x

        self.loss = -np.sum(np.log(self.prediction+1e-6) * self.label)  # 加一個極小值避免預測值剛好為0
        return self.loss / N

    def prediction_func(self, x):
        x = np.exp(x - np.max(x, axis=1)[:, np.newaxis])  # 減去最大值以避免數值過大
        sum_x = np.sum(x, axis=1)[:, np.newaxis]
        self.out = x / sum_x
        return self.out


    def gradient(self):
        self.eta = self.prediction.copy() - self.label
        return self.eta




