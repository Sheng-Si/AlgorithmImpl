# coding=utf-8
"""
简单的bp神经网络

Create by kyle 2019.05.08
"""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))


class DenseLayer(object):
    def __init__(self, units, activation=sigmoid, learning_rate=0.1, is_input_layer=False):
        """Layers of BP"""
        self.units = units  # 神经元数量
        self.activation = activation  # 激活函数
        self.learning_rate = learning_rate  # 学习率
        self.is_input_layer = is_input_layer  # 是否为输入层

        self.weight = None  # 权重
        self.bias = None  # 偏置
        self.output = None
        self.x_data = None
        self.wx_plus_b = None

        self._gradient_weight = None
        self._gradient_bias = None
        self._gradient_x = None
        self.gradient_weight = None
        self.gradient_bias = None
        self.gradient = None

    def initializer(self, back_units):
        """Init weight and bias"""
        self.weight = np.asmatrix(np.random.normal(0, 0.5, (self.units, back_units)))
        self.bias = np.asmatrix(np.random.normal(0, 0.5, self.units)).T

    def cal_gradient(self):
        if self.activation == sigmoid:
            gradient_mat = np.dot(self.output, (1 - self.output).T)
            gradient_activation = np.diag(np.diag(gradient_mat))
        else:
            gradient_activation = 1
        return gradient_activation

    def forward_propagation(self, x_data):
        self.x_data = x_data
        if self.is_input_layer:
            self.wx_plus_b = x_data
            self.output = x_data
            return x_data
        else:
            self.wx_plus_b = np.dot(self.weight, self.x_data) - self.bias
            self.output = self.activation(self.wx_plus_b)
            return self.output

    def back_propagation(self, gradient):
        gradient_activation = self.cal_gradient()
        gradient = np.asmatrix(np.dot(gradient.T, gradient_activation))

        self._gradient_weight = np.asmatrix(self.x_data)
        self._gradient_bias = -1
        self._gradient_x = self.weight

        self.gradient_weight = np.dot(gradient.T, self._gradient_weight.T)
        self.gradient_bias = gradient * self._gradient_bias
        self.gradient = np.dot(gradient, self._gradient_x).T

        self.weight = self.weight - self.learning_rate * self.gradient_weight
        self.bias = self.bias - self.learning_rate * self.gradient_bias.T

        return self.gradient


class BPNN(object):
    def __init__(self):
        self.layers = []
        self.train_mse = []
        self.fig_loss = plt.figure()
        self.ax_loss = self.fig_loss.add_subplot(1, 1, 1)

        self.train_round = None
        self.accuracy = None
        self.loss = None
        self.loss_gradient = None

    def add_layer(self, layer):
        self.layers.append(layer)

    def build(self):
        for i, layer in enumerate(self.layers[:]):
            if i < 1:
                layer.is_input_layer = True
            else:
                layer.initializer(self.layers[i - 1].units)

    def summary(self):
        for i, layer in enumerate(self.layers[:]):
            print("--------------- layer %d ---------------" % i)
            print("weight.shape", np.shape(layer.weight))
            print("bias.shape", np.shape(layer.bias))

    def call_loss(self, real_y_data, out_y_data):
        self.loss = np.sum(np.power(real_y_data - out_y_data, 2))
        self.loss_gradient = 2 * (real_y_data - out_y_data)
        return self.loss, self.loss_gradient

    def plot_loss(self):
        if self.ax_loss.lines:
            self.ax_loss.lines.remove(self.ax_loss.lines[0])
        self.ax_loss.plot(self.train_mse, "r-")
        plt.ion()
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.show()
        plt.pause(0.1)

    def train(self, x_data, y_data, train_round, accuracy):
        self.train_round = train_round
        self.accuracy = accuracy

        self.ax_loss.hlines(self.accuracy, 0, self.train_round * 1.1)

        x_shape = np.shape(x_data)
        for round_i in range(train_round):
            all_loss = 0
            for row in range(x_shape[0]):
                _x_data = np.asmatrix(x_data[row, :]).T
                _y_data = np.asmatrix(y_data[row, :]).T

                # forward propagation
                for layer in self.layers:
                    _x_data = layer.forward_propagation(_x_data)

                loss, gradient = self.call_loss(_y_data, _x_data)
                all_loss = all_loss + loss

                # back propagation
                for layer in self.layers[:0:-1]:
                    gradient = layer.back_propagation(gradient)

            mse = all_loss / x_shape[0]
            self.train_mse.append(mse)

            self.plot_loss()

            if mse < self.accuracy:
                return mse


def example():
    x = np.random.randn(10, 10)
    y = np.asarray([[0.8, 0.4], [0.4, 0.3], [0.34, 0.45], [0.67, 0.32],
                    [0.88, 0.67], [0.78, 0.77], [0.55, 0.66], [0.55, 0.43], [0.54, 0.1],
                    [0.1, 0.5]])

    model = BPNN()
    model.add_layer(DenseLayer(10))
    model.add_layer(DenseLayer(20))
    model.add_layer(DenseLayer(30))
    model.add_layer(DenseLayer(2))

    model.build()

    model.summary()

    model.train(x_data=x, y_data=y, train_round=100, accuracy=0.1)


if __name__ == '__main__':
    example()
