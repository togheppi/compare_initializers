# Comparison of weight and bias initializers
import torch
from torch.autograd import Variable
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# Parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 15
data_dir = '../Data/MNIST_data/'
save_dir = 'results/'

# MNIST dataset
mnist_train = dsets.MNIST(root=data_dir,
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)

mnist_test = dsets.MNIST(root=data_dir,
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True)


# Truncated normal initializer
def trunc_normal(size, mean=0.0, std=1.0):
    a, b = 0.0, (1.0 - np.exp(-2.0))
    x = np.random.rand(size)
    y = (np.random.rand(size) * (b - a)) + a
    rho = np.sqrt(-2.0 * np.log(1.0 - y))
    ret = rho * np.cos(2.0 * np.pi * x) * std + mean
    return ret.astype(np.float32)

# Model
class MLP(torch.nn.Module):
    def __init__(self, weight_init='normal', bias_init='const'):
        super(MLP, self).__init__()

        # Fully-connected layer
        fc1 = torch.nn.Linear(784, 256)
        fc2 = torch.nn.Linear(256, 256)
        out = torch.nn.Linear(256, 10)

        # Weight initializers
        if weight_init == 'normal':
            torch.nn.init.normal(fc1.weight)
            torch.nn.init.normal(fc2.weight)
            torch.nn.init.normal(out.weight)
        elif weight_init == 'trunc_normal':
            w1_size = fc1.weight.data.size()[0] * fc1.weight.data.size()[1]
            w1 = trunc_normal(w1_size).reshape(fc1.weight.data.size()[0], fc1.weight.data.size()[1])
            fc1.weight.data = torch.from_numpy(w1)

            w2_size = fc1.weight.data.size()[0] * fc2.weight.data.size()[1]
            w2 = trunc_normal(w2_size).reshape(fc2.weight.data.size()[0], fc2.weight.data.size()[1])
            fc2.weight.data = torch.from_numpy(w2)

            w3_size = out.weight.data.size()[0] * out.weight.data.size()[1]
            w3 = trunc_normal(w3_size).reshape(out.weight.data.size()[0], out.weight.data.size()[1])
            out.weight.data = torch.from_numpy(w3)
        elif weight_init == 'xavier':
            torch.nn.init.xavier_uniform(fc1.weight)
            torch.nn.init.xavier_uniform(fc2.weight)
            torch.nn.init.xavier_uniform(out.weight)
        elif weight_init == 'he':
            torch.nn.init.kaiming_normal(fc1.weight)
            torch.nn.init.kaiming_normal(fc2.weight)
            torch.nn.init.kaiming_normal(out.weight)

        # Bias initializers
        if bias_init == 'const':
            torch.nn.init.constant(fc1.bias, 0.0)
            torch.nn.init.constant(fc2.bias, 0.0)
            torch.nn.init.constant(out.bias, 0.0)
        elif bias_init == 'normal':
            torch.nn.init.normal(fc1.bias)
            torch.nn.init.normal(fc2.bias)
            torch.nn.init.normal(out.bias)

        # Layers
        self.layers = torch.nn.Sequential(
            fc1,
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            fc2,
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            out
        )

        # define loss & optimizer
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        out = self.layers(x)
        return out

    def predict(self, x):
        self.eval()
        return self.forward(x)

    def get_accuracy(self, x, y):
        prediction = self.predict(x)
        correct_prediction = (torch.max(prediction.data, 1)[1] == y.data)
        self.accuracy = torch.mean(correct_prediction.float())
        return self.accuracy

    def train_model(self, x, y):
        self.train()
        self.optimizer.zero_grad()
        hypothesis = self.forward(x)
        self.loss = self.criterion(hypothesis, y)
        self.loss.backward()
        self.optimizer.step()
        return self.loss

# Models
model_wn_bc = MLP(weight_init='normal', bias_init='const').cuda()
model_wn_bn = MLP(weight_init='normal', bias_init='normal').cuda()
model_wt_bc = MLP(weight_init='trunc_normal', bias_init='const').cuda()
model_wt_bn = MLP(weight_init='trunc_normal', bias_init='normal').cuda()
model_wx_bc = MLP(weight_init='xavier', bias_init='const').cuda()
model_wx_bn = MLP(weight_init='xavier', bias_init='normal').cuda()
model_wh_bc = MLP(weight_init='he', bias_init='const').cuda()
model_wh_bn = MLP(weight_init='he', bias_init='normal').cuda()

# Training
print('Learning started. It takes sometime.')

avg_accuracy1 = []
avg_accuracy2 = []
avg_accuracy3 = []
avg_accuracy4 = []
avg_accuracy5 = []
avg_accuracy6 = []
avg_accuracy7 = []
avg_accuracy8 = []
for epoch in range(num_epochs):
    train_accuracy1 = 0
    train_accuracy2 = 0
    train_accuracy3 = 0
    train_accuracy4 = 0
    train_accuracy5 = 0
    train_accuracy6 = 0
    train_accuracy7 = 0
    train_accuracy8 = 0

    total_batch = len(mnist_train) // batch_size
    for i, (batch_xs, batch_ys) in enumerate(data_loader):
        x_ = Variable(batch_xs.view(-1, 28*28).cuda())    # image is already size of (28x28), no reshape
        y_ = Variable(batch_ys.cuda())    # label is not one-hot encoded

        # losses
        loss1 = model_wn_bc.train_model(x_, y_)
        loss2 = model_wn_bn.train_model(x_, y_)
        loss3 = model_wt_bc.train_model(x_, y_)
        loss4 = model_wt_bn.train_model(x_, y_)
        loss5 = model_wx_bc.train_model(x_, y_)
        loss6 = model_wx_bn.train_model(x_, y_)
        loss7 = model_wh_bc.train_model(x_, y_)
        loss8 = model_wh_bn.train_model(x_, y_)

        # training accuracy
        accuracy1 = model_wn_bc.get_accuracy(x_, y_)
        accuracy2 = model_wn_bn.get_accuracy(x_, y_)
        accuracy3 = model_wt_bc.get_accuracy(x_, y_)
        accuracy4 = model_wt_bn.get_accuracy(x_, y_)
        accuracy5 = model_wx_bc.get_accuracy(x_, y_)
        accuracy6 = model_wx_bn.get_accuracy(x_, y_)
        accuracy7 = model_wh_bc.get_accuracy(x_, y_)
        accuracy8 = model_wh_bn.get_accuracy(x_, y_)

        train_accuracy1 += accuracy1 / total_batch
        train_accuracy2 += accuracy2 / total_batch
        train_accuracy3 += accuracy3 / total_batch
        train_accuracy4 += accuracy4 / total_batch
        train_accuracy5 += accuracy5 / total_batch
        train_accuracy6 += accuracy6 / total_batch
        train_accuracy7 += accuracy7 / total_batch
        train_accuracy8 += accuracy8 / total_batch

    avg_accuracy1.append(train_accuracy1)
    avg_accuracy2.append(train_accuracy2)
    avg_accuracy3.append(train_accuracy3)
    avg_accuracy4.append(train_accuracy4)
    avg_accuracy5.append(train_accuracy5)
    avg_accuracy6.append(train_accuracy6)
    avg_accuracy7.append(train_accuracy7)
    avg_accuracy8.append(train_accuracy8)

    print("[Epoch: {:>4}] train_accuracy1 = {:>.4}".format(epoch + 1, train_accuracy1))

print('Learning Finished!')

# plot losses
fig, ax = plt.subplots()
plt.xlabel('Epochs')
plt.ylabel('Training accuracy')
plt.plot(avg_accuracy1, label='(normal, zero)')
plt.plot(avg_accuracy2, label='(normal, normal)')
plt.plot(avg_accuracy3, label='(trunc_normal, zero)')
plt.plot(avg_accuracy4, label='(trunc_normal, normal)')
plt.plot(avg_accuracy5, label='(xavier, zero)')
plt.plot(avg_accuracy6, label='(xavier, normal)')
plt.plot(avg_accuracy7, label='(He, zero)')
plt.plot(avg_accuracy8, label='(He, normal)')
plt.legend()

# save figure
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
save_fn = save_dir + 'initializer_comparison.png'
plt.savefig(save_fn)

plt.show()
