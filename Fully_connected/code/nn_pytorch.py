"""
使用深度学习框架完成性别判断问题
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt


class Net(nn.Module):

    def __init__(self, input_dim=2, out_put_dim=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim, out_features=2)
        self.fc2 = nn.Linear(in_features=2, out_features=out_put_dim)

    def forward(self, x):

        x = F.relu(self.fc1(x))
        outputs = self.fc2(x)

        return outputs

# --- 实例化一个网络 --- #
my_net = Net()
print("Network:", my_net)
# parameters
for name, params in my_net.named_parameters():
    print(name, params)
print('Total parameters:', sum(param.numel() for param in my_net.parameters()))
# --- optimizer --- #
optimizer = optim.Adam(params=my_net.parameters(), lr=0.01)

# --- data --- #
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])

all_y_trues = np.array([
  1, # Alice
  0, # Bob
  0, # Charlie
  1, # Diana
])

# --- train --- #
loss_list = list()
for i in range(1000):
    # forward
    sample_index = np.random.randint(0, 4)
    x_train = data[sample_index]
    y_label = all_y_trues[sample_index]
    # numpy to tensor
    x_train = torch.tensor(x_train).to(dtype=torch.float32)
    y_label = torch.tensor(y_label).to(dtype=torch.float32).unsqueeze(dim=0)
    y_pred = my_net(x_train)
    # loss
    loss = F.mse_loss(y_pred, y_label)
    # grad zero
    optimizer.zero_grad()
    # back propagation
    loss.backward()
    # update weight
    optimizer.step()
    print('Epoch: %d Loss : %f'%(i, loss.item()))
    loss_list.append(loss.item())
plt.plot(loss_list)
plt.show()
