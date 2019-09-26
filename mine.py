from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.stats.contingency import margins
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def sample(joint=True,
           mean=[0, 0],
           cov=[[1, 0.9], [0.9, 1]],
           n_data=1000000):
    xy = np.random.multivariate_normal(mean=mean,
                                       cov=cov,
                                       size=n_data)
    if joint:
        return xy 
    y = np.random.multivariate_normal(mean=mean,
                                      cov=cov,
                                      size=n_data)
    x = xy[:,0].reshape(-1,1)
    y = y[:,1].reshape(-1,1)
   
    xy = np.concatenate([x, y], axis=1)
    return xy

def compute_mi(cov_xy=0.9, n_bins=100):
    cov=[[1, cov_xy], [cov_xy, 1]]
    data = sample(cov=cov)
    joint, edge = np.histogramdd(data, bins=n_bins)
    joint /= joint.sum()
    eps = np.finfo(float).eps
    joint[joint<eps] = eps
    x, y = margins(joint)
    xy = x*y
    xy[xy<eps] = eps
    mi = joint*np.log(joint/xy)
    mi = mi.sum()
    print("Computed MI:", mi)
    return mi


class Mine1(nn.Module):
    def __init__(self, hidden_units=10):
        super(Mine1, self).__init__()
        self.fc1 = nn.Linear(1, hidden_units)
        self.fc2 = nn.Linear(1, hidden_units)
        self.fc3 = nn.Linear(hidden_units, 1)

    def forward(self, x, y):
        x = F.relu(self.fc1(x) + self.fc2(y))
        x = self.fc3(x)
        return x

def train_mine(cov_xy=0.9, device='cpu', loss_func=1):
    model = Mine1().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    plot_loss = []
    n_samples = 10000
    n_epoch = 1000
    cov=[[1, cov_xy], [cov_xy, 1]]
    pred_x_y = np.empty(n_epoch, dtype=object)
    for epoch in tqdm(range(n_epoch)):
        xy = sample(n_data=n_samples, cov=cov)
        x1 = xy[:,0].reshape(-1,1)
        y1 = xy[:,1].reshape(-1,1)
        xy = sample(joint=False, n_data=n_samples, cov=cov)
        x2 = xy[:,0].reshape(-1,1)
        y2 = xy[:,1].reshape(-1,1)
    
        x1 = torch.from_numpy(x1).to(device)
        y1 = torch.from_numpy(y1).to(device)
        x2 = torch.from_numpy(x2).to(device)
        y2 = torch.from_numpy(y2).to(device)
        x1 = x1.type(torch.FloatTensor)
        y1 = y1.type(torch.FloatTensor)
        x2 = x2.type(torch.FloatTensor)
        y2 = y2.type(torch.FloatTensor)

        pred_xy = model(x1, y1)
		
        pred_x_y = model(x2, y2)
        if loss_func==2:
           ret= 1+torch.mean(pred_xy) - torch.mean(torch.exp(pred_x_y)) #f-divergence
        else:
           ret= torch.mean(pred_xy) -torch.log(torch.mean(torch.exp(pred_x_y))) #dv
		   
        loss = -ret #maximize
        plot_loss.append(loss.data.numpy())
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()

    x = np.arange(len(plot_loss))
    y = np.array(plot_loss).reshape(-1,)
    sns.set()
    sns.scatterplot(x=x, y=-y)
    func=" "
    if loss_func==2:
        func="F-Divergence"
    else:
	    func="Donsker-Varadhan"
    plt.xlabel('Epoch')
    plt.ylabel('Predicted MI')
    plt.title('Computed MI=%f with COV=%f Loss Function = %s' % (compute_mi(cov_xy=args.covariance),args.covariance,func))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MI on 2D Gaussian')
    parser.add_argument('--covariance',
                        type=float,
                        default=0.9,
                        help='gaussian off diagonal element')
    parser.add_argument('--no-gpu',
                        action='store_true',
                        default=False,
                        help='disables GPU training')
    parser.add_argument('--loss-func',
                        type=int,
                        default=1,
                        help='Set loss function to [1]DV, [2]F-Divergence')						
						
    args = parser.parse_args()
    loss_func=args.loss_func
    use_cuda = not args.no_gpu and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if loss_func==2:
	    print("Loss Function: F-Divergence")
    else:
	    print("Loss Function: Donsker-Varadhan")
    print("Device: ", device)
    print("Covariace off diagonal:", args.covariance)
    compute_mi(cov_xy=args.covariance)
    train_mine(cov_xy=args.covariance,device=device,loss_func=loss_func)