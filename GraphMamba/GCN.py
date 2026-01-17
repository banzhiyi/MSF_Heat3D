import torch
import torch.nn as nn
import torch.nn.functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from einops import rearrange, repeat



class GCNLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(GCNLayer, self).__init__()
        self.BN = nn.BatchNorm1d(input_dim)
        self.Activition = nn.LeakyReLU()
        self.sigma1 = torch.nn.Parameter(torch.tensor([0.1], requires_grad=True))
        self.GCN_liner_theta_1 = nn.Sequential(nn.Linear(input_dim, 256))
        self.GCN_liner_out_1 = nn.Sequential(nn.Linear(input_dim, output_dim))


    def A_to_D_inv(self, A: torch.Tensor):
        D = A.sum(2)
        batch,l=D.shape
        D1=torch.reshape(D, (batch * l,1))
        D1=D1.squeeze(1)
        D2=torch.pow(D1, -0.5)
        D2=torch.reshape(D2,(batch,l))
        D_hat=torch.zeros([batch,l,l],dtype=torch.float)
        for i in range(batch):
            D_hat[i] = torch.diag(D2[i])
        return D_hat.cuda()

    def forward(self, H, A ):
        nodes_count = A.shape[1]
        I = torch.eye(nodes_count, nodes_count, requires_grad=False).to(device)
        A = A + I
        (batch, l, c) = H.shape
        H1 = torch.reshape(H,(batch*l, c))
        H2 = self.BN(H1)
        H=torch.reshape(H2,(batch,l, c))
        D_hat = self.A_to_D_inv(A)
        A_hat = torch.matmul(D_hat, torch.matmul(A,D_hat))#点乘
        # A_hat = I + A_hat
        output = torch.matmul(A_hat, self.GCN_liner_out_1(H))#矩阵相乘
        output = self.Activition(output)
        return output


class GCN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, layers_count: int):
        super(GCN, self).__init__()
        self.channel = changel
        self.height = height
        self.width = width
        self.GCN_Branch = nn.Sequential()
        for i in range(layers_count):
            self.GCN_Branch.add_module('GCN_Branch' + str(i), GCNLayer(self.channel, self.channel))

        # self.Softmax_linear = nn.Sequential(nn.Linear(64, self.class_count))

        self.BN = nn.BatchNorm1d(64)

    def forward(self, x: torch.Tensor,A: torch.Tensor):
        H = x
        for i in range(len(self.GCN_Branch)):
            H = self.GCN_Branch[i](H, A)
        return H