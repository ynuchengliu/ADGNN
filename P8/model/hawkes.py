import math
from time import time
import numpy as  np
import torch
from torch import nn, optim
from tqdm import tqdm




# A = np.load('../data/PEMS04/all_A(train,val,test).npy')
# train_A = np.load('../data/PEMS04/AM_D4_Conv_Harry_Karl_norm_Shuffle-False.npy')


def get_event_list(all_A,event_flag):
    event_list = []
    for i,j in zip(all_A[:-1],all_A[1:]):
        event = j - i
        event[event>=event_flag] = 2
        event[event<=-event_flag] = 3
        event[(event<event_flag)&(event>-event_flag)] =1
        event_list.append(event)
    event_list = np.array(event_list)
    return event_list

def get_event_train_and_event_ture(event_list,batch_size):
    event_ture = []
    for j in event_list[batch_size::batch_size]:
        event_ture.append(j)
    event_ture = np.array(event_ture)
    return event_list,event_ture

def get_u_begin(train_A):
    u=[]
    u_list = []
    num_nodes = len(train_A[0])
    temp=len(train_A[train_A==0])/(len(train_A)*num_nodes*num_nodes)
    temp2 = (1-temp)/2
    u.append(temp)
    u.append(temp2)
    u.append(temp2)
    u = np.array(u)
    for i in range(num_nodes*num_nodes):
        u_list.append(u)
    u_list = np.array(u_list)
    u_list=u_list.reshape(num_nodes,num_nodes,3)
    return u_list
class generator(nn.Module):
    def __init__(self,num_nodes,event_list_length):
        super(generator, self).__init__()
        self.a =nn.Parameter(torch.FloatTensor(event_list_length,num_nodes,num_nodes,3),requires_grad=True)
        self.w = nn.Parameter(torch.FloatTensor(event_list_length,num_nodes, 1, num_nodes), requires_grad=True)
        self.liner = nn.ModuleList()
        for i in range(num_nodes):
            self.liner.append(nn.Linear(3,1))
        torch.nn.init.xavier_uniform_(self.a)
        torch.nn.init.xavier_uniform_(self.w)


    def forward(self, time,event_list,u_begin,batch_size):


        event_prediction_list = []
        event_list=event_list[:-1]
        sequence = range(len(event_list))
        num_nodes = len(event_list[0])
        for start in tqdm(sequence[::batch_size],ncols=80):
            lamda_temp = []
            event_begin = event_list[start:start+batch_size]
            event_begin = torch.FloatTensor(event_begin).cuda(1)

            for node in range(num_nodes):
                u = u_begin[node]
                lamda = u + self.a[start][node] * math.exp((-torch.matmul(self.w[start][node], event_begin[0][node]) * time[0])) + \
                        self.a[start+1][node] * math.exp((-torch.matmul(self.w[start+1][node], event_begin[1][node]) * time[1]))
                temp = torch.softmax(lamda,dim=1)
                temp2 = temp.detach().cpu().numpy()
                temp = self.liner[node](temp)
                event_prediction_list.append(temp)
                lamda_temp.append(temp2)
            lamda_temp = np.array(lamda_temp)
            lamda_temp = lamda_temp.reshape(num_nodes,num_nodes,3)
            u_begin = lamda_temp
            u_begin = torch.FloatTensor(u_begin).cuda(1)
        temp3 = event_prediction_list[0]
        for i in event_prediction_list[1:]:
            temp3 = torch.cat((temp3,i),0)
        temp3 = temp3.view(int(len(event_list)/2),num_nodes,num_nodes)
        return temp3


class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        self.lose_functional = torch.nn.MSELoss()
    def forward(self, ture_A ,fake_A):
        loss = self.lose_functional(ture_A, fake_A)
        start_time_test = time()
        loss.backward()
        end_time_test = time()
        print('\nBackward cost time:{:.2f}'.format((end_time_test-start_time_test)/60))
        return loss
