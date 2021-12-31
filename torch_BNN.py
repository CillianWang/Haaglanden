"""
A basic CNN model
/**
 * @author Xinping Wang
 * @email [x.wang3@student.tue.nl]
 * @create date 2021-09-11 09:32:41
 * @modify date 2021-09-11 09:32:41
 * @desc [description]
 */
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torchbnn as bnn
from blitz.modules import BayesianLSTM




def calc_entropy_mean(input_tensor):
    # As the last layers is already logsoftmax:
    
    
    probs = torch.exp(input_tensor)
    p_log_p = torch.log(probs) * probs
    entropy = -p_log_p.mean()
    if entropy < 0:
        print('now')
    return entropy

def calc_entropy_var(input_tensor):
    # As the last layers is already logsoftmax:
    
    probs = input_tensor/input_tensor.sum()
    p_log_p = torch.log(probs) * probs
    entropy = -p_log_p.mean()
    if entropy < 0:
        print('now')
    return entropy

classes = ("W", "R", "N1", "N2", "N3")


class DeepSleepNet_30s(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(8, 64, 128, 16)
        self.conv2 = nn.Conv1d(64, 128, 8, 1)
        self.conv3 = nn.Conv1d(128, 128, 8, 1)
        self.conv4 = nn.Conv1d(128, 128, 8, 1)

        
        self.conv1c = nn.Conv1d(8, 64, 1024, 128)
        self.conv2c = nn.Conv1d(64, 128, 6, 1)
        self.conv3c = nn.Conv1d(128, 128, 6, 1)
        self.conv4c = nn.Conv1d(128, 128, 6, 1)      
        
             
        self.flat = nn.Flatten(1, -1)
        self.fc = nn.Linear(1280, 5)
        
        self.maxpool1 = nn.MaxPool1d(8, stride=8)
        self.maxpool2 = nn.MaxPool1d(4, stride=4)
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        
        
        self.leakyrelu = nn.LeakyReLU(0.01)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax()
        self.avgpool = nn.AdaptiveAvgPool1d(5)
        self.num_layers = 3
        self.hidden_size = 64
        # self.lstm1 = nn.LSTM(34, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        
    def forward(self, x):
        input_x = x
        
        x = self.dropout1(self.maxpool1(self.leakyrelu(self.conv1(x))))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.maxpool2(self.leakyrelu(self.conv4(x)))
        
        x_c = self.dropout1(self.maxpool2(self.leakyrelu(self.conv1c(input_x))))
        x_c = self.leakyrelu(self.conv2c(x_c))
        x_c = self.leakyrelu(self.conv3c(x_c))
        x_c = self.maxpool3(self.leakyrelu(x_c))
        
        

        

        x_ = torch.cat([x, x_c], dim=-1)
        x_ = self.dropout1(x_)
        x_ = self.softmax(self.fc(self.flat(x_)))
        

        return x_


class DeepSleepNet(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(8, 64, 128, 16)
        self.conv2 = nn.Conv1d(64, 128, 8, 1)
        self.conv3 = nn.Conv1d(128, 128, 8, 1)
        self.conv4 = nn.Conv1d(128, 128, 8, 1)

        
        self.conv1c = nn.Conv1d(8, 64, 1024, 128)
        self.conv2c = nn.Conv1d(64, 128, 6, 1)
        self.conv3c = nn.Conv1d(128, 128, 6, 1)
        self.conv4c = nn.Conv1d(128, 128, 6, 1)      
        
             
        self.flat = nn.Flatten(1, -1)
        self.fc = nn.Linear(6784, 5)
        
        self.maxpool1 = nn.MaxPool1d(8, stride=8)
        self.maxpool2 = nn.MaxPool1d(4, stride=4)
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        
        
        self.leakyrelu = nn.LeakyReLU(0.01)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax()
        self.avgpool = nn.AdaptiveAvgPool1d(5)
        self.num_layers = 3
        self.hidden_size = 64
        # self.lstm1 = nn.LSTM(34, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        
    def forward(self, x):
        input_x = x
        
        x = self.dropout1(self.maxpool1(self.leakyrelu(self.conv1(x))))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.maxpool2(self.leakyrelu(self.conv4(x)))
        
        x_c = self.dropout1(self.maxpool2(self.leakyrelu(self.conv1c(input_x))))
        x_c = self.leakyrelu(self.conv2c(x_c))
        x_c = self.leakyrelu(self.conv3c(x_c))
        x_c = self.maxpool3(self.leakyrelu(self.conv4c(x_c)))
        
        

        

        x_ = torch.cat([x, x_c], dim=-1)
        x_ = self.dropout1(x_)
        x_ = self.softmax(self.fc(self.flat(x_)))
        

        return x_



class DeepSleepNet_BNN(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        self.conv1 = bnn.BayesConv1d(prior_mu=-0.001, prior_sigma=0.0005, in_channels=8, out_channels=64, kernel_size=128, stride=16)
        self.conv2 = bnn.BayesConv1d(prior_mu=-0.0035, prior_sigma=0.0008, in_channels=64, out_channels=128, kernel_size=8, stride=1)
        self.conv3 = bnn.BayesConv1d(prior_mu=-0.0023, prior_sigma=0.0004, in_channels=128, out_channels=128, kernel_size=8, stride=1)
        self.conv4 = bnn.BayesConv1d(prior_mu=-0.0017, prior_sigma=0.0004, in_channels=128, out_channels=128, kernel_size=8, stride=1)

        self.conv1c = bnn.BayesConv1d(prior_mu=-0.0012, prior_sigma=0.0001, in_channels=8, out_channels=64, kernel_size=1024, stride=128)
        self.conv2c = bnn.BayesConv1d(prior_mu=-0.0048, prior_sigma=0.001, in_channels=64, out_channels=128, kernel_size=6, stride=1)
        self.conv3c = bnn.BayesConv1d(prior_mu=-0.0026, prior_sigma=0.0005, in_channels=128, out_channels=128, kernel_size=6, stride=1)
        self.conv4c = bnn.BayesConv1d(prior_mu=-0.0019, prior_sigma=0.0005, in_channels=128, out_channels=128, kernel_size=6, stride=1)     
        
             
        self.flat = nn.Flatten(1, -1)
        self.fc = bnn.BayesLinear(prior_mu=-0.0005, prior_sigma=0.0001, in_features=6784, out_features=5)
        
        self.maxpool1 = nn.MaxPool1d(8, stride=8)
        self.maxpool2 = nn.MaxPool1d(4, stride=4)
        self.maxpool3 = nn.MaxPool1d(2, stride=2)
        
        
        self.leakyrelu = nn.LeakyReLU(0.01)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.softmax = nn.LogSoftmax()
        self.avgpool = nn.AdaptiveAvgPool1d(5)
        # self.num_layers = 3
        # self.hidden_size = 64
        # self.lstm1 = nn.LSTM(53, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        
    def forward(self, x):
        input_x = x
        
        x = self.dropout1(self.maxpool1(self.leakyrelu(self.conv1(x))))
        x = self.leakyrelu(self.conv2(x))
        x = self.leakyrelu(self.conv3(x))
        x = self.maxpool2(self.leakyrelu(self.conv4(x)))
        
        x_c = self.dropout1(self.maxpool2(self.leakyrelu(self.conv1c(input_x))))
        x_c = self.leakyrelu(self.conv2c(x_c))
        x_c = self.leakyrelu(self.conv3c(x_c))
        x_c = self.maxpool3(self.leakyrelu(self.conv4c(x_c)))
        
        

        x_ = torch.cat([x, x_c], dim=-1)
        # h0 = torch.zeros(2*self.num_layers, x_.size(0), self.hidden_size).to('cuda') 
        # c0 = torch.zeros(2*self.num_layers, x_.size(0), self.hidden_size).to('cuda')
        # x_, hidden = self.lstm1(x_, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x_ = self.dropout1(x_)
        x_ = self.softmax(self.fc(self.flat(x_)))
        

        return x_



class Passthrough(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        
        self.conv1 = nn.Conv1d(8, 16, 15)
        self.conv2 = nn.Conv1d(16, 32, 9)
        self.conv3 = nn.Conv1d(32, 64, 5)
        self.conv4 = nn.Conv1d(64, 128, 3)
        self.conv5 = nn.Conv1d(128, 128, 3)
        
        # self.conv1c = nn.Conv1d(8, 16, 6)
        # self.conv2c = nn.Conv1d(16, 32, 6)
        # self.conv3c = nn.Conv1d(32, 64, 6)
        # self.conv4c = nn.Conv1d(64, 128, 6)
        # self.conv5c = nn.Conv1d(128, 128, 6)       
        
        
        
        self.flat = nn.Flatten(1, -1)
        self.fc = nn.Linear(128*5, 5)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.LogSoftmax()
        self.avgpool = nn.AdaptiveAvgPool1d(5)
        self.num_layers = 3
        self.hidden_size = 64
        self.lstm1 = nn.LSTM(34, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        
        
    def forward(self, x):
        input_x = x
        
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv1(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv2(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv3(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv4(x))))
        x = self.conv5(x)
        
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv1c(input_x))))
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv2c(x_c))))
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv3c(x_c))))
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv4c(x_c))))
        # x_c = self.conv5c(x_c)
        
        # x = torch.cat([x, x_c], dim=-1)
        
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to('cuda') 
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to('cuda')
        x, hidden = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = self.avgpool(x)
        
        x = self.softmax(self.fc(self.flat(x)))

        # x = nn.MaxPool1d(5,5)
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.softmax(self.fc(self.flat(x)), dim=1)

        return x

def model_cnn():
    model = nn.Sequential(
        nn.Conv1d(in_channels=8, out_channels=16, kernel_size=15),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        nn.Conv1d(in_channels=16, out_channels=32, kernel_size=9),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3),
        nn.AdaptiveAvgPool1d(5),
        
        
        
        
        # nn.Flatten(1,-1),
        # nn.Linear(in_features=128*5, out_features=5),
        # nn.LogSoftmax()
        
        
    )
    return model




class BNN_LSTM(nn.Module):
    def __init__(self, n_channels=8, n_classes=5):
        super().__init__()
        
        self.conv1 = bnn.BayesConv1d(prior_mu=0, prior_sigma=0.004, in_channels=8, out_channels=16, kernel_size=15)
        self.conv2 = bnn.BayesConv1d(prior_mu=-0.002, prior_sigma=0.005, in_channels=16, out_channels=32, kernel_size=9)
        self.conv3 = bnn.BayesConv1d(prior_mu=-0.002, prior_sigma=0.006, in_channels=32, out_channels=64, kernel_size=5)
        self.conv4 = bnn.BayesConv1d(prior_mu=-0.025, prior_sigma=0.006, in_channels=64, out_channels=128, kernel_size=3)
        self.conv5 = bnn.BayesConv1d(prior_mu=0, prior_sigma=0.003, in_channels=128, out_channels=128, kernel_size=3)
        
        # self.conv1c = nn.Conv1d(8, 16, 6)
        # self.conv2c = nn.Conv1d(16, 32, 6)
        # self.conv3c = nn.Conv1d(32, 64, 6)
        # self.conv4c = nn.Conv1d(64, 128, 6)
        # self.conv5c = nn.Conv1d(128, 128, 6)       
        
        
        
        self.flat = nn.Flatten(1, -1)
        self.fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.0014, in_features=128*5, out_features=5)
        self.maxpool = nn.MaxPool1d(5, stride=5)
        self.leakyrelu = nn.LeakyReLU(0.01)
        self.dropout = nn.Dropout(p=0.1)
        self.softmax = nn.LogSoftmax()
        self.avgpool = nn.AdaptiveAvgPool1d(5)
        self.num_layers = 3
        self.hidden_size = 64
        self.lstm1 = nn.LSTM(34, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)

        #self.lstm_bnn = BayesianLSTM(34, self.hidden_size, prior_sigma_1=0.003, prior_pi=0.003)
        
        
    def forward(self, x):
        input_x = x
        
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv1(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv2(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv3(x))))
        x = self.dropout(self.maxpool(self.leakyrelu(self.conv4(x))))
        x = self.conv5(x)
        
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv1c(input_x))))
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv2c(x_c))))
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv3c(x_c))))
        # x_c = self.dropout(self.maxpool(self.leakyrelu(self.conv4c(x_c))))
        # x_c = self.conv5c(x_c)
        
        # x = torch.cat([x, x_c], dim=-1)
        #x,_ = self.lstm_bnn(x)
        h0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to('cuda') 
        c0 = torch.zeros(2*self.num_layers, x.size(0), self.hidden_size).to('cuda')

        x, hidden = self.lstm1(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)
        x = self.avgpool(x)
        
        x = self.softmax(self.fc(self.flat(x)))

        # x = nn.MaxPool1d(5,5)
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = F.softmax(self.fc(self.flat(x)), dim=1)

        return x

def model_bnn():
    model = nn.Sequential(
        bnn.BayesConv1d(prior_mu=0, prior_sigma=0.004, in_channels=8, out_channels=16, kernel_size=15),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        bnn.BayesConv1d(prior_mu=-0.002, prior_sigma=0.005, in_channels=16, out_channels=32, kernel_size=9),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        bnn.BayesConv1d(prior_mu=-0.002, prior_sigma=0.006, in_channels=32, out_channels=64, kernel_size=5),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        bnn.BayesConv1d(prior_mu=-0.025, prior_sigma=0.006, in_channels=64, out_channels=128, kernel_size=3),
        nn.LeakyReLU(0.01),
        nn.MaxPool1d(kernel_size=5, stride=5),
        nn.Dropout(p=0.1),

        bnn.BayesConv1d(prior_mu=0, prior_sigma=0.003, in_channels=128, out_channels=128, kernel_size=3),
        

        nn.AdaptiveAvgPool1d(5),
        nn.Flatten(1,-1),
        bnn.BayesLinear(prior_mu=0, prior_sigma=0.0014, in_features=128*5, out_features=5),
        nn.LogSoftmax()
    )
    return model

# def train_bnn(model, optimizer, dataloader, val_dataloader, batch_size, epochs, gamma, block, kl_weight = 0.1, kl_loss = bnn.BKLLoss(reduction='mean', last_layer_only=False)):
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma)
#     for epoch in range(epochs):
#         conf_matrix = np.zeros([5,5])
#         correct = 0
#         count = 0
#         total_loss = 0
#         index = 0
#         for dl, rl in dataloader:
#             batch_correct = 0
#             dl, rl = dl.cuda(), rl.cuda()
#             # if rl.max()==5:
#             #     continue
#             optimizer.zero_grad()


#             output = model(dl)

#             if output.shape[0] != batch_size:
#                 continue
            
#             # samples_per_cls = count_class(rl, 5)
#             # loss = CB_loss(rl.reshape(batch_size,).cpu(), output.cpu(), samples_per_cls, 5, "focal", 0.9999, 2)
#             loss = nn.CrossEntropyLoss()(output.reshape(batch_size, 5, 1), rl.long())
#             kl = kl_loss(model)
#             # rl = F.one_hot(rl.to(torch.int64), 5)\
#             loss = loss + kl_weight*kl
#             loss.backward()
#             optimizer.step()
#             # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

#             total_loss += loss


#             estimate = output.argmax(-1)
#             count += batch_size
#             for i in range(batch_size):
#                 conf_matrix[int(estimate[i])][int(rl[i])]+=1
#                 if int(estimate[i])==int(rl[i]):
#                     correct += 1
#                     batch_correct +=1
            
            

#         acc = correct/count


#         # validation:
#         conf_matrix_val = np.zeros([5,5])
#         correct_val = 0
#         count_val = 0
#         total_loss_val = 0
#         total_uncertainty = 0

#         certainty = 0
#         uncertainty = 0
#         almost_certain = 0
#         vague = 0

#         certainty_correct = 0
#         uncertainty_correct = 0
#         almost_certain_correct = 0
#         vague_correct = 0

#         certainty_e_var = 0
#         uncertainty_e_var = 0
#         almost_certain_e_var = 0
#         vague_e_var = 0
        
#         certainty_e_mean = 0
#         uncertainty_e_mean = 0
#         almost_certain_e_mean = 0
#         vague_e_mean = 0
        
#         index = 0
#         for dl, rl in val_dataloader:

#             batch_correct = 0
#             dl, rl = dl.cuda(), rl.cuda()
#             # if rl.max()==5:
#             #     continue
            

#             if dl.shape[0] != batch_size:
#                 continue
            
            
#             block_mean = 0
#             block_var  = 0
#             batch_correct_val = 0
#             if epoch> -1:
#                 out_vote = torch.zeros(10, batch_size)
#                 out_var = torch.zeros(10, batch_size, 5)
#                 for i in range(10):
#                     with torch.no_grad():
#                         out= model(dl)
#                         out_vote[i] = out.argmax(-1)
#                         out_var[i] = out.cpu()
                
#                 out_entropy_mean = out_var.mean(0)
#                 out_entropy_var = out_var.var(0)
                
#                 for i in range(batch_size):
#                     if int(rl[i]) == block:
#                         block_mean += calc_entropy_mean(out_entropy_mean[i])
#                         block_var += calc_entropy_var(out_entropy_var[i])
                        
#                     if out_vote[:, i].max() != out_vote[:, i].min():
#                         dice_count = np.zeros(5)
#                         for dice in out_vote[:,i]:
#                             dice_count[int(dice)] += 1

#                         if dice_count.max() < 5:
#                             uncertainty += 1
#                             if int(out_vote[:,i].max()) == int(rl[i]):
#                                 uncertainty_correct += 1
#                             uncertainty_e_mean += calc_entropy_mean(out_entropy_mean[i])
#                             uncertainty_e_var +=  calc_entropy_var(out_entropy_var[i])
                                

#                         elif dice_count.max() > 8:
#                             almost_certain += 1    
#                             if int(out_vote[:,i].max()) == int(rl[i]):
#                                 almost_certain_correct += 1
#                             almost_certain_e_mean += calc_entropy_mean(out_entropy_mean[i])
#                             almost_certain_e_var +=  calc_entropy_var(out_entropy_var[i])
#                         else:
#                             vague += 1
#                             if int(out_vote[:,i].max()) == int(rl[i]):
#                                 vague_correct += 1
#                             vague_e_mean += calc_entropy_mean(out_entropy_mean[i])
#                             vague_e_var +=  calc_entropy_var(out_entropy_var[i])
#                     else:
#                         certainty += 1
#                         if int(out_vote[:,i].max()) == int(rl[i]):
#                             certainty_correct += 1
#                         certainty_e_mean += calc_entropy_mean(out_entropy_mean[i])
#                         certainty_e_var +=  calc_entropy_var(out_entropy_var[i])
                        
#             with torch.no_grad():
#                 output = model(dl)

                
            

#             if output.shape[0] != batch_size:
#                 continue

#             loss = nn.CrossEntropyLoss()(output.reshape(batch_size, 5, 1), rl.long())
#             kl = kl_loss(model)
#             # rl = F.one_hot(rl.to(torch.int64), 5)\
#             loss = loss + kl_weight*kl
#             # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

#             total_loss_val += loss

#             estimate = torch.Tensor(out_vote.max(0).values)
#             count_val += batch_size
#             for i in range(batch_size):
#                 conf_matrix_val[int(estimate[i])][int(rl[i])]+=1
#                 if int(estimate[i])==int(rl[i]):
#                     correct_val += 1
#                     batch_correct_val +=1
            

#         acc_val = correct_val/count_val             

#         def certainty_acc(correct, total_num):
#             if total_num == 0:
#                 return 0
#             else:
#                 return correct/total_num

#         scheduler.step()
#         print("epoch:" +str(epoch)+"         Training loss:" + str(round(float(total_loss),1)) + "  acc:"+str(round(acc,3)) +
#                          "       ;  Validation loss:" + str(round(float(total_loss_val),1))+ "  acc:"+ str(round(acc_val,3)) +
#                           " ;uncertain:" + str(uncertainty)+ " ;vauge:"+str(vague)+ " ;almost:"+str(almost_certain)+ " ;certain:"+str(certainty))
        
#         print("uncertain acc: " + str(certainty_acc(uncertainty_correct,uncertainty))) 
#         print("vague acc:     " + str(certainty_acc(vague_correct,vague)))
#         print("almost acc:    " + str(certainty_acc(almost_certain_correct,almost_certain)))
#         print("certain acc:   " + str(certainty_acc(certainty_correct,certainty)))
#         print("")
#         print("Uncertainty means'entropy     : " + str(certainty_acc(uncertainty_e_mean, uncertainty)))
#         print("Vague means'entropy           : " + str(certainty_acc(vague_e_mean, vague)))
#         print("almost_certainty means'entropy: " + str(certainty_acc(almost_certain_e_mean, almost_certain)))
#         print("certainty means'entropy       : " + str(certainty_acc(certainty_e_mean, certainty)))
        
#         print("")
#         print("Uncertainty var's entropy     : " + str(certainty_acc(uncertainty_e_var, uncertainty)))
#         print("Vague var's entropy           : " + str(certainty_acc(vague_e_var, vague)))
#         print("almost_certainty var's entropy: " + str(certainty_acc(almost_certain_e_var, almost_certain)))
#         print("certainty var's entropy       : " + str(certainty_acc(certainty_e_var, certainty)))
        
        
#         print("blocked class's mean total entropy:" + str(block_mean))
#         print("blocked class's var total entropy :" + str(block_var))

#         if epoch%15 == 0:
#             print("After 15 epochs, Training Confusion matrix:")
#             print(conf_matrix)
#         if epoch%15 == 0:
#             print("After 15 epochs, Validation Confusion matrix:")
#             print(conf_matrix_val)
        
        



# # training cnn
# def train_cnn(model, optimizer, dataloader, val_dataloader, batch_size, epochs):
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,0.99)
#     for epoch in range(epochs):
        
#         correct = 0
#         count = 0
#         total_loss = 0
#         index = 0
#         for dl, rl in dataloader:
#             batch_correct = 0
#             dl, rl = dl.cuda(), rl.cuda()
#             # if rl.max()==5:
#             #     continue
#             optimizer.zero_grad()


#             output = model(dl)

#             if output.shape[0] != batch_size:
#                 continue
            
#             # samples_per_cls = count_class(rl, 5)
#             # loss = CB_loss(rl.reshape(batch_size,).cpu(), output.cpu(), samples_per_cls, 5, "focal", 0.9999, 2)
#             loss = nn.CrossEntropyLoss()(output.reshape(batch_size, 5, 1), rl.long())
#             # rl = F.one_hot(rl.to(torch.int64), 5)\
#             loss.backward()
#             optimizer.step()
#             # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

#             total_loss += loss


#             estimate = output.argmax(-1)
#             count += batch_size
#             for i in range(batch_size):
#                 if int(estimate[i])==int(rl[i]):
#                     correct += 1
#                     batch_correct +=1
            
#             # if index%150 == 0:
#             #     print("20 batch's acc:"+str(batch_correct))
#             # index += 1
        
#         acc = correct/count

#         acc = correct/count


#         # validation:
#         conf_matrix_val = np.zeros([5,5])
#         correct_val = 0
#         count_val = 0
#         total_loss_val = 0
#         for dl, rl in val_dataloader:
#             dl, rl = dl.cuda(), rl.cuda()
            
#             if dl.shape[0] != batch_size:
#                 continue

#             batch_correct_val = 0
            
#             with torch.no_grad():
#                 output = model(dl)
            

#             if output.shape[0] != batch_size:
#                 continue

#             loss = nn.CrossEntropyLoss()(output.reshape(batch_size, 5, 1), rl.long())
#             # rl = F.one_hot(rl.to(torch.int64), 5)\
#             # print(str(epoch)+":"+"loss equals to"+str(float(loss)))

#             total_loss_val += loss

#             estimate = output.argmax(-1)
#             count_val += batch_size
#             for i in range(batch_size):
#                 conf_matrix_val[int(estimate[i])][int(rl[i])]+=1
#                 if int(estimate[i])==int(rl[i]):
#                     correct_val += 1
#                     batch_correct_val +=1
            

#         acc_val = correct_val/count_val


#         scheduler.step()
#         print("epoch:" +str(epoch)+"         Training loss:" + str(round(float(total_loss),1)) + "  acc:"+str(round(acc,3)) +
#                          "       ;  Validation loss:" + str(round(float(total_loss_val),1))+ "  acc:"+ str(round(acc_val,3)))
#         if epoch%15 == 0:
#             print("After 15 epochs, Validation Confusion matrix:")
#             print(conf_matrix_val)