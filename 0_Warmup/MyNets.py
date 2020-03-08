import torch.nn as nn
import torch.nn.functional as F
import torch

class SimpleNet(nn.Module):
    def __init__(self,class_num):
        super(SimpleNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        # nn.Conv2d(3,6,5)
        # nn.MaxPool2d(2,2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_num)
        # nn.Linear()
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x= 4 x 84
        x = self.fc3(x)
        #x = 4 x 10
        return x #4x10

class MasterNet_v0(nn.Module):
    def __init__(self,device,classe_num):
        super(MasterNet_v0,self).__init__()

        self.device=device
        self.sub_models=[]
        self.add_a_modeldict(SimpleNet(),"./Saves/simple_net_1.pth")
        self.add_a_modeldict(SimpleNet(),"./Saves/simple_net_2.pth")
        # self.linears=[nn.Linear(10,10) for model in self.sub_models]
        self.final_linear=nn.Linear(classe_num*len(self.sub_models),classe_num) # 10 is cifar-10 claases

    def forward(self,x): # bs 4 x rgb 3 x w32 x h32   or  14 x 14
        # o1=self.sub_models[0](x)
        # o2=self.sub_models[1](x)
        out=[self.sub_models[i](x) for i in range(len(self.sub_models))]
        # out=[o1,o2]
        # out 0 = 4 x 10
        # out 1 = 4 x 10
        out=torch.cat(out,1)
        # out = 4 x 30 # if layer =3
        out = self.final_linear(out)

        return out
    def add_a_modeldict(self,modelnet,modelpth):
        net=modelnet
        net.load_state_dict(torch.load(modelpth))
        net.to(self.device)
        net.eval()
        self.sub_models.append(net)

if __name__=="__main__":
    # net=SimpleNet()
    net=MasterNet_v0()