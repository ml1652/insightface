import torch
import torch.nn.functional as F

class NonID_regressor(torch.nn.Module):
    def __init__(self,num_input_features = 512):
        super(NonID_regressor,self).__init__()
        '''
        self.flatten = torch.nn.Flatten()
        #self.dense1 = torch.nn.Linear(100352, 224) # 100352 = 128X28X28
        self.dense1 = torch.nn.Linear(2048, 2048)  # pool layer2048 = 2048X1X1
        #self.dense2 = torch.nn.Linear(224, (18 * 512))
        self.dense2 = torch.nn.Linear(2048,  2048)
        self.dense3 = torch.nn.Linear(2048, 512)
        '''
        self.flatten = torch.nn.Flatten()
        self.num_input_features = num_input_features
        self.dense1 = torch.nn.Linear(num_input_features,256)  # pool layer2048 = 2048X1X1
        self.bn1 = torch.nn.BatchNorm1d(num_features=256)
        self.dense2 = torch.nn.Linear(256, 256)
        self.bn2 = torch.nn.BatchNorm1d(num_features=256)
        self.dense3 = torch.nn.Linear(256, 1)
        self.bn3 = torch.nn.BatchNorm1d(num_features=1)
        #self.dense4 = torch.nn.Linear(1)

    def forward(self, latent):
        x = self.flatten(latent)
        x = self.dense1(x)
        x = F.relu(self.bn1(x))
        x = self.dense2(x)
        x = F.relu(self.bn2(x))
        x = self.dense3(x)
        return x