import torch
import torch.nn as nn

class Cosinelin(nn.Linear):
    def __init__(self,nin,nout):
        super(Cosinelin,self).__init__(nin,nout,bias=False)
        self.b = nn.Parameter(torch.zeros((nout,)))

    def forward(self,x):
        y = super(Cosinelin,self).forward(x)
        wn = self.weight.norm()
        z = y/wn
        z = z.view(-1)
        xn = x.norm(dim=1)
        z = z/xn
        res = 0.5*(1.0-z) # + self.b
        return res

    def setWeightsFromCentroids(self,c):
        cc = torch.from_numpy(c).view(-1)
        self.weight[0].data.copy_(cc.data)

