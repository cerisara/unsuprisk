import math
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np

def binrisk(mu0, mu1, var0, var1, prior0, device):
    with torch.set_grad_enabled(True):
        sq2 = torch.tensor(math.sqrt(2.)).to(device)
        sigma0 = torch.sqrt(var0)
        sigma1 = torch.sqrt(var1)
        nor0 = torch.distributions.normal.Normal(mu0,sigma0)
        mor0 = torch.exp(nor0.log_prob(-1.))
        nor1 = torch.distributions.normal.Normal(mu1,sigma1)
        mor1 = torch.exp(nor1.log_prob(1.))
        prior1 = 1.-prior0

        m = mu0+1.
        r = torch.mul(prior0/2.,m)
        mm = -mu0-1.
        nn = torch.mul(sq2,sigma0)
        mm = torch.div(mm,nn)
        mm = torch.erf(mm)
        mm = 1.-mm
        term1 = torch.mul(r,mm)
        r = term1

        term2 = torch.mul(prior0,var0)
        term2 = torch.mul(term2,mor0)
        r = r+term2

        m3 = 1.-mu1
        term3 = torch.mul(prior1/2.,m3)
        nn3 = torch.mul(sq2,sigma1)
        mm3 = torch.div(m3,nn3)
        mm3 = 1. + torch.erf(mm3)
        term3 = torch.mul(term3,mm3)
        r = r+term3

        term4 = torch.mul(prior1,var1)
        term4 = torch.mul(term4,mor1)
        r = r+term4
        return r

class UnsupRisk(nn.Module):
    def __init__(self, prior0=0.5,device='cuda'):
        super(UnsupRisk,self).__init__()
        self.device=device
        self.p0 = torch.tensor(prior0, requires_grad=False).to(self.device)
        self.logpriors = torch.tensor([np.log(prior0), np.log(1.-prior0)], requires_grad=False).to(self.device)
        self.numcall=0

    def forward(self, x):
        xx,_ = torch.sort(x.view(-1))
        n = self.p0 * x.size(0)
        n = n.int()
        llow = torch.mean(xx[0:n])
        lhig = torch.mean(xx[n:])
        sigbas = torch.std(xx[0:n])
        sighaut = torch.std(xx[n:])
        l = binrisk(llow,lhig,sigbas,sighaut,self.p0,self.device)
        self.pdist = xx[n] * xx[n]
        # this hyper-parameters has been loosely tuned on the training corpus, so that the second term of the loss's influence is similar to the first term influence
        self.pdist = 1. * self.pdist
        l = l + self.pdist

        return l

