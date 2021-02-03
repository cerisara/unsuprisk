from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from networks.cvdd_Net import CVDDNet
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from utils.vocab import Vocab

from optim.unsuprisk import UnsupRisk
from torch.utils.data import Subset

import logging
import time
import torch
import torch.optim as optim
import numpy as np
import random
import dill as pickle
import math

import cosinelin
import datasets.globconfig as glob
from itertools import chain

class CVDDTrainer(BaseTrainer):

    def __init__(self, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150, lr_milestones: tuple = (),
                 batch_size: int = 128, lambda_p: float = 0.0, alpha_scheduler: str = 'hard',
                 weight_decay: float = 1e-6, device: str = 'cuda', n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        self.lambda_p = lambda_p
        self.c = None

        self.train_dists = None
        self.train_att_matrix = None
        self.train_top_words = None

        self.test_dists = None
        self.test_att_matrix = None
        self.test_top_words = None
        self.test_auc = 0.0
        self.test_scores = None
        self.test_att_weights = None

        # alpha annealing strategy
        self.alpha_milestones = np.arange(1, 6) * int(n_epochs / 5)  # 5 equidistant milestones over n_epochs
        if alpha_scheduler == 'soft':
            self.alphas = [0.0] * 5
        if alpha_scheduler == 'linear':
            self.alphas = np.linspace(.2, 1, 5)
        if alpha_scheduler == 'logarithmic':
            self.alphas = np.logspace(-4, 0, 5)
        if alpha_scheduler == 'hard':
            self.alphas = [100.0] * 4

        # unsuprisk parameters
        self.inputsize = 300
        self.lastlay0 = torch.nn.Linear(1+self.inputsize,1)
        self.lastlay = torch.nn.Sequential(
                self.lastlay0,
                torch.nn.Dropout(glob.dropout)
                ).to(device)
        self.device = device
        # init last linear layer so that it returns the CVDD score
        self.lastlay0.weight.data.fill_(0.0)
        self.lastlay0.bias.data.fill_(0.0)
        tmpidx = torch.tensor([0]).to(device)
        self.lastlay0.weight.data.index_fill_(1,tmpidx,1.)
        self.withUnsuprisk = True
        self.unsupepoch = 0

    def train(self, dataset: BaseADDataset, net: CVDDNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get number of attention heads
        n_attention_heads = net.n_attention_heads

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Initialize context vectors
        net.c.data = torch.from_numpy(
            initialize_context_vectors(net, train_loader, self.device)[np.newaxis, :]).to(self.device)

        # Set parameters and optimizer (Adam optimizer for now)
        parameters = filter(lambda p: p.requires_grad, net.parameters())
        optimizer = optim.Adam(parameters, lr=self.lr, weight_decay=self.weight_decay)

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        alpha_i = 0
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            if epoch in self.alpha_milestones:
                net.alpha = float(self.alphas[alpha_i])
                logger.info('  Temperature alpha scheduler: new alpha is %g' % net.alpha)
                alpha_i += 1

            epoch_loss = 0.0
            n_batches = 0
            att_matrix = np.zeros((n_attention_heads, n_attention_heads))
            dists_per_head = ()
            epoch_start_time = time.time()
            for data in train_loader:
                _, text_batch, _, _ = data

                text_batch = text_batch.to(self.device)
                # text_batch.shape = (sentence_length, batch_size)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                # Update network parameters via backpropagation: forward + backward + optimize

                # forward pass
                cosine_dists, context_weights, A = net(text_batch)
                # The cosine_dists are the scores that are used at test time to compute AUC (for context_dist_mean)

                scores = context_weights * cosine_dists
                # scores.shape = (batch_size, n_attention_heads)
                # A.shape = (batch_size, n_attention_heads, sentence_length)

                # get orthogonality penalty: P = (CCT - I)
                I = torch.eye(n_attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                # compute loss
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P

                # Get scores
                dists_per_head += (cosine_dists.cpu().data.numpy(),)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)  # clip gradient norms in [-0.5, 0.5]
                optimizer.step()

                # Get attention matrix
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()

                epoch_loss += loss.item()
                n_batches += 1

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info(f'| Epoch: {epoch + 1:03}/{self.n_epochs:03} | Train Time: {epoch_train_time:.3f}s '
                        f'| Train Loss: {epoch_loss / n_batches:.6f} |')

            # Save distances per attention head and attention matrix
            self.train_dists = np.concatenate(dists_per_head)
            self.train_att_matrix = att_matrix / n_batches
            self.train_att_matrix = self.train_att_matrix.tolist()

        self.train_time = time.time() - start_time

        # Get context vectors
        self.c = np.squeeze(net.c.cpu().data.numpy())
        self.c = self.c.tolist()

        # Get top words per context
        self.train_top_words = get_top_words_per_context(dataset.train_set, dataset.encoder, net, train_loader, self.device)

        # Log results
        logger.info('Training Time: {:.3f}s'.format(self.train_time))
        logger.info('Finished training.')


        # now train our unsuprisk
        if self.withUnsuprisk:

            self.coslin = [cosinelin.Cosinelin(self.inputsize,1) for i in range(glob.nheads)]
            for i in range(glob.nheads):
                self.coslin[i].setWeightsFromCentroids(net.c.squeeze()[i].detach().numpy())

            dataset.train_set = Subset(dataset.train_set0,dataset.alltrain)
            for i, row in enumerate(dataset.train_set): row['index'] = i
            tmptrloader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

            trainbatches = []
            for data in tmptrloader:
                idx, text_batch, label_batch, _ = data
                trainbatches.append((idx,text_batch,label_batch))
            # split all batches into a list of singletons
            ltext,llab = [],[]
            for idx,text_batch,label_batch in trainbatches:
                # text_batch.tolist() gives a list of list of floats (python)
                # in text batch, we have (uttlen, batch)
                ltext += text_batch.transpose(0,1).tolist()
                llab += label_batch.tolist()
            print("lens %d %d" % (len(ltext),len(llab)))
            posclassidxs = [i for i in range(len(ltext)) if llab[i]==0]
            negclassidxs = [i for i in range(len(ltext)) if llab[i]==1]
            print("n0 n1", len(posclassidxs), len(negclassidxs))

            # add (eventually) some outliers in the training set
            random.shuffle(negclassidxs)
            ncorr = min(len(negclassidxs),int(glob.traincorr*float(len(posclassidxs))))
            trainidx = posclassidxs + negclassidxs[0:ncorr]
            print("trainidx",trainidx)

            # and prepare a dev set
            try:
                with open("devdata."+str(glob.nc)+"."+str(glob.nheads),"rb") as devfile:
                    devidx = pickle.load(devfile)
            except:
                random.shuffle(negclassidxs)
                ncorr = min(len(negclassidxs),int(glob.devcorr*float(len(posclassidxs))))
                print("devncorr "+str(ncorr),glob.traincorr,len(negclassidxs),len(posclassidxs))
                devidx = posclassidxs + negclassidxs[0:ncorr]
                with open("devdata."+str(glob.nc)+"."+str(glob.nheads),"wb") as devfile:
                    pickle.dump(devidx, devfile, pickle.HIGHEST_PROTOCOL)
            ndev1 = sum([llab[i] for i in devidx])
            ndev0 = len(devidx)-ndev1
            print("DEV %d %d" % (ndev0,ndev1))

            if True:
                # also prepare the test set for our unsuprisk
                _, tstloader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)
                testbatches = []
                for data in tstloader:
                    idx, text_batch, label_batch, _ = data
                    testbatches.append((idx,text_batch,label_batch))
                testtext=[]
                for idx,text_batch,label_batch in testbatches:
                    testtext += text_batch.transpose(0,1).tolist()

            # test the original CVDD model on the test set:
            self.test(dataset,net)
            self.testtrain(dataset,net)
            print("detson from there on, we start unsuprisk training")

            n_attention_heads = net.n_attention_heads
            logger.info('Starting training unsup...')
            n_batches = 0
            net.eval()
            uttembeds, scores = [], []
            with torch.no_grad():
                for data in [ltext[i] for i in trainidx]:
                    data = torch.tensor(data)
                    text_batch = data.unsqueeze(0).to(self.device)
                    # we want (utt len, batch)
                    text_batch=text_batch.transpose(0,1)

                    # forward pass
                    cosine_dists, context_weights, A = net(text_batch)
                    uttembeds.append(net.M.detach().numpy())
                    tsc = torch.mean(cosine_dists, dim=1)
                    scores.append(tsc.item())
                    n_batches += 1

            print("text_batch",text_batch.size())
            print("cosine_dists",cosine_dists.size())
            xscores = torch.tensor(scores).to(self.device)
            print("xscores",xscores.size())
            # xscores = (batch,)
            print(xscores[0:3])
            uttemb = torch.tensor(uttembeds).to(self.device)
            print("uttembb",uttemb.size())
            # uttemb = (batch, 1, 3, 300)

            if False:
                # save training + test corpus
                with open("traindata."+str(glob.nc)+"."+str(glob.nheads),"wb") as devfile:
                    pickle.dump(uttemb.detach().cpu().numpy(), devfile, pickle.HIGHEST_PROTOCOL)
                with open("coslin."+str(glob.nc)+"."+str(glob.nheads),"ab") as devfile:
                    pickle.dump(glob.nheads, devfile, pickle.HIGHEST_PROTOCOL)

            lossfct = UnsupRisk(glob.p0,self.device)
            detparms = [l.parameters() for l in self.coslin]
            optimizerRisk = optim.Adam(chain.from_iterable(detparms), lr=glob.lr, weight_decay=self.weight_decay)
            print("starting unsup epochs %d" % (glob.nep,))
            for epoch in range(glob.nep):
                self.unsupepoch = epoch
                optimizerRisk.zero_grad()
                for l in self.coslin: l.train()
                # uttemb contains utt embeddings (fullbatch,nheads,hidden_size)
                cospred = []
                cosmean = torch.zeros((uttemb.size(0),))
                assert net.c.size(0)==1   # why is there this 1 dim ?
                for i in range(glob.nheads):
                    mm = uttemb.transpose(0,2)[i].squeeze()
                    conehead = self.coslin[i](mm)
                    cospred.append(conehead)
                    cosmean += conehead
                cosmean /= float(glob.nheads)

                loss = lossfct(cosmean)
                if not (float('-inf') < float(loss.item()) < float('inf')):
                    print("WARNING %f at unsup epoch %d" % (loss.item(),epoch))
                    # nan or inf
                    continue
                loss.backward()
                optimizerRisk.step()

                if False:
                    # save the weights at every epoch
                    detparms = [l.parameters() for l in self.coslin]
                    with open("coslin."+str(glob.nc)+"."+str(glob.nheads),"ab") as devfile:
                        for pr in detparms:
                            for p in pr:
                                pickle.dump(p.detach().cpu().numpy(), devfile, pickle.HIGHEST_PROTOCOL)

                if True:
                    # compute unsup loss on DEV
                    uttembeds=[]
                    with torch.no_grad():
                        for data in [ltext[i] for i in devidx]:
                            data = torch.tensor(data)
                            text_batch = data.unsqueeze(0).to(self.device)
                            text_batch=text_batch.transpose(0,1)
                            cosine_dists, context_weights, A = net(text_batch)
                            selfattemb = net.M.detach().numpy()
                            noise = 0.0*(np.random.rand(*selfattemb.shape)-0.5)
                            noise2 = np.multiply(selfattemb,noise)
                            selfattemb += noise2
                            uttembeds.append(selfattemb)
                    uttemb = torch.tensor(uttembeds).to(self.device)
                    for l in self.coslin: l.eval()
                    cospred = []
                    cosmean = torch.zeros((uttemb.size(0),))
                    for i in range(glob.nheads):
                        mm = uttemb.transpose(0,2)[i].squeeze()
                        conehead = self.coslin[i](mm)
                        cospred.append(conehead)
                        cosmean += conehead
                    cosmean /= float(glob.nheads)
                    devloss = lossfct(cosmean)
                    if not (float('-inf') < float(devloss.item()) < float('inf')):
                        print("WARNING %f at unsup epoch DEV %d" % (devloss.item(),epoch))
                        # nan or inf

                if True:
                    # compute unsup loss on TEST
                    uttembeds=[]
                    with torch.no_grad():
                        for data in testtext:
                            data = torch.tensor(data)
                            text_batch = data.unsqueeze(0).to(self.device)
                            text_batch=text_batch.transpose(0,1)
                            cosine_dists, context_weights, A = net(text_batch)
                            uttembeds.append(net.M.detach().numpy())
                    uttemb = torch.tensor(uttembeds).to(self.device)
                    for l in self.coslin: l.eval()
                    cospred = []
                    cosmean = torch.zeros((uttemb.size(0),))
                    for i in range(glob.nheads):
                        mm = uttemb.transpose(0,2)[i].squeeze()
                        conehead = self.coslin[i](mm)
                        cospred.append(conehead)
                        cosmean += conehead
                    cosmean /= float(glob.nheads)
                    testloss = lossfct(cosmean)
                    if not (float('-inf') < float(testloss.item()) < float('inf')):
                        print("WARNING %f at unsup epoch TEST %d" % (testloss.item(),epoch))
                        # nan or inf


                print("unsuprisk epoch %d trainloss %f devloss %f testloss %f" % (epoch,loss.item(),devloss.item(),testloss.item()))
                self.test(dataset,net)
                self.testtrain(dataset,net)

        return net

    def testdevloc(self, devtxt, devlab, net: CVDDNet, ad_score='context_dist_mean'):
        n_attention_heads = net.n_attention_heads

        n_batches = 0
        att_matrix = np.zeros((n_attention_heads, n_attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        att_weights = []
        start_time = time.time()
        net.eval()
        npscores = []
        self.lastlay.eval()
        with torch.no_grad():
            for di in range(len(devtxt)):
                dat = torch.tensor(devtxt[di])
                text_batch = dat.unsqueeze(0).to(self.device)
                # we want (utt len, batch)
                text_batch=text_batch.transpose(0,1)

                # forward pass
                cosine_dists, context_weights, A = net(text_batch)
                ad_scores = torch.mean(cosine_dists, dim=1)

                # net.M contains utt embeddings (batch,nheads,hidden_size)
                nx = torch.mean(net.M,dim=1)
                # nx = (batch,hidden_size)
                detx = torch.cat((ad_scores.view(-1,1),nx),dim=1)
                ad_scores = self.lastlay(detx).view(-1)
                npscores.append(ad_scores)
                n_batches += 1

        lossfct = UnsupRisk(glob.p0,self.device)
        allscores = torch.cat(npscores,dim=0)
        print(allscores[0:3])
        devloss = lossfct(allscores).item()
        return devloss

    def testdev(self, test_loader, net: CVDDNet, ad_score='context_dist_mean'):
        n_attention_heads = net.n_attention_heads

        epoch_loss = 0.0
        n_batches = 0
        att_matrix = np.zeros((n_attention_heads, n_attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        att_weights = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                idx, text_batch, label_batch, _ = data
                text_batch, label_batch = text_batch.to(self.device), label_batch.to(self.device)

                # forward pass
                cosine_dists, context_weights, A = net(text_batch)
                scores = context_weights * cosine_dists
                _, best_att_head = torch.min(scores, dim=1)

                # get orthogonality penalty: P = (CCT - I)
                I = torch.eye(n_attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                # compute loss
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P

                # Save tuples of (idx, label, score, best_att_head) in a list
                dists_per_head += (cosine_dists.cpu().data.numpy(),)
                ad_scores = torch.mean(cosine_dists, dim=1)

                if self.withUnsuprisk:
                    # net.M contains utt embeddings (batch,nheads,hidden_size)
                    nx = torch.mean(net.M,dim=1)
                    detx = torch.cat((ad_scores.view(-1,1),nx),dim=1)
                    self.lastlay.eval()
                    ad_scores = self.lastlay(detx).view(-1)

                idx_label_score_head += list(zip(idx,
                                                 label_batch.cpu().data.numpy().tolist(),
                                                 ad_scores.cpu().data.numpy().tolist(),
                                                 best_att_head.cpu().data.numpy().tolist()))
                att_weights += A[range(len(idx)), best_att_head].cpu().data.numpy().tolist()

                # Get attention matrix
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()

                epoch_loss += loss.item()
                n_batches += 1

        # Save distances per attention head and attention matrix
        self.test_dists = np.concatenate(dists_per_head)
        self.test_att_matrix = att_matrix / n_batches
        self.test_att_matrix = self.test_att_matrix.tolist()

        # Save list of (idx, label, score, best_att_head) tuples
        self.test_scores = idx_label_score_head
        self.test_att_weights = att_weights

        # Compute AUC
        _, labels, scores, _ = zip(*idx_label_score_head)
        labels = np.array(labels)
        scores = np.array(scores)

        if np.sum(labels) > 0:
            best_context = None
            if ad_score == 'context_dist_mean':
                self.test_auc = roc_auc_score(labels, scores)
            if ad_score == 'context_best':
                self.test_auc = 0.0
                for context in range(n_attention_heads):
                    auc_candidate = roc_auc_score(labels, self.test_dists[:, context])
                    print(auc_candidate)
                    if auc_candidate > self.test_auc:
                        self.test_auc = auc_candidate
                        best_context = context
                    else:
                        pass
        else:
            best_context = None
            self.test_auc = 0.0

        return epoch_loss/float(n_batches),self.test_auc

    def testtrain(self, dataset: BaseADDataset, net: CVDDNet, ad_score='context_dist_mean'):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get number of attention heads
        n_attention_heads = net.n_attention_heads

        # Get test data loader
        test_loader,_ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        att_matrix = np.zeros((n_attention_heads, n_attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        noscos = []
        att_weights = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                idx, text_batch, label_batch, _ = data
                text_batch, label_batch = text_batch.to(self.device), label_batch.to(self.device)

                # forward pass
                cosine_dists, context_weights, A = net(text_batch)
                scores = context_weights * cosine_dists
                _, best_att_head = torch.min(scores, dim=1)

                # get orthogonality penalty: P = (CCT - I)
                I = torch.eye(n_attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                # compute loss
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P

                # Save tuples of (idx, label, score, best_att_head) in a list
                dists_per_head += (cosine_dists.cpu().data.numpy(),)
                # cosine_dists = (batch,nheads)
                ad_scores = torch.mean(cosine_dists, dim=1)

                if self.withUnsuprisk:
                    # net.M contains utt embeddings (batch,nheads,hidden_size)
                    # our coslin outputs should be equal to cosine_dists
                    cospred = []
                    cosmean = torch.zeros((cosine_dists.size(0),))
                    assert net.c.size(0)==1   # why is there this 1 dim ?
                    for i in range(glob.nheads):
                        mm = net.M.transpose(0,1)[i]
                        conehead = self.coslin[i](mm)
                        cospred.append(conehead)
                        cosmean += conehead
                    cosmean /= float(glob.nheads)
                    noscos += cosmean.cpu().data.numpy().tolist()

                idx_label_score_head += list(zip(idx,
                                                 label_batch.cpu().data.numpy().tolist(),
                                                 ad_scores.cpu().data.numpy().tolist(),
                                                 best_att_head.cpu().data.numpy().tolist()))
                att_weights += A[range(len(idx)), best_att_head].cpu().data.numpy().tolist()

                # Get attention matrix
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()

                epoch_loss += loss.item()
                n_batches += 1

        # Save distances per attention head and attention matrix
        self.test_dists = np.concatenate(dists_per_head)
        self.test_att_matrix = att_matrix / n_batches
        self.test_att_matrix = self.test_att_matrix.tolist()

        # Save list of (idx, label, score, best_att_head) tuples
        self.test_scores = idx_label_score_head
        self.test_att_weights = att_weights
        _, labels, scores, _ = zip(*idx_label_score_head)

        self.calcAUC(labels,scores)
        aucCVDD = self.test_auc
        self.calcAUC(labels,noscos)
        auccosl = self.test_auc
        print("TESTTRAIN AUC CVDD %f COSLIN %f" % (aucCVDD, auccosl,))
        print("DETCVDDTRAIN ",idx_label_score_head)
        print("DETRISKTRAIN ",noscos)

    def test(self, dataset: BaseADDataset, net: CVDDNet, ad_score='context_dist_mean'):

        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get number of attention heads
        n_attention_heads = net.n_attention_heads

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        epoch_loss = 0.0
        n_batches = 0
        att_matrix = np.zeros((n_attention_heads, n_attention_heads))
        dists_per_head = ()
        idx_label_score_head = []
        noscos = []
        att_weights = []
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            if self.unsupepoch==0 and False:
                # save training + test corpus
                with open("testdata."+str(glob.nc)+"."+str(glob.nheads),"wb") as devfile:
                    pickle.dump(start_time, devfile, pickle.HIGHEST_PROTOCOL)

            # should be run only once to get the text of the test
            # for row in dataset.test_set:
            #     print("DETINTEST", row['stext'])

            for data in test_loader:
                idx, text_batch, label_batch, _ = data
                text_batch, label_batch = text_batch.to(self.device), label_batch.to(self.device)

                # forward pass
                cosine_dists, context_weights, A = net(text_batch)
                scores = context_weights * cosine_dists
                _, best_att_head = torch.min(scores, dim=1)

                # get orthogonality penalty: P = (CCT - I)
                I = torch.eye(n_attention_heads).to(self.device)
                CCT = net.c @ net.c.transpose(1, 2)
                P = torch.mean((CCT.squeeze() - I) ** 2)

                # compute loss
                loss_P = self.lambda_p * P
                loss_emp = torch.mean(torch.sum(scores, dim=1))
                loss = loss_emp + loss_P

                # Save tuples of (idx, label, score, best_att_head) in a list
                dists_per_head += (cosine_dists.cpu().data.numpy(),)
                # cosine_dists = (batch,nheads)
                ad_scores = torch.mean(cosine_dists, dim=1)

                if self.withUnsuprisk:
                    if self.unsupepoch==0 and False:
                        # save training + test corpus
                        with open("testdata."+str(glob.nc)+"."+str(glob.nheads),"ab") as devfile:
                            pickle.dump(net.M.detach().cpu().numpy(), devfile, pickle.HIGHEST_PROTOCOL)
                    # net.M contains utt embeddings (batch,nheads,hidden_size)
                    # our coslin outputs should be equal to cosine_dists
                    cospred = []
                    cosmean = torch.zeros((cosine_dists.size(0),))
                    assert net.c.size(0)==1   # why is there this 1 dim ?
                    for i in range(glob.nheads):
                        mm = net.M.transpose(0,1)[i]
                        conehead = self.coslin[i](mm)
                        cospred.append(conehead)
                        cosmean += conehead
                    cosmean /= float(glob.nheads)
                    noscos += cosmean.cpu().data.numpy().tolist()

                    if False:
                        # check we get the same cosine dists
                        for i in range(glob.nheads):
                            for b in range(cospred[i].size(0)):
                                manu=0.
                                n1,n2=0.,0.
                                for j in range(self.inputsize):
                                    manu += net.M[b,i,j].item()*net.c[0,i,j].item()
                                    n1 += net.M[b,i,j].item()*net.M[b,i,j].item()
                                    n2 += net.c[0,i,j].item()*net.c[0,i,j].item()
                                n1 = math.sqrt(n1)
                                n2 = math.sqrt(n2)
                                manu /= n1*n2
                                manu = 0.5 * (1. - manu)

                idx_label_score_head += list(zip(idx,
                                                 label_batch.cpu().data.numpy().tolist(),
                                                 ad_scores.cpu().data.numpy().tolist(),
                                                 best_att_head.cpu().data.numpy().tolist()))
                att_weights += A[range(len(idx)), best_att_head].cpu().data.numpy().tolist()

                # Get attention matrix
                AAT = A @ A.transpose(1, 2)
                att_matrix += torch.mean(AAT, 0).cpu().data.numpy()

                epoch_loss += loss.item()
                n_batches += 1

        # TODO alo compute the unsup loss

        self.test_time = time.time() - start_time

        # Save distances per attention head and attention matrix
        self.test_dists = np.concatenate(dists_per_head)
        self.test_att_matrix = att_matrix / n_batches
        self.test_att_matrix = self.test_att_matrix.tolist()

        # Save list of (idx, label, score, best_att_head) tuples
        self.test_scores = idx_label_score_head
        self.test_att_weights = att_weights
        _, labels, scores, _ = zip(*idx_label_score_head)

        self.calcAUC(labels,scores)
        aucCVDD = self.test_auc
        self.calcAUC(labels,noscos)
        auccosl = self.test_auc
        print("TEST AUC CVDD %f COSLIN %f" % (aucCVDD,auccosl))
        print("DETCVDD ",idx_label_score_head)
        print("DETRISK ",noscos)

        # Get top words per context
        self.test_top_words = get_top_words_per_context(dataset.test_set, dataset.encoder, net, test_loader, self.device)

        # Log results
        logger.info('Test Loss: {:.6f}'.format(epoch_loss / n_batches))
        logger.info('Test AUC: {:.2f}%'.format(100. * self.test_auc))
        logger.info('Test Time: {:.3f}s'.format(self.test_time))
        logger.info('Finished testing.')


    def calcAUC(self,labels,scores):
        # Compute AUC
        labels = np.array(labels)
        scores = np.array(scores)

        if np.sum(labels) > 0:
            best_context = None
            if True:
                # if ad_score == 'context_dist_mean':
                self.test_auc = roc_auc_score(labels, scores)
            else:
                # if ad_score == 'context_best':
                self.test_auc = 0.0
                for context in range(glob.nheads):
                    auc_candidate = roc_auc_score(labels, self.test_dists[:, context])
                    if auc_candidate > self.test_auc:
                        self.test_auc = auc_candidate
                        best_context = context
                    else:
                        pass
        else:
            best_context = None
            self.test_auc = 0.0


def initialize_context_vectors(net, train_loader, device):
    """
    Initialize the context vectors from an initial run of k-means++ on simple average sentence embeddings

    Returns
    -------
    centers : ndarray, [n_clusters, n_features]
    """
    logger = logging.getLogger()

    logger.info('Initialize context vectors...')

    # Get vector representations
    X = ()
    for data in train_loader:
        _, text, _, _ = data
        text = text.to(device)
        # text.shape = (sentence_length, batch_size)

        X_batch = net.pretrained_model(text)
        # X_batch.shape = (sentence_length, batch_size, embedding_size)

        # compute mean and normalize
        X_batch = torch.mean(X_batch, dim=0)
        X_batch = X_batch / torch.norm(X_batch, p=2, dim=1, keepdim=True).clamp(min=1e-08)
        X_batch[torch.isnan(X_batch)] = 0
        # X_batch.shape = (batch_size, embedding_size)

        X += (X_batch.cpu().data.numpy(),)

    X = np.concatenate(X)
    n_attention_heads = net.n_attention_heads

    kmeans = KMeans(n_clusters=n_attention_heads).fit(X)
    centers = kmeans.cluster_centers_ / np.linalg.norm(kmeans.cluster_centers_, ord=2, axis=1, keepdims=True)

    logger.info('Context vectors initialized.')

    return centers


def get_top_words_per_context(dataset, encoder, net, data_loader, device, k_top=25, k_sentence=100, k_words=10):
    """
    Extract the top k_words words (according to self-attention weights) from the k_sentence nearest sentences per
    context.
    :returns list (of len n_contexts) of lists of (<word>, <count>) pairs of top k_words given by occurrence
    """
    logger = logging.getLogger()
    logger.info('Get top words per context...')

    n_contexts = net.n_attention_heads

    # get cosine distances
    dists_per_context = ()
    idxs = []
    net.eval()
    with torch.no_grad():
        for data in data_loader:
            idx, text_batch, _, _ = data
            text_batch = text_batch.to(device)
            cosine_dists, _, _ = net(text_batch)
            dists_per_context += (cosine_dists.cpu().data.numpy(),)
            idxs += idx

    dists_per_context = np.concatenate(dists_per_context)
    idxs = np.array(idxs)

    # get indices of top k_sentence sentences
    idxs_top_k_sentence = []
    for context in range(n_contexts):
        sort_idx = np.argsort(dists_per_context[:, context])  # from smallest to largest cosine distance
        idxs_top_k_sentence.append(idxs[sort_idx][:k_sentence].tolist())

    top_words_list = []
    for context in range(n_contexts):
        vocab = Vocab()

        for idx in idxs_top_k_sentence[context]:

            tokens = dataset[idx]['text']
            tokens = tokens.to(device)
            _, _, A = net(tokens.view((-1, 1)))

            attention_weights = A.cpu().data.numpy()[0, context, :]
            idxs_top_k_words = np.argsort(attention_weights)[::-1][:k_words]  # from largest to smallest att weight
            text = encoder.decode(tokens.cpu().data.numpy()[idxs_top_k_words])

            vocab.add_words(text.split())

        top_words_list.append(vocab.top_words(k_top))

    logger.info('Top words extracted.')

    return top_words_list
