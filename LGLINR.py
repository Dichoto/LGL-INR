import torch
import torch.nn as nn
import torch.nn.functional as F

class LGL_INR(nn.Module):
    
    def __init__(self, num_classes, beta=1, use_gpu=True, detach_weight=False):
        super(LGL_INR, self).__init__()
        self.num_classes = num_classes
        self.use_gpu = use_gpu
        self.sigmoid = nn.Sigmoid()
        self.beta = beta
        self.detach_weight=detach_weight # whether to allow the weight in backpropagation 
    
    def forward(self, inputs, targets):
        """
        Args: 
            inputs: predicted logit, of shape (batch_size, num_classes)
            targets: true label of shape (batch_size)
        """
        eps = 1e-7
        probs = self.sigmoid(inputs)
        targets = torch.zeros(probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1) # label to one-hot
        if self.use_gpu: targets = targets.cuda()
        size_class = targets.sum(dim=0) # size per class in minibatch
        
        loss_pos = 0 # loss of positive class
        mean_probs = list()
        mean_loss_neg = list()
        for k in range(self.num_classes): # class-wise processing
            idx_k = targets[:,k].type(torch.bool) # indicator if sample belongs to class k
            targets_k = targets[idx_k]
            if targets_k.size()[0] == 0: # some class may have no samples in minibatch
                continue
            probs_k = probs[idx_k]
            mean_probs_k = probs_k.mean(dim=0)
            mean_probs_k[k] = -1 # mask out the probability of true label
            if self.detach_weight==True:
                mean_probs_k = mean_probs_k.detach()
            mean_probs.append(mean_probs_k)
            
            logp_pos_k = torch.log(probs_k+eps)
            logp_neg_k = torch.log(1.0-probs_k+eps)
            loss_pos = loss_pos + (targets_k * logp_pos_k).sum()/size_class[k] # sumup 
            mean_loss_neg_k =  ((1.0-targets_k) * logp_neg_k).mean(dim=0) # no sumup, need reweighting
            mean_loss_neg.append(mean_loss_neg_k)
        
        mean_probs = torch.stack(mean_probs, dim=0)
        mean_loss_neg = torch.stack(mean_loss_neg, dim=0)
        
        loss_neg = 0 # loss of reweighted negative class
        for k in range(self.num_classes):
            idx_neg = (mean_probs[:,k]!=-1).type(torch.bool)
            prob_neg_k = mean_probs[idx_neg,k]
            weight_neg = F.softmax(self.beta * prob_neg_k, dim=0)
            loss_neg_k = (weight_neg * mean_loss_neg[idx_neg,k]).sum()
            loss_neg = loss_neg + loss_neg_k
        
        loss = -1 * (loss_pos + loss_neg)
        return loss