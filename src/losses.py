import torch
import torch.nn as nn
import torch.nn.functional as F

logsoftmax = nn.LogSoftmax(dim=1)  #LogSoftmax(x_i) = log(exp(x_i)/\sum_j exp(x_j))


def softlabel_crossentropy(logits, softlabels):
    return torch.sum(torch.neg(logsoftmax(logits)) * softlabels)

def softlabel_crossentropy_split(logits, softlabels):
    return torch.neg(logsoftmax(logits)) * softlabels

def flat_crossentropy(logits, labels):
    probabilities = F.softmax(logits, dim=1)
    
    # Use the log probabilities of the true labels
    loss = torch.neg(torch.log(probabilities.gather(1, labels.view(-1, 1))))
    
    return loss