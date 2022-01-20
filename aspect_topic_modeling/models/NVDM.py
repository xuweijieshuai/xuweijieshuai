import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.functional import normalize
import sys
#change to any directory you have to store the repo
sys.path.insert(1, '/home/ec2-user/SageMaker/github/aspect_topic_modeling')
from src.models.utils import get_wordnet_pos, remove_stopWords, get_emb, generate_emb, train, kld_normal, get_common_words, generate_bow


def sinkhorn_torch(M, a, b, lambda_sh, numItermax=5000, stopThr=.5e-2, cuda=False):    

    if cuda:
        u = (torch.ones_like(a) / a.size()[0]).double().cuda() 
        v = (torch.ones_like(b)).double().cuda()
    else:
        u = (torch.ones_like(a) / a.size()[0])
        v = (torch.ones_like(b))

    K = torch.exp(-M * lambda_sh) 
    err = 1
    cpt = 0
    while err > stopThr and cpt < numItermax:
        u = torch.div(a, torch.matmul(K, torch.div(b, torch.matmul(u.t(), K).t()))) 
        cpt += 1
        if cpt % 20 == 1:
            v = torch.div(b, torch.matmul(K.t(), u))  
            u = torch.div(a, torch.matmul(K, v))
            bb = torch.mul(v, torch.matmul(K.t(), u))
            err = torch.norm(torch.sum(torch.abs(bb - b), dim=0), p=float('inf'))

    sinkhorn_divergences = torch.sum(torch.mul(u, torch.matmul(torch.mul(K, M), v)), dim=0)
    return sinkhorn_divergences

def topic_covariance_penalty(topic_emb, EPS=1e-12):
    """topic_emb: T x topic_dim."""
    #normalized the topic
    normalized_topic = topic_emb / (torch.norm(topic_emb, dim=-1, keepdim=True) + EPS)
    #get topic similarity absolute value
    cosine = (normalized_topic @ normalized_topic.transpose(0, 1)).abs()
    #average similarity
    mean = cosine.mean()
    #variance
    var = ((cosine - mean) ** 2).mean()
    return mean - var, var, mean

class NTM(nn.Module):
    """NTM that keeps track of output
    """
    def __init__(self, hidden, normal, h_to_z, topics):
        super(NTM, self).__init__()
        self.hidden = hidden
        self.normal = normal
        self.h_to_z = h_to_z
        self.topics = topics
        self.output = None
        self.drop = nn.Dropout(p=0.5)
    def forward(self, x, n_sample=1):
        h = self.hidden(x)
        h = self.drop(h)
        mu, log_sigma = self.normal(h)
        #identify how far it is away from normal distribution
        kld = kld_normal(mu, log_sigma)
        #print(kld.shape)
        rec_loss = 0
        for i in range(n_sample):
            #reparametrician trick
            z = torch.zeros_like(mu).normal_() * torch.exp(0.5*log_sigma) + mu
            #decode
            
            z = self.h_to_z(z)
            self.output = z
            #print(z)
            #z = self.drop(z)
            #get log probability for reconstruction loss
            log_prob = self.topics(z)
            rec_loss = rec_loss - (log_prob * x).sum(dim=-1)
        #average reconstruction loss
        rec_loss = rec_loss / n_sample
        #print(rec_loss.shape)
        minus_elbo = rec_loss + kld

        return {
            'loss': minus_elbo,
            'minus_elbo': minus_elbo,
            'rec_loss': rec_loss,
            'kld': kld
        }

    def get_topics(self):
        return self.topics.get_topics()



def topic_embedding_weighted_penalty(embedding_weight, topic_word_logit, EPS=1e-12):
    """embedding_weight: V x dim, topic_word_logit: T x V."""
    #add importance to each word
    w = topic_word_logit.transpose(0, 1)  # V x T
    #normalized embeddings
    nv = embedding_weight / (torch.norm(embedding_weight, dim=1, keepdim=True) + EPS)  # V x dim
    #normalized importance
    nw = w / (torch.norm(w, dim=0, keepdim=True) + EPS)  # V x T
    #get topic representation
    t = nv.transpose(0, 1) @ w  # dim x T
    nt = t / (torch.norm(t, dim=0, keepdim=True) + EPS)  # dim x T
    #word - topics similarity matrix
    s = nv @ nt  # V x T
    #we want normalized importance is closed their similarity
    return -(s * nw).sum()  # minus for minimization

def negative_sampling_prior(softmax_top, index, embedding, epoch,
                           beta = 1, gamma = 1, iter2 = 30, sample = 20):
    """ add prior as a semi-supervised loss
    
    parameters
    ----------
    softmax_top: softmax results from decoder
    index: list: a list of list with number as index
    embedding: numpy array, word embedding trained by spherical word embeddings
    beta: float, weights for prior loss
    gamma: float, weights for negative sampling
    iter2: int, how many epochs to train for third phase
    sample: int, sample number
    
    
    Returns:
    --------
    int
        loss functions
    
    """
    all_index = [j for i in index for j in i]
    non_index = [i for i in range(softmax_top.shape[1]) if i not in all_index]

    logs = torch.log(softmax_top + 1e-6)
    val = logs[:, all_index].max(axis=1).values
    logns = torch.log(1 - softmax_top + 1e-6)
    #entropy = softmax_top *  torch.log(softmax_top + 1e-6) 
    penalty = 0
    for i in index:
        # print(i)
        sg = torch.argmax(logs[:, i].mean(axis=1) - val)
        #sample weights
        ns = torch.topk(logs[sg], sample).indices
        prob = 1 - (embedding.weight[ns] @ embedding.weight[i].T).max(axis = 1).values
        prob = torch.clamp(prob, max=1, min=0)
        samples = torch.bernoulli(prob)

        if 0 ==  samples.sum():
            nsl1 = 0
        else:
            negative_samples = [i for i, j in zip(ns, samples) if j == 1]
            nsl1 = logs[sg, negative_samples].mean()
        psl = logs[sg, i].mean()
        if epoch > iter2:
            penalty += (- gamma*nsl1 + beta*psl)
        else:
            penalty += beta*psl
    return - penalty #+ entropy.mean()


class EmbTopic(nn.Module):
    """
    A class used to represent decoder for Embedded Topic Modeling 
    reimplement of: https://github.com/lffloyd/embedded-topic-model
    
    Attributes
    ----------
    topic_emb: nn.Parameters
        represent topic embedding
    
    
    Methods:
    --------
    forward(logit)
        Output the result from decoder
    get_topics
        result before log
    
    
    """
    def __init__(self, embedding, k, normalize = False):
        super(EmbTopic, self).__init__()
        self.embedding = embedding
        n_vocab, topic_dim = embedding.weight.size()
        self.k = k
        self.topic_emb = nn.Parameter(torch.Tensor(k, topic_dim))
        self.reset_parameters()
        self.normalize = normalize

    def forward(self, logit):
        # return the log_prob of vocab distribution
#         if normalize:
#             self.topic_emb = torch.nn.Parameter(normalize(self.topic_emb))
        if self.normalize:
            val = normalize(self.topic_emb) @ self.embedding.weight.transpose(0, 1)
        else: 
            val = self.topic_emb @ self.embedding.weight.transpose(0, 1)
        # print(val.shape)
        beta = F.softmax(val, dim=1)
        # print(beta.shape)
        # return beta
        return torch.log(torch.matmul(logit, beta) + 1e-10)

    def get_topics(self):
        return F.softmax(self.topic_emb @ self.embedding.weight.transpose(0, 1), dim=1)
    
    
    def get_rank(self):
        #self.topic_emb = torch.nn.Parameter(normalize(self.topic_emb))
        return normalize(self.topic_emb) @ self.embedding.weight.transpose(0, 1)

    def reset_parameters(self):
        init.normal_(self.topic_emb)
        # init.kaiming_uniform_(self.topic_emb, a=math.sqrt(5))
        # init.normal_(self.embedding.weight, std=0.01)

    def extra_repr(self):
        k, d = self.topic_emb.size()
        return 'topic_emb: Parameter({}, {})'.format(k, d)
    

    
class NSSTM(NTM):
    """
    A class used for negative sampling based semi-supervised topic modeling
    
    Attributes
    ----------
    beta
    gamma
    diversity_penalty: coefficients for diversity penalty
    index: a list of index of keywords
    sample
    
    Methods:
    --------
    forward(logit)
        a disctionary of loss function
    
    
    
    """
    def __init__(self, hidden, normal, h_to_z, topics, diversity_penalty, 
                 index, beta = 1, gamma = 1, iter1=10, iter2=20, sample = 20):
        # h_to_z will output probabilities over topics
        super(NSSTM, self).__init__(hidden, normal, h_to_z, topics)
        self.beta = beta
        self.gamma = gamma
        self.diversity_penalty = diversity_penalty
        self.index = index
        self.beta = beta
        self.gamma = gamma
        self.iter1 = iter1
        self.iter2 = iter2
        self.sample = sample
    def forward(self, x, n_sample=1, epoch = 0):
        stat = super(NSSTM, self).forward(x, n_sample)
        loss = stat['rec_loss'] + stat['kld']
        #penalty is mean - variance standard deviation
        if self.index == [] or epoch < self.iter1:
            self.ppenalty = 0
        else:
            self.ppenalty = negative_sampling_prior(self.topics.get_topics(),self.index, self.topics.embedding, epoch,
                                                   self.beta, self.gamma, self.iter2, self.sample)
        dpenalty, _, _ = topic_covariance_penalty(self.topics.topic_emb)
        stat.update({
            #loss add some penalty
            'loss': loss + self.ppenalty  + dpenalty * self.diversity_penalty,
            'penalty': self.ppenalty  + dpenalty * self.diversity_penalty,
            'prior': self.ppenalty
        })

        return stat
    
def optimal_transport_prior(softmax_top,  ind, index, embedding, epoch,
                            lambda_sh = 30, beta = 1, gamma = 1,
                            iter2 = 20, sample = 20):
    """ add prior as a semi-supervised loss
    
    parameters
    ----------
    softmax_top: softmax results from decoder
    index: list: a list of list with number as index
    embedding: numpy array, word embedding trained by spherical word embeddings
    beta: float, weights for prior loss
    gamma: float, weights for negative sampling
    iter2: int, how many epochs to train for third phase
    sample: int, sample number
    lambda_sh: low means high entrophy
    
    Returns:
    --------
    int
        loss functions
    
    """
    
    m = - torch.log(softmax_top + 1e-12)
    loss = torch.cat([m[:, i].mean(axis = 1).reshape(1, -1) for i in index])
    #print(loss.shape)
    b = torch.ones(loss.shape[1]).cuda() 
    a = torch.ones(loss.shape[0]).cuda()
    #negative sampling
    all_index = [j for i in index for j in i]
    logs = -m
    val = logs[:, all_index].max(axis=1).values
    #logns = torch.log(1 - softmax_top + 1e-6)
    nsll = 0
    psll = 0
    for inde, i in enumerate(index):
        
        sg = torch.argmax(logs[:, i].mean(axis=1) - val)
        #print(sm.shape, inde, ind, sg)
        psll +=  (- logs[sg, ind[inde]] ).mean()  
        #sample weights
        if epoch > iter2:
            ns = torch.topk(logs[sg], sample).indices
            #print(ns)
            ns = [int(i.cpu().numpy()) for i in ns if i >= 0 and i < logs.shape[1]]
            #print(ns)

            #print(ns)
            prob = 1 - (embedding.weight[ns] @ embedding.weight[i].T).max(axis = 1).values
            prob = torch.clamp(prob, max=1, min=0)
            samples = torch.bernoulli(prob)

            if 0 ==  samples.sum():
                nsl = 0
            else:
                negative_samples = [i for i, j in zip(ns, samples) if j == 1]
                nsl = logs[sg, negative_samples].mean()        
            nsll += nsl
    if epoch > iter2:
        return torch.Tensor([beta*sinkhorn_torch(loss, a, b, lambda_sh).sum() , gamma*(nsll), psll])
    else: 
        return torch.Tensor([beta*sinkhorn_torch(loss, a, b, lambda_sh).sum() , 0, psll])
        
class OTETM(NTM):
    """
    A class used for negative sampling based semi-supervised topic modeling
    
    Attributes
    ----------
    beta
    gamma
    diversity_penalty: coefficients for diversity penalty
    index: a list of index of keywords
    sample
    
    
    Methods:
    --------
    forward(logit)
        a disctionary of loss function
    
    
    
    """
    def __init__(self, hidden, normal, h_to_z, topics, diversity_penalty, index, 
                 slowstart = 200, beta = 1, gamma = 1, lambda_sh = 20, iter1 = 10, iter2 = 20, sample = 20):
        # h_to_z will output probabilities over topics
        super(OTETM, self).__init__(hidden, normal, h_to_z, topics)
        self.diversity_penalty = diversity_penalty
        self.index = index
        self.lambda_sh = lambda_sh
        self.beta = beta
        self.gamma = gamma
        self.iter1 = iter1
        self.iter2 = iter2
        self.sample = sample
        self.sm = (self.topics.embedding.weight[index, :] @ self.topics.embedding.weight.T).max(1).values
        self.ind = torch.topk(self.sm, 10).indices.cuda()
        self.val = torch.topk(self.sm, 10).values.cuda()
    def forward(self, x, n_sample=1, epoch = 0, slowstart = 200):
        stat = super(OTETM, self).forward(x, n_sample)
        loss = stat['rec_loss'] +  stat['kld']
        #penalty is mean - variance standard deviation
        if self.index == [] or epoch < self.iter1:
            self.ppenalty = torch.Tensor([0, 0, 0])
        else:
            self.ppenalty = optimal_transport_prior(self.topics.get_topics(), self.ind, self.index, self.topics.embedding, epoch,
                                                    self.lambda_sh, self.beta, self.gamma, self.iter2, self.sample)
        dpenalty, _, _ = topic_covariance_penalty(self.topics.topic_emb)
        stat.update({
            #loss add some penalty
            'loss': loss + torch.sum(self.ppenalty[:2]), # + dpenalty * self.diversity_penalty,
            'penalty': self.ppenalty, #+ dpenalty * self.diversity_penalty,
            'prior': self.ppenalty
        })

        return stat

class NormalParameter(nn.Module):
    def __init__(self, in_features, out_features):
        super(NormalParameter, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Linear(in_features, out_features)
        self.log_sigma = nn.Linear(in_features, out_features)
        self.reset_parameters()

    def forward(self, h):
        return self.mu(h), self.log_sigma(h)

    def reset_parameters(self):
        init.zeros_(self.log_sigma.weight)
        init.zeros_(self.log_sigma.bias)



class Sequential(nn.Sequential):
    """Wrapper for torch.nn.Sequential."""
    def __init__(self, args):
        super(Sequential, self).__init__(args)


def get_mlp(features, activate):
    """features: mlp size of each layer, append activation in each layer except for the first layer."""
    if isinstance(activate, str):
        activate = getattr(nn, activate)
    layers = []
    for in_f, out_f in zip(features[:-1], features[1:]):
        layers.append(nn.Linear(in_f, out_f))
        layers.append(activate())
    return nn.Sequential(*layers)

class Reshape(nn.Module):
    def __init__(self, args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        #print(self.shape)
        shape = (x.shape[0], ) + self.shape
        return torch.reshape(x, shape)
    

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        if len(input) == 1:
            return input[0]
        return input


class Topics(nn.Module):
    def __init__(self, k, vocab_size, bias=True):
        super(Topics, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.topic = nn.Linear(k, vocab_size, bias=bias)

    def forward(self, logit):
        # return the log_prob of vocab distribution
        return torch.log_softmax(self.topic(logit), dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic.weight.data.transpose(0, 1), dim=-1)

    def get_topic_word_logit(self):
        """topic x V.
        Return the logits instead of probability distribution
        """
        return self.topic.weight.transpose(0, 1)


    def get_topics(self):
        return self.topics.get_topics()
