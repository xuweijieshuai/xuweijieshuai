import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
import pandas as pd
from collections import Counter
from stop_words import get_stop_words
import swifter
from scipy.sparse import csr_matrix
import numpy as np
import gc
import torch
stop_words = get_stop_words('en')
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.corpus import wordnet

def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def remove_stopWords(s):
    '''For removing stop words
    '''
    if not isinstance(s, str):
        return []
    s = str.lower(s)
    s = " ".join(re.findall('[\w]+',s))
    #s = " ".join(re.findall('[\w]+',s))
    s = [lemmatizer.lemmatize(w[0], get_wordnet_pos(w[1])) for w in nltk.pos_tag(nltk.word_tokenize(s)) if w[0] != 'p']
    #s = ' '.join(s)
    
    return(s)


def get_emb( vec_file):
        """
        get embedding of sphere topic modeling files
        """
        f = open(vec_file, 'r', encoding = "ISO-8859-1")
        tmp = f.readlines()
        contents = tmp[1:]
        doc_emb = {}

        for i, content in enumerate(contents):
            content = content.strip()
            tokens = content.split(' ')
            vec = tokens[1:]
            vec = [float(ele) for ele in vec]
            doc_emb[tokens[0]] = np.array(vec)

        return doc_emb
    
def generate_emb(vec_file, common_words, emb_size = 50):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # I have not considered the update of vec_file
        doc_emb = get_emb(vec_file)
        vocab_size = len(common_words)
        
        embeddings = np.zeros((vocab_size, emb_size))
        words_found = 0
        for i, word in enumerate(common_words):
            try:
                embeddings[i] = doc_emb[word]
                words_found += 1
            except KeyError:
                embeddings[i] = np.random.normal(scale=0.6, size=(emb_size,))
        embeddings = torch.from_numpy(embeddings).to(device)

        return embeddings
    
def train(model, X, batch_size, epoch, optimizer):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    total_nll = 0.0
    total_kld = 0.0
    total_words = 0
    total_penalty = 0.0
    #size = epoch_size * batch_size
    indices = torch.randperm(X.shape[0])
    indices = torch.split(indices, batch_size)
    #print(indices)
    length = len(indices)
    for idx, ind in enumerate(indices):
        data_batch = torch.from_numpy(X[ind].toarray()).float().to(device)
        
        d = model(x = data_batch, epoch = epoch)
            
        
        
        total_nll += d['rec_loss'].sum().item() / batch_size
        total_kld += d['kld'].sum().item() / batch_size  
        #total_penalty += d['penalty'].sum().item() / batch_size  
#         if i < 3:
#             loss = d['minus_elbo']
#         else:
        loss = d['loss']

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

    print(total_nll/length, total_kld/length)

def get_common_words(common_words_ct, ct = 40):
        #         if self.common_words == None:
        common_words = [x for x in common_words_ct if (not (x.isdigit() or x[0] == '-' and x[1:].isdigit())) \
                        and (x not in stop_words) and common_words_ct[x] >= ct and len(x) > 2 \
                        and (x not in ['just', 'will', 'may', 'please', 'need', 'one', 'can', 'wa', 'com', 'also']) 
                       ]
        return common_words

    
def generate_bow(df, common_words, bow_length = None):
        # tract_df = df[df['dt_ky'] > time]
        index_num = df['index_num'].values.tolist()
        if not bow_length:
            length = len(common_words)
        else:
            length = bow_length
        matrix = []
        indices = []
        for ind, i in enumerate(index_num):

            l = [0] * length
            for j in i:
                l[j] += 1
            if sum(l) != 0:
                matrix.append(l)
                indices.append(ind)

        X = csr_matrix(matrix)
        return X, indices



def kld_normal(mu, log_sigma):
    """KL divergence to standard normal distribution.
    mu: batch_size x dim
    log_sigma: batch_size x dim
    """
    #normal distribution KL divergence of two gaussian
    #https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
    return -0.5 * (1 - mu ** 2 + 2 * log_sigma - torch.exp(2 * log_sigma)).sum(dim=-1)


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


class EmbTopic(nn.Module):
    def __init__(self, embedding, k):
        super(EmbTopic, self).__init__()
        self.embedding = embedding
        n_vocab, topic_dim = embedding.weight.size()
        self.k = k
        self.topic_emb = nn.Parameter(torch.Tensor(k, topic_dim))
        self.reset_parameters()

    def forward(self, logit):
        # return the log_prob of vocab distribution
        logit = (logit @ self.topic_emb) @ self.embedding.weight.transpose(0, 1)
        return torch.log_softmax(logit, dim=-1)

    def get_topics(self):
        return torch.softmax(self.topic_emb @ self.embedding.weight.transpose(0, 1), dim=-1)

    def reset_parameters(self):
        init.normal_(self.topic_emb)
        # init.kaiming_uniform_(self.topic_emb, a=math.sqrt(5))
        init.normal_(self.embedding.weight, std=0.01)

    def extra_repr(self):
        k, d = self.topic_emb.size()
        return 'topic_emb: Parameter({}, {})'.format(k, d)


# class ETopic(nn.Module):
#     def __init__(self, embedding, k):
#         super(ETopic, self).__init__()
#         n_vocab, topic_dim = embedding.weight.size()
#         self.embedding = nn.Parameter(torch.Tensor(n_vocab, topic_dim))
#         self.k = k
#         self.topic_emb = nn.Parameter(torch.Tensor(k, topic_dim))
#         self.reset_parameters()
#
#     def forward(self, logit):
#         # return the log_prob of vocab distribution
#         logit = (logit @ self.topic_emb) @ self.embedding.transpose(0, 1)
#         return torch.log_softmax(logit, dim=-1)
#
#     def get_topics(self):
#         return torch.softmax(self.topic_emb @ self.embedding.transpose(0, 1), dim=-1)
#
#     def reset_parameters(self):
#         init.normal_(self.topic_emb)
#         # init.normal_(self.topic_emb, std=0.01)
#         init.normal_(self.embedding, std=0.01)
#
#     def extra_repr(self):
#         k, d = self.topic_emb.size()
#         return 'topic_emb: Parameter({}, {})\nembedding: Parameter({}, {})'.format(k, d, self.embedding.size(0),
#                                                                                    self.embedding.size(1))


class ScaleTopic(nn.Module):
    def __init__(self, k, vocab_size, bias=True, logit_importance=True, s=2):
        super(ScaleTopic, self).__init__()
        self.k = k
        self.vocab_size = vocab_size
        self.importance = nn.Parameter(torch.Tensor(k, vocab_size))
        if bias:
            self.register_parameter('bias', nn.Parameter(torch.Tensor(1, vocab_size)))
        self.scale = nn.Parameter(torch.Tensor(1, vocab_size))
        self.s = s
        self.logit_importance = logit_importance
        self.reset_parameters()

    def forward(self, logit):
        scale = torch.sigmoid(self.scale) * self.s
        if self.logit_importance:
            topics = self.importance * scale
        else:
            topics = torch.softmax(self.importance, dim=-1) * scale

        r = logit @ topics
        if hasattr(self, 'bias'):
            r = r + self.bias

        return torch.log_softmax(r, dim=-1)

    def get_topics(self):
        return torch.softmax(self.importance, dim=-1)

    def get_topic_word_logit(self):
        return self.importance

    def reset_parameters(self):
        init.kaiming_uniform_(self.importance, a=math.sqrt(5))
        init.zeros_(self.scale)
        if hasattr(self, 'bias'):
            init.zeros_(self.bias)


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
