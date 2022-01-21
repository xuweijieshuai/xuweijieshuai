import sys
#change to any directory you have to store the repo
sys.path.insert(1, '/home/ec2-user/SageMaker/github/aspect_topic_modeling')
import pandas as pd
from collections import Counter
from stop_words import get_stop_words
import swifter
from scipy.sparse import csr_matrix, save_npz
import numpy as np
import gc
import torch
stop_words = get_stop_words('en')
import re
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
from nltk.corpus import wordnet
from src.features.metric import diversity, get_topic_coherence
import torch
from torch.nn.functional import normalize
import pickle 
import torch.nn.functional as F 
import numpy as np 
from sklearn.metrics import f1_score
from torch import nn, optim
from models.NVDM import topic_covariance_penalty, sinkhorn_torch, NTM, negative_sampling_prior, optimal_transport_prior,  NormalParameter, get_mlp, EmbTopic, NSSTM, OTETM
from src.models.utils import get_wordnet_pos, remove_stopWords, get_emb, generate_emb, train, kld_normal, get_common_words, generate_bow
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
#nltk.download('wordnet')
#nltk.download('omw-1.4')
from sklearn.preprocessing import normalize
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
#data import
print('hehe')
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
data = 'agnews'
if data == 'agnews':
    news = pd.read_csv('https://raw.githubusercontent.com/yumeng5/WeSTClass/master/agnews/dataset.csv', 
                       error_bad_lines=False,
                       names = ['class', 'title', 'description'])
    news['text'] = news.apply(lambda x: ' '.join(remove_stopWords(x['title'] + x['description'])), axis=1)
    news['clean_text']  = news.apply(lambda x: x['text'].split(), axis = 1)
    #get clean data
    print('hehehe')
    common_words_ct = Counter([j for i in news['clean_text'].values for j in i])
    common_words = get_common_words(common_words_ct, ct = 200)
    word_track = {i: ind for ind, i in enumerate(common_words)}
    index_track = {ind: i for ind, i in enumerate(common_words)}
    news['index_num'] = news.apply(
                lambda x: [word_track[i] for i in x['clean_text'] if i in word_track], axis=1)

    #change it to any location you save your embeddings
    vec = '/home/ec2-user/SageMaker/ORMCorpVoatp/ormcorpvoatp/ormcorpvoatp/data/Spherical-Text-Embedding/datasets/agnews/jose.txt'
    embed = generate_emb(vec, common_words).cpu()
    X, indices = generate_bow(df = news, common_words = common_words)
    seed_topic_list  = [['government', 'military', 'war'],
     ['basketball', 'football', 'athlete'],
     ['stock', 'market', 'industry'],
     ['computer', 'telescope', 'software']]
    labels = [[word_track[j] for j in i] for i in seed_topic_list ]
    label1 = labels + [[1317, 2216,  676]]
    iter1, iter2, epochs, dif = 10, 20, 20, 1
if data == 'bibi':
    news = pd.read_csv('corpusbibi.tsv',  sep = '\t',
                       error_bad_lines=False,
                       names = ['text', 'train', 'class'])
    #change it to any location you save your embeddings
    news['clean_text']  = news.apply(lambda x: x['text'].split(), axis = 1)
    #get clean data
    common_words_ct = Counter([j for i in news['clean_text'].values for j in i])
    common_words = get_common_words(common_words_ct, ct = 50) #this vocab
    word_track = {i: ind for ind, i in enumerate(common_words)} #word dict
    index_track = {ind: i for ind, i in enumerate(common_words)} 
    news['index_num'] = news.apply(
                lambda x: [word_track[i] for i in x['clean_text'] if i in word_track], axis=1)

    vec = '/home/ec2-user/SageMaker/ORMCorpVoatp/ormcorpvoatp/ormcorpvoatp/data/Spherical-Text-Embedding/datasets/bibi/jose.txt'
    embed = generate_emb(vec, common_words).cpu()
    X, indices = generate_bow(df = news, common_words = common_words)
    seed_topic_list = seed_topic_list = [['database', 'query', 'information'], 
                                         ['learn', 'network', 'neural'],
                                         ['image', 'video', 'detection'], 
                                         ['mining', 'cluster', 'pattern']]
    labels = [[word_track[j] for j in i] for i in seed_topic_list ]
    label1 = labels + [[975, 453, 785]]
    iter1, iter2, epochs, dif = 10, 20, 20, 0
if data == '20News':
    news = pd.read_csv('corpus20news.tsv',  sep = '\t',
                       error_bad_lines=False,
                       names = ['text', 'train', 'name', 'class', 'index'])
    news['clean_text']  = news.apply(lambda x: x['text'].split(), axis = 1)
    #get clean data
    common_words_ct = Counter([j for i in news['clean_text'].values for j in i])
    common_words = get_common_words(common_words_ct, ct = 100)
    word_track = {i: ind for ind, i in enumerate(common_words)}
    index_track = {ind: i for ind, i in enumerate(common_words)}
    news['index_num'] = news.swifter.apply(
                lambda x: [word_track[i] for i in x['clean_text'] if i in word_track], axis=1)
    #change it to any location you save your embeddings
    vec = '/home/ec2-user/SageMaker/ORMCorpVoatp/ormcorpvoatp/ormcorpvoatp/data/Spherical-Text-Embedding/datasets/20news4/jose.txt'
    embed = generate_emb(vec, common_words).cpu()
    X, indices = generate_bow(df = news, common_words = common_words)
    seed_topic_list = [['faith','accept','world'], ['evidence','religion','belief'], 
                    ['algorithm','information','problem'], ['earth','solar','satellite']]

    labels = [[word_track[j] for j in i] for i in seed_topic_list]
    #labels = [[word_track[j] for j in i] for i in seed_topic_list ]
    label1 = labels + [[351, 122, 200]]
    iter1, iter2, epochs, dif = 100, 200, 200, 0
    
def optimal_transport_prior(softmax_top,  index, embedding, epoch,
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
    #psll = 0
    for inde, i in enumerate(index):
        
        sg = torch.argmax(logs[:, i].mean(axis=1) - val)
        #print(sm.shape, inde, ind, sg)
        #psll +=  (- sm[inde] * logs[sg, ind[inde]] ).mean()  
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
        return beta*sinkhorn_torch(loss, a, b, lambda_sh).sum() + gamma*(nsll)
    else: 
        return beta*sinkhorn_torch(loss, a, b, lambda_sh).sum()
        
def train(model, X, batch_size, epoch, optimizer, scheduler):
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
        total_penalty += d['prior']  
#         if i < 3:
#             loss = d['minus_elbo']
#         else:
        loss = d['loss']

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
    print(total_nll/length, total_kld/length, total_penalty/length)
     

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
        self.fc_mean = nn.Linear(64, 5)
        self.fc_var = nn.Linear(64, 1)
        #self.dirichlet = torch.distributions.dirichlet.Dirichlet((torch.ones(self.topics.k)/self.topics.k).cuda())
    def forward(self, x, n_sample=1):
        h = self.hidden(x)
        h = self.drop(h)
        z_mean = self.fc_mean(h)
        z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
        # the `+ 1` prevent collapsing behaviors
        z_var = F.softplus(self.fc_var(h)) + 1
        
        q_z = VonMisesFisher(z_mean, z_var)
        p_z = HypersphericalUniform(4, device=device)
        kld = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
        #print(q_z)
        #mu, log_sigma = self.normal(h)
        #identify how far it is away from normal distribution
        
        #print(kld.shape)
        rec_loss = 0
        for i in range(n_sample):
            #reparametrician trick
            z = q_z.rsample()
            #z = nn.Softmax()(z)
            #decode
            #print(z)
            
            z = self.h_to_z(10 * z)
            self.output = z
            #print(z)
            
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
        #self.sm = (self.topics.embedding.weight[index, :] @ self.topics.embedding.weight.T).max(1).values
        #self.ind = torch.topk(self.sm, 10).indices.cuda()
        #self.val = torch.topk(self.sm, 10).values.cuda()
    def forward(self, x, n_sample=1, epoch = 0, slowstart = 200):
        stat = super(OTETM, self).forward(x, n_sample)
        loss = stat['loss']
        #penalty is mean - variance standard deviation
        if self.index == [] or epoch < self.iter1:
            self.ppenalty = 0
        else:
            self.ppenalty = optimal_transport_prior(self.topics.get_topics(), self.index, self.topics.embedding, epoch,
                                                    self.lambda_sh, self.beta, self.gamma, self.iter2, self.sample)
        dpenalty, _, _ = topic_covariance_penalty(self.topics.topic_emb)
        stat.update({
            #loss add some penalty
            'loss': loss + self.ppenalty + dpenalty * self.diversity_penalty,
            'penalty': self.ppenalty + dpenalty * self.diversity_penalty,
            'prior': self.ppenalty
        })

        return stat
    
numb_embeddings = len(seed_topic_list) + 1
torch.cuda.empty_cache()
gc.collect()
#X = normalize(X, norm='l1', axis=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, label1)
result = []
beta = 1
for rand in range(10):
    for gamma in [0.25]:
        for diversity_p in [0]:
            torch.cuda.empty_cache()
            gc.collect()

            numb_embeddings = len(seed_topic_list) + 1
            hidden = get_mlp([X.shape[1], 64], nn.ReLU)
            normal = NormalParameter(64, numb_embeddings)
            h_to_z = nn.Softmax()
            embedding = nn.Embedding(X.shape[1], 50)
            # p1d = (0, 0, 0, 10000 - company1.embeddings.shape[0]) # pad last dim by 1 on each side
            # out = F.pad(company1.embeddings, p1d, "constant", 0)  # effectively zero padding

            embedding.weight = torch.nn.Parameter(torch.Tensor(embed.float()))
            embedding.weight.requires_grad=False
            topics = EmbTopic(embedding = embedding,
                              k = numb_embeddings, normalize = False)



            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


            model = OTETM(hidden = hidden,
                        normal = normal,
                        h_to_z = h_to_z,
                        topics = topics,
                        diversity_penalty = 0, 
                        index = label1,
                        iter1 = iter1,
                        iter2 = iter2,
                        beta = 0.25,
                        gamma = 0.25,
                        lambda_sh = 1

                        ).to(device).float()
            # larger hidden size make topics more diverse
            #num_docs_train = 996318
            batch_size = 256
            optimizer = optim.Adam(model.parameters(), 
                                   lr=0.002, 
                                   weight_decay=1.2e-6)


            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=int(X.shape[0]/batch_size) + 1, epochs=epochs)

            
            for epoch in range(epochs):
                train(model, X,  batch_size, epoch, optimizer, scheduler)
            #prior(softmax_top, indexes)
            emb = model.topics.get_topics().cpu().detach().numpy()
            topics =  [[index_track[ind] for ind in np.argsort(emb[i])[::-1][:10] ] for i in range(numb_embeddings)]

            #visualize keywords
            mapping = [emb[:, i].mean(1).argmax() for i in labels]

            data_batch = torch.from_numpy(X.toarray()).float()
            model.cpu()
            z = model.hidden(data_batch)
            z_mean = model.fc_mean(z)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            #z_var = F.softplus(model.fc_var(z)) + 1

            #q_z = VonMisesFisher(z_mean, z_var)
            #p_z = HypersphericalUniform(4, device=device)
            #z = q_z.rsample()
            z = model.h_to_z(z_mean)
            zz = torch.stack([z[:, i] for i in mapping]).T
            zz = zz.cpu().detach().numpy()
            # zz[:,0] = zz[:,0] * np.sqrt(0.219)
            # zz[:,1] = zz[:,1]* np.sqrt(0.3826)
            # zz[:,2] = zz[:,2]* np.sqrt(0.3106)
            # zz[:,3] = zz[:,3] * np.sqrt(0.08724) 
            y_pred = zz.argmax(1)
            y_true = news['class'].iloc[indices].values - dif
            accuracy = np.sum(y_pred == y_true)/news.shape[0]
            coherence_score = get_topic_coherence(X.toarray(), seed_topic_list, word_track)
            diversity_score = np.mean(diversity(topics))
            macro = f1_score(y_true, y_pred, average='macro')
            micro = f1_score(y_true, y_pred, average='micro')
            print('The accuracy of the model is ' + str(accuracy))
            print('The Diversity of the model is ' + str(diversity_score) )
            print('The F1 macro score of the model is ' + str(macro))
            enc = OneHotEncoder(handle_unknown='ignore')
            y_real = enc.fit_transform(y_true.reshape(-1, 1)).toarray()
            aucroc = roc_auc_score(y_real, zz, multi_class = 'ovo')  
            print(topics)
            print(accuracy, diversity_score, macro, aucroc)
            result += [[accuracy, diversity_score, macro, aucroc]]

            pd.DataFrame(result).to_csv('otstm_result' + data +  'vmf1.csv')
