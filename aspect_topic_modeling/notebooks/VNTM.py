import sys
sys.path.insert(1, '/home/ec2-user/SageMaker/github/aspect_topic_modeling')
import torch
from torch.nn.functional import normalize
import pickle 
import torch.nn.functional as F 
import numpy as np 
from sklearn.metrics import f1_score
from torch import nn, optim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import pickle
import random
from scipy import sparse
import itertools
from scipy.io import savemat, loadmat
import re
import string
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import pandas as pd
import swifter
from src.models.utils import remove_stopWords
from src.models.utils import get_wordnet_pos, remove_stopWords, get_emb, generate_emb, train, kld_normal, get_common_words, generate_bow
from collections import Counter
from models.NVDM import VNTM, topic_covariance_penalty, sinkhorn_torch, NTM, negative_sampling_prior, optimal_transport_prior,  NormalParameter, get_mlp, EmbTopic, NSSTM, OTETM
from src.models.utils import get_wordnet_pos, remove_stopWords, get_emb, generate_emb, train, kld_normal, get_common_words
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
from src.features.metric import diversity, get_topic_coherence,top_purity
from sklearn.cluster import KMeans
from collections import Counter
from sklearn import metrics
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.matutils import Sparse2Corpus


def evaluate(topics, X, z, labels):
    result = []
    result += sorted(diversity(topics))


    labels_pred = torch.argmax(z, 1).numpy()
    labels_true = labels
    #coherence_score = get_topic_coherence(X.toarray(), topics, word_track)
    kmeans = KMeans(n_clusters=numb_embeddings, random_state=0).fit(z.detach().numpy())
    result += [top_purity(labels_true, labels_pred), metrics.normalized_mutual_info_score(labels_true, labels_pred), top_purity(labels_true, kmeans.labels_), metrics.normalized_mutual_info_score(labels_true, kmeans.labels_)]  
    corpus = Sparse2Corpus(X, documents_columns=False)
    #decoder_weight = self.autoencoder.decoder.linear.weight.detach().cpu()
    id2word = {index: str(index) for index in range(X.shape[1])}
    texts = df.apply(lambda x:[str(i) for i in x['index_num']], axis = 1).values
    cm = CoherenceModel(
                topics=topics,
                corpus=corpus,
                dictionary=Dictionary.from_corpus(corpus, id2word),
                coherence="u_mass",
            )
    result += [cm.get_coherence()]
    cm = CoherenceModel(
        topics=topics,
        texts = texts,
        corpus=corpus,
        dictionary=Dictionary.from_corpus(corpus, id2word),
        coherence='c_npmi',
    )

    result += sorted(cm.get_coherence_per_topic())

    cm = CoherenceModel(
        topics=topics,
        texts = texts,
        corpus=corpus,
        dictionary=Dictionary.from_corpus(corpus, id2word),
        coherence='c_v',
    )

    result += sorted(cm.get_coherence_per_topic())
    return result

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
        
        d = model(x = data_batch)
            
        
        
        total_nll += d['rec_loss'].sum().item() / batch_size
        total_kld += d['kld'].sum().item() / batch_size  
        #total_penalty += d['prior']  
#         if i < 3:
#             loss = d['minus_elbo']
#         else:
        loss = d['loss']

        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
        scheduler.step()
    print(total_nll/length, total_kld/length)
     
data = fetch_20newsgroups(subset = 'all')
df = pd.DataFrame()
df['class'] = data['target']
df['text'] = data['data']
df['text'] = df.apply(lambda x: ' '.join(remove_stopWords(x['text'])), axis=1)
df['clean_text']  = df.apply(lambda x: x['text'].split(), axis = 1)

common_words_ct = Counter([j for i in df['clean_text'].values for j in i])
common_words = get_common_words(common_words_ct, ct = 200)
word_track = {i: ind for ind, i in enumerate(common_words)}
index_track = {ind: i for ind, i in enumerate(common_words)}
df['index_num'] = df.apply(
            lambda x: [word_track[i] for i in x['clean_text'] if i in word_track], axis=1)
#change it to any location you save your embeddings
vec = '/home/ec2-user/SageMaker/ORMCorpVoatp/ormcorpvoatp/ormcorpvoatp/data/Spherical-Text-Embedding/datasets/20news/jose.txt'
embed = generate_emb(vec, common_words).cpu()
X, indices = generate_bow(df = df, common_words = common_words)
labels = df['class'].values
pdf = []
for iii in range(10):
    for penalty in [0.1, 0.5, 1, 2]:
                result = [penalty] 

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                numb_embeddings = 20
                layer = 64
                hidden = get_mlp([X.shape[1], layer], nn.ReLU)
                normal = NormalParameter(layer, numb_embeddings)
                h_to_z = nn.Softmax()
                embedding = nn.Embedding(X.shape[1], 50)
                # p1d = (0, 0, 0, 10000 - company1.embeddings.shape[0]) # pad last dim by 1 on each side
                # out = F.pad(company1.embeddings, p1d, "constant", 0)  # effectively zero padding

                embedding.weight = torch.nn.Parameter(torch.Tensor(embed.float()))
                embedding.weight.requires_grad=False
                topics = EmbTopic(embedding = embedding,
                                  k = numb_embeddings, normalize = False)



                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


                model = VNTM(hidden = hidden,
                            normal = normal,
                            h_to_z = h_to_z,
                            topics = topics,
                            layer = layer, 
                            top_number = numb_embeddings,
                            penalty = penalty
                            ).to(device).float()
                # larger hidden size make topics more diverse
                #num_docs_train = 996318
                batch_size = 256
                optimizer = optim.Adam(model.parameters(), 
                                       lr=0.002, 
                                       weight_decay=1.2e-6)

                epochs = 20
                scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.05, steps_per_epoch=int(X.shape[0]/batch_size) + 1, epochs=epochs)


                for epoch in range(epochs):
                    train(model, X,  batch_size, epoch, optimizer, scheduler)
                #Add Diversity
                emb = model.topics.get_topics().cpu().detach().numpy()
                topics =  [[str(ind) for ind in np.argsort(emb[i])[::-1][:25] ] for i in range(10)]
                    #Add purity . 
                data_batch = torch.from_numpy(X.toarray()).float()
                model.cpu()
                z = model.hidden(data_batch)
                z_mean = model.fc_mean(z)
                z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
                z = model.h_to_z(z_mean)

                result += evaluate(topics, X, z, labels)
                pdf += [result]
                pd.DataFrame(pdf).to_csv('vntm_result.csv')


