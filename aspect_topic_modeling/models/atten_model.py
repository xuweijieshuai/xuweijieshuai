#-------------------------------------
#nn architecture and loss
#-------------------------------------
import torch
import torch.nn as nn
from torch.nn.functional import normalize
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MODEL_ATT_COMP(nn.Module):
    def __init__(self, d_word, d_key, d_value, n_topic, n, embeddings):
        super(MODEL_ATT_COMP, self).__init__()
        self.embeddings = nn.Embedding(len(embeddings), 200)
        self.embeddings.weight = torch.nn.Parameter(embeddings)
        self.embeddings.weight.requires_grad = False
        self.K = nn.Linear(d_word,d_key)
        self.Q = nn.Linear(d_word,d_key)
        self.V = nn.Linear(d_word,d_value)
        self.V2T = nn.Linear(d_value,n_topic)
        self.T2L = nn.Linear(n, 1)
        self.L2T = nn.Linear(1, n)
        self.soft1 = nn.Softmax(dim = 2)
        self.T2V = nn.Linear(n_topic,d_value)
        self.V2W = nn.Linear(d_value,d_word)
        self.sqrtdk = torch.tensor([d_key**0.5]).to(device)
        self.soft2 = nn.Softmax(dim = 1)
    
    def loss_max_margin_neg_sample(self, x):
        """Maximize word level embedding reconstruction with negative sampling
        """
        word_repre_x = normalize(self.word_repre, dim = 2) #batch n dvalue
        value_recon_x = normalize(self.value_recon, dim = 2) #batch n dvalue
        sim_matrix = torch.matmul(word_repre_x, value_recon_x.transpose(2,1)) #batch n n
        sim_x = torch.diagonal(sim_matrix, 0, 1, 2) #batch n 
        ns = torch.randperm(sim_x.shape[1]) # n 
        loss =  1 - sim_x + torch.diagonal(sim_matrix[:, ns], 0, 1, 2)
        loss = loss.mean(1)  #batch 
        return loss

    def loss_word_prediction_no_self(self, x):
        """Maximize word distribution reconstruction without given word
        """
        word_recon_no_self_normalized = normalize(self.word_recon_no_self, dim = 2) #batch n d_word
        x_normalized = normalize(x, dim = 2).transpose(2,1) #batch d_word n 
        sim_matrix = torch.matmul(word_recon_no_self_normalized, x_normalized) #batch n n
        return 1 - torch.diagonal(sim_matrix, 0, 1, 2).mean(1) #batch 

    def reconstruction_loss(self):
        """Sparsity/Entropy loss to make sure each word goes to 1 topic
        """
        distribution = self.topic
        return - torch.log(distribution) * distribution
    
    def sinkhorn_distance(self, lambda_sh = 10):
        """Make sure sentence goes to less topics and documents are uniformly distributed to each topics
        """
        d1, d2, d3 = self.topic.shape
        sentence_topic = self.topic.reshape(d1, d2)
        a = torch.ones(sentence_topic.shape[0]).to(device) * sentence_topic.shape[1] / sentence_topic.shape[0] #batch dimension
        b = torch.ones(sentence_topic.shape[1]).to(device)  #topics dimension 
        #print(a.shape, b.shape)
        return sinkhorn_torch( - torch.log(sentence_topic + 1e-6), a, b, lambda_sh).sum()
    
    
    def similarity_loss(self):
        """If word has high attention on another word, they should have similar topics distribution
        """
        d1, d2, d3 = self.att_weight.shape
        normal_weights = self.att_weight.reshape(-1, d3) # batch * n n
        #print(normal_weights.shape)
        samples = torch.multinomial(normal_weights, 1).reshape(-1) #batch * n
        normalize_weights = normalize(self.topic_weight, dim = 2)
        #print(normalize_weights.shape)
        topic_similarity = torch.matmul(normalize_weights, normalize_weights.transpose(1,2)).reshape(d1*d2, -1) #batch n n
        #print(topic_similarity.shape, samples.shape)
        return 1 - topic_similarity[torch.arange(topic_similarity.shape[0]), samples].reshape(d1, d2).mean(1) #batch n
         
    def word_topics(self):
        x = self.embeddings.weight
        self.soft2 = nn.Softmax(dim = 1)
        self.k = self.K(x).transpose(0,1) #d_key n 
        self.q = self.Q(x) #n d_key
        self.att_score = torch.matmul(self.q, self.k) #n n
        self.att_weight = self.soft2(self.att_score/self.sqrtdk) #n n, row sum = 1
        self.v = self.V(x) #n d_key
        self.word_repre = torch.matmul(self.att_weight, self.v) #batch n d_value
        self.topic_score = self.V2T(self.word_repre) #n n_topic
        self.word2topic = self.soft2(self.topic_score) #n n_topic, row sum = 1       
        return self.word2topic
    
    def forward(self,x):
        '''
        x: tensor, batch by n
        Output: a dictionary that contains different loss
        '''
        x = self.embeddings(x) #batch n d_word
        self.k = self.K(x).transpose(2,1) #batch d_key n 
        self.q = self.Q(x) #batch n d_key
        self.att_score = torch.matmul(self.q, self.k) #batch n n
        self.att_weight = self.soft1(self.att_score/self.sqrtdk) #batch n n, row sum = 1
        self.v = self.V(x) #batch n d_key
        self.word_repre = torch.matmul(self.att_weight, self.v) #batch n d_value
        self.topic_score = self.V2T(self.word_repre) #batch n n_topic
        self.topic_weight = self.soft1(self.topic_score) #batch n n_topic , row sum = 1
        #print(self.topic_weight.shape)
        self.topic = self.soft2(self.T2L(self.topic_weight.transpose(2,1))) #batch n_topic 1
        #print(self.topic.shape)
        self.topic_recon = self.L2T(self.topic).transpose(2,1) #batch  n n_topic
        #print(self.topic_recon.shape)
        self.value_recon = self.T2V(self.topic_recon) #batch n d_value
        self.word_recon = self.V2W(self.word_repre)#batch n d_word
        #no self computation, effectively masked
        #print(self.k.shape, self.att_score.shape, self.att_weight.shape, self.word_repre.shape, self.topic_weight.shape)
        self.att_score_no_self = self.att_score -  torch.diag(torch.zeros(self.att_score.shape[1])+torch.tensor(float('inf'))).to(device)#batch n n
        self.att_weight_no_self = self.soft1(self.att_score_no_self/self.sqrtdk) #batch n n 
        self.word_repre_no_self = torch.matmul(self.att_weight_no_self, self.v)#batch n d_key
        self.word_recon_no_self = self.V2W(self.word_repre_no_self) #batch n d_word
        word_pred_loss = self.loss_word_prediction_no_self(x).sum()
        margin_loss = self.loss_max_margin_neg_sample(x).sum()
        recon_loss = self.reconstruction_loss().sum()
        sim_loss = self.similarity_loss().sum()
        sinkhorn_loss = self.sinkhorn_distance()
        return {
            'loss' : word_pred_loss + margin_loss,
            'margin_loss': margin_loss,
            'word_loss': word_pred_loss,
            'reconstruct_loss': recon_loss,
            'similarity_loss': sim_loss,
            'sinkhorn_loss': sinkhorn_loss
            
        }