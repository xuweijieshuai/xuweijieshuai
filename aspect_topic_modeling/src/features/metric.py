import numpy as np
from collections import defaultdict
from collections import Counter
from collections import defaultdict
#diversity
def diversity(topics):
    div = []
    for i in range(len(topics)):
        group1 = topics[i]
        group2 = [word  for topic in topics[:i] + topics[i+1:] for word in topic]
        div.append(len([key for key in group1 if key not in group2])/len(group1))
    return div


def get_document_frequency(data, wi, w2ind, wj=None):
    index1 = w2ind[wi]
    if wj is None:
        return data[:, index1].sum()
    index2 = w2ind[wj]
    return data[:, index2].sum(), (data[:, index1] * data[:, index2]  >= 1).astype(int).sum()


def get_topic_coherence(data, topics, w2ind):
    D = data.shape[0] ## number of docs...data is list of documents
    #print('D: ', D)
    TC = []
    num_topics = len(topics)
    for k in range(num_topics):
        #print('k: {}/{}'.format(k, num_topics))

        top_words = topics[k]
        TC_k = 0
        counter = 0
        for i, word in enumerate(top_words):
            # get D(w_i)
            D_wi = get_document_frequency(data, word, w2ind)
            j = i + 1
            tmp = 0
            while j < len(top_words) and j > i:
                # get D(w_j) and D(w_i, w_j)
                D_wj, D_wi_wj = get_document_frequency(data, word, w2ind, top_words[j])
                # get f(w_i, w_j)
                if D_wi_wj == 0:
                    f_wi_wj = -1
                else:
                    f_wi_wj = -1 + ( np.log(D_wi) + np.log(D_wj)  - 2.0 * np.log(D) ) / ( np.log(D_wi_wj) - np.log(D) )
                # update tmp:
                tmp += f_wi_wj
                j += 1
                counter += 1
            TC_k += tmp 
        TC.append(TC_k)
#     print('counter: ', counter)
#     print('num topics: ', len(TC))
    TC = np.mean(TC) / counter
    return TC

def top_purity(labels_true, labels_pred):
    pred = np.unique(labels_pred)
    d = defaultdict(list)
    for i, j in zip(labels_pred, labels_true):
        d[i].append(j)
    ct = []
    for i in d:
        ct += [Counter(d[i]).most_common(1)[0][1]]
        #print(Counter(d[i]).most_common(1)[0], i, len(d[i]))
    print(ct)
    return np.sum(ct)/len(labels_pred)
        