"""Distance-based geometry with sampled queries and complete reference sets."""
import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score


def geometry(x,labels):
    _,y = np.unique(labels,return_inverse=True)
    n,k = len(y),int(y.max()+1)
    if not 1<k<n:
        raise ValueError('Geometry requires between 2 and n-1 nonempty groups')
    counts = np.bincount(y)
    sums = np.zeros((k,x.shape[1])); np.add.at(sums,y,x)
    means = sums/counts[:,None]
    between = np.sum(counts[:,None]*(means-x.mean(0))**2)
    total = np.sum((x-x.mean(0))**2)
    return {'calinski_harabasz':float(calinski_harabasz_score(x,y)),
            'davies_bouldin':float(davies_bouldin_score(x,y)),
            'between_fraction':float(between/total) if total else 0.0,
            'classes':k,'minimum_class_size':int(counts.min()),'maximum_class_size':int(counts.max())}


def query_silhouettes(x,partitions,query_indices,chunk_size=32):
    """Exact Euclidean silhouette for each query against ALL reference tokens.

    Only the outer mean is estimated by uniform query sampling. Distances are
    shared across target, cluster and shuffled-target partitions. Self distance
    is zeroed, own-group divisor is n_c-1; singleton score follows sklearn: 0.
    """
    prep = {}
    for name,labels in partitions.items():
        _,y = np.unique(labels,return_inverse=True)
        counts = np.bincount(y)
        if not 1<len(counts)<len(y):
            raise ValueError('Invalid partition for silhouette')
        order = np.argsort(y,kind='stable')
        starts = np.r_[0,np.cumsum(counts)[:-1]]
        prep[name] = y,counts,order,starts
    result = {name:[] for name in prep}
    for start in range(0,len(query_indices),chunk_size):
        ix = query_indices[start:start+chunk_size]
        distances = cdist(x[ix],x,metric='euclidean')
        distances[np.arange(len(ix)),ix] = 0
        for name,(y,counts,order,starts) in prep.items():
            sums = np.add.reduceat(distances[:,order],starts,axis=1)
            own = y[ix]
            a = sums[np.arange(len(ix)),own]/np.maximum(counts[own]-1,1)
            means = sums/counts[None,:]
            means[np.arange(len(ix)),own] = np.inf
            b = means.min(1)
            score = np.divide(b-a,np.maximum(a,b),out=np.zeros_like(a),where=np.maximum(a,b)>0)
            score[counts[own]==1] = 0
            result[name].append(score)
    return {name:np.concatenate(values) for name,values in result.items()}
