# Required function for running this assignment
# Written by Mehdi Rezvandehy

from IPython.display import Image
from matplotlib import gridspec
from IPython.display import display, Math, Latex
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import PercentFormatter
from IPython.display import HTML
from matplotlib.ticker import FixedLocator, FixedFormatter
from sklearn.mixture import GaussianMixture
from nbconvert import HTMLExporter
from matplotlib.colors import ListedColormap, BoundaryNorm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
import time
#scikit-learn==0.24.2
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples


##############################################################
def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]),
                linewidths=1, colors='k')

##############################################################################

def kmean_eda(data, k_start=3,k_end=20,kint=1,n_init=10,minibatch=False, batch_size=10,silhouette=False):
    """
    Apply iterative k-mean prediction for diffirent k (numner of classes); calculating inertia and 
    time (minute) for each
    """
    time_c=[]
    inertia_c=[]
    n_clus=[]
    silhouette_c=[]
    for ik in range(k_start,k_end+1,kint):
        start_time = time.time()
        np.random.seed(42)
        if (minibatch):
            kmeans = MiniBatchKMeans(n_clusters=ik,batch_size=batch_size,n_init=n_init)
            kmeans.fit(data)
            if(silhouette):
                silhouette_c.append(silhouette_score(data, kmeans.labels_))    
        else:
            kmeans = KMeans(n_clusters=ik,n_init=n_init)
            kmeans.fit_predict(data)
            if(silhouette):
                silhouette_c.append(silhouette_score(data, kmeans.labels_))
        # Calculate running time in minute
        time_took = time.time() - start_time
        n_clus.append(ik)
        time_c.append(time_took)
        inertia_c.append(kmeans.inertia_)
    if(silhouette):    
        return n_clus,time_c,inertia_c, silhouette_c
    else:
        return n_clus,time_c,inertia_c

##############################################################################

def gmm_eda(data, k_start=3,k_end=20,kint=1,n_init=10,predict=False):
    """
    Apply iterative Gaussian Mixture model prediction for diffirent k (numner of classes); calculating silhouette and 
    time (minute) for each run
    """
    time_c=[]
    n_clus=[]
    silhouette_c=[]
    pred=[]
    for ik in range(k_start,k_end+1,kint):
        start_time = time.time()
        gmm = GaussianMixture(n_components=ik, n_init=n_init)
        gmm.fit(data)
        silhouette_c.append(silhouette_score(data, gmm.predict(data)))
        # Calculate running time in minute
        time_took = time.time() - start_time
        n_clus.append(ik)
        time_c.append(time_took)
        pred.append(gmm.predict(data))
    if(predict):  
        return n_clus,time_c, silhouette_c, pred
    else:
        return n_clus,time_c, silhouette_c       

##############################################################################

def wss_value(data, cluster_center):
    """
    Calculate WSS (Within Cluster Sums of Square difference)
    """
    val=np.zeros((len(data)))
    for i in range(len(data)):
        val[i]=min([np.sqrt(sum((j-data[i])**2)) for j in cluster_center])
    return sum(val)       

##############################################################################

def gmm_eda(data, k_start=3,k_end=20,kint=1,n_init=10,predict=False):
    """
    Apply iterative Gaussian Mixture model prediction for diffirent k (numner of classes); calculating silhouette and 
    time (minute) for each run
    """
    time_c=[]
    n_clus=[]
    silhouette_c=[]
    pred=[]
    for ik in range(k_start,k_end+1,kint):
        start_time = time.time()
        gmm = GaussianMixture(n_components=ik, n_init=n_init)
        gmm.fit(data)
        silhouette_c.append(silhouette_score(data, gmm.predict(data)))
        # Calculate running time in minute
        time_took = time.time() - start_time
        n_clus.append(ik)
        time_c.append(time_took)
        pred.append(gmm.predict(data))
    if(predict):  
        return n_clus,time_c, silhouette_c, pred
    else:
        return n_clus,time_c, silhouette_c      