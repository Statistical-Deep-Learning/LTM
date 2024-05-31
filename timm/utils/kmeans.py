import numpy as np
import faiss
import torch
import torch.nn as nn


def run_kmeans(x, num_cluster, deviceID=0):

    print('performing kmeans clustering')
    results = {'im2cluster':[],'centroids':[],'density':[]}
    # intialize faiss clustering parameters
    d = x.shape[1]
    k = int(num_cluster)
    clus = faiss.Clustering(d, k)
    clus.verbose = True
    clus.niter = 20
    clus.nredo = 5
    # clus.seed = seed
    clus.max_points_per_centroid = 1000
    clus.min_points_per_centroid = 10

    res = faiss.StandardGpuResources()
    cfg = faiss.GpuIndexFlatConfig()
    cfg.useFloat16 = False
    cfg.device = deviceID   
    index = faiss.GpuIndexFlatL2(res, d, cfg)  

    clus.train(x, index)   

    D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
    im2cluster = [int(n[0]) for n in I]
    
    # get cluster centroids
    centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)
    
    # sample-to-centroid distances for each cluster 
    Dcluster = [[] for c in range(k)]          
    for im,i in enumerate(im2cluster):
        Dcluster[i].append(D[im][0])
    
    # concentration estimation (phi)        
    # density = np.zeros(k)
    # for i,dist in enumerate(Dcluster):
    #     if len(dist)>1:
    #         d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)            
    #         density[i] = d     
            
    # if cluster only has one point, use the max to estimate its concentration        
    # dmax = density.max()
    # for i,dist in enumerate(Dcluster):
    #     if len(dist)<=1:
    #         density[i] = dmax 

    # density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
    # density = temperature*density/density.mean()  #scale the mean to temperature 
    
    # convert to cuda Tensors for broadcast
    centroids = torch.Tensor(centroids).cuda(deviceID)
    centroids = nn.functional.normalize(centroids, p=2, dim=1)    

    im2cluster = torch.LongTensor(im2cluster).cuda(deviceID)               
    # density = torch.Tensor(density).cuda(deviceID)
    
    results['centroids'].append(centroids)
    # results['density'].append(density)
    results['im2cluster'].append(im2cluster)    
    
    return results
