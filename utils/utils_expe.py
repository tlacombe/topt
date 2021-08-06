import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import tensorflow as tf
import networkx as nx
import scipy.signal as signal
import random

from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import KernelPCA
from sklearn.neighbors import KernelDensity
from sklearn.metrics import pairwise_distances


def write_st(st, path):
    with open(path,'w') as f:
        for (s,_) in st.get_filtration():
            f.write(' '.join([str(v) for v in s]))
            f.write('\n')


def recursive_insert(st, base_splx, splx, name_dict, filt):
    if len(splx) == 1:
        st.insert(base_splx + [name_dict[tuple(splx)]], filt)
    else:
        for idx in range(len(splx)):
            coface = splx[:idx] + splx[idx+1:]
            recursive_insert(st, base_splx + [name_dict[tuple(splx)]], coface, name_dict, max(filt, st.filtration([name_dict[tuple(coface)]])))


def barycentric_subdivision(st, list_splx=None):

    bary = gd.SimplexTree()
    bary_splx = {}

    splxs = st.get_filtration() if list_splx is None else list_splx
    count = 0
    for splx, f in splxs:
        bary.insert([count], f)
        bary_splx[tuple(splx)] = count
        count += 1

    for splx, f in st.get_filtration():
        if len(splx) == 1:
            continue
        else:
            recursive_insert(bary, [], splx, bary_splx, bary.filtration([bary_splx[tuple(splx)]]))

    return bary


def build_model(N, model, save=False, name='mapper'):

    stbase = gd.SimplexTree()
    stbase_obs = gd.SimplexTree()

    if model == 'circle':

        for i in range(N):
            stbase.insert([i,i+1])
        for i in range(N+1,2*N-1):
            stbase.insert([i,i+1])
        stbase.insert([0,N+1])
        stbase.insert([2*N-1,N])

        npts = stbase.num_vertices()
        assert npts == 2*N

        F1 = np.concatenate([np.arange(N+1), np.arange(1,N)])/N
        F2 = np.concatenate([np.concatenate(
        [1e-8+np.linspace(0,.5,int(N/3)),1e-6+np.linspace(.5,.05,int(N/3)),1e-7+np.linspace(.05,1,N+1-2*int(N/3))]),
                             np.concatenate(
        [1e-8+np.linspace(0,.3,int(N/4)),1e-6+np.linspace(.3,.6,int(N/3)),1e-7+np.linspace(.6,1,N-1-int(N/3)-int(N/4))])
                            ])
        F3 = np.concatenate([np.concatenate(
            [np.linspace(0,.3,int(N/3)),np.linspace(.3,.2,int(N/3)),np.linspace(.2,1,N+1-2*int(N/3))]),
             np.linspace(0,1,N-1)])
        F4 = np.concatenate([np.concatenate(
            [np.linspace(0,1,N+1),
             np.linspace(0,.4,int(N/3)),np.linspace(.4,.1,int(N/3)),np.linspace(.1,1,N-1-2*int(N/3))])])
        F = np.hstack([F1[:,np.newaxis]]) #,F2[:,np.newaxis],F3[:,np.newaxis],F4[:,np.newaxis]])

        Finit = F2 #np.concatenate([np.linspace(.5,1,int(N/2)),np.linspace(1,0,N),np.linspace(0,.5,N-int(N/2))])

    elif model == 'peaks':
        #print("peaks is for tobogan-TDA expe, N is harcoded")
        #Finit = np.array([0, 0.01, 0.02, 1, -1, 0.03])
        Finit = np.zeros(N)
        Finit[2 * np.arange(int(N/2))] = 2
        Finit = Finit - 1
        # Finit = Finit + 0.01 * np.random.rand(N)
        N = len(Finit)
        F1 = np.zeros(N)  # The target filtration (actually, we target its diagram, which is empty)
        F = np.hstack([F1[:, np.newaxis]])

        for i in range(N-1):
            stbase.insert([i, i+1])

        npts = stbase.num_vertices()
        assert npts == N

        # sh = 0.5
        #
        # F1 = np.zeros(N)  # The target filtration (actually, we target its diagram, which is empty)
        # #F1[-2] = 0.1
        # Finit = np.zeros(N)
        # Finit[:N-2] = sh * np.arange(N-2)
        # Finit[-2] = -sh
        # Finit[-1] = sh / 10
        # Finit[0] = -2
        # F = np.hstack([F1[:,np.newaxis]])
        #recommended epsilon for strata = 0.2

    elif model == 'Mapper':

        npts = 1000
        mapper_data = 'schic'
        n_boot = 10

        if mapper_data == 'synth':

            npts = 1000
            np.random.seed(0)

            angles = np.random.uniform(0,2*np.pi,size=[npts,1])
            circle = np.hstack([np.cos(angles),np.sin(angles)])
            noise_circle = np.random.uniform(-.2,.2,size=[npts,2])

            times = np.random.uniform(0,1,size=[npts,1])
            a, b = np.array([[-1,0]]), np.array([[-3,-.5]])
            branch = np.multiply(a,1-times) + np.multiply(b,times)
            noise_branch = np.random.uniform(-.4,.4,size=[npts,2])

            times = np.arange(0,1,.2)[:,None] #np.random.uniform(0,1,size=[20,1]) #[int(npts/100),1])
            a, b = np.array([[1.3,0]]), np.array([[2,-1]])
            small_branch = np.multiply(a,1-times) + np.multiply(b,times)

            X = np.vstack([noise_circle + circle, small_branch]) #, noise_branch + branch])
            params = {
                      'filters': X,
                      'filter_bnds': np.array([[-4,2],[-2,2]]),
                      'colors': X,
                      'resolutions': np.array([20,20]),
                      'gains': np.array([.4,.4]),
                      'mask': 0,
                      'clustering': AgglomerativeClustering(affinity='euclidean', distance_threshold=1, n_clusters=None)
                     }

        elif mapper_data == 'schic':

            SCC = np.load('scc_500k_5M_h1_chr-1.npy')
            X = KernelPCA(n_components=30, kernel='precomputed', eigen_solver='arpack').fit_transform(SCC)
            kde = KernelDensity(kernel='gaussian', bandwidth=.1).fit(X)
            #X = X[kde.score_samples(X) >= 34.5]

            params = {
                      'filters': X[:,:2],
                      'filter_bnds': np.array([[np.nan,np.nan],[np.nan,np.nan]]),
                      'colors': X[:,:2],
                      'resolutions': np.array([15,15]), #np.array([np.nan,np.nan]), #
                      'gains': np.array([.4,.4]),
                      'mask': 0,
                      'clustering': AgglomerativeClustering(affinity='euclidean', distance_threshold=2, n_clusters=None)
                     }
            #params = {
            #          'filters': X[:,1:2], 
            #          'filter_bnds': np.array([[np.nan,np.nan]]), 
            #          'colors': X[:,:2], 
            #          'resolutions': np.array([20]), #np.array([np.nan]),  
            #          'gains': np.array([.1]), 
            #          'mask': 0,
            #          'clustering': AgglomerativeClustering(affinity='euclidean', distance_threshold=.75, n_clusters=None)
            #         }

        plt.figure()
        plt.scatter(X[:,0], X[:,1], s=10) #, c=kde.score_samples(X))
        plt.colorbar()
        if save:
            plt.savefig(name + 'PCinit')
        else:
            plt.show()

        from sklearn_tda import MapperComplex
        from statmapper import mapper2networkx

        M = MapperComplex(**params)
        M.fit(X)
        stbase = M

        G = mapper2networkx(M)

        plt.figure()
        nx.draw(G, pos={k: [M.node_info_[k]['colors'][0], M.node_info_[k]['colors'][1]] for k in G.nodes}, node_color=[M.node_info_[k]['colors'][1] for k in G.nodes])
        if save:
            plt.savefig(name + 'MPinit')
        else:
            plt.show()
        #plt.close()

        Finit = np.zeros(shape=[len(M.node_info_),2]) #G.nodes)])
        for k in G.nodes:
            Finit[k,:] = np.array([M.node_info_[k]['colors'][0], M.node_info_[k]['colors'][1]])

        stbase_obs, F = [], []
        for idxboot in range(n_boot):
            #subsample = np.random.permutation(len(X))[:int(  len(X)/(.35*np.log(len(X)))  )]
            #subsample = np.random.choice(np.arange(len(X)),int(  len(X)/(np.log(len(X)))  ))
            subsample = np.random.choice(np.arange(len(X)),len(X))
            Xboot = X[subsample]

            if mapper_data == 'schic':
                params_boot = {
                               'filters': params['filters'][subsample],
                               'filter_bnds': params['filter_bnds'],
                               'colors': params['colors'][subsample],
                               'resolutions': np.array([10,10]), #params['resolutions'],
                               'gains': params['gains'],
                               'mask': params['mask'],
                               'clustering': AgglomerativeClustering(affinity='euclidean', distance_threshold=3, n_clusters=None) #params['clustering'],
                              }
            elif mapper_data == 'synth':
                params_boot = {
                               'filters': params['filters'][subsample],
                               'filter_bnds': params['filter_bnds'],
                               'colors': params['colors'][subsample],
                               'resolutions': np.array([15,15]), #params['resolutions'],
                               'gains': params['gains'],
                               'mask': params['mask'],
                               'clustering': AgglomerativeClustering(affinity='euclidean', distance_threshold=2, n_clusters=None) #params['clustering'],
                              }

            M = MapperComplex(**params_boot)
            M.fit(Xboot)
            stbase_obs.append(M.mapper_)

            plt.figure()
            plt.scatter(Xboot[:,0], Xboot[:,1], s=1)
            if save:
                plt.savefig(name + 'PC' + str(idxboot))
            else:
                plt.show()

            G = mapper2networkx(M)

            plt.figure()
            nx.draw(G, pos={k: [M.node_info_[k]['colors'][0], M.node_info_[k]['colors'][1]] for k in G.nodes}, node_color=[M.node_info_[k]['colors'][1] for k in G.nodes])
            if save:
                plt.savefig(name + 'MP' + str(idxboot))
            else:
                plt.show()


            Fboot = np.zeros(shape=[len(M.node_info_),2]) #G.nodes)])
            for k in G.nodes:
                Fboot[k,:] = np.array([M.node_info_[k]['colors'][0], M.node_info_[k]['colors'][1]])
            F.append(Fboot)


    elif model == '1Drandom':
        for i in range(N-1):
            stbase.insert([i,i+1])

        npts = stbase.num_vertices()
        assert npts == N

        F = np.zeros([npts,1])
        Finit = np.random.uniform(0,1,size=[npts])

    elif model == 'FCcircle':

        N = 6

        for i in range(N-1):
            stbase.insert([i,i+1])
        stbase.insert([N-1,0])

        npts = stbase.num_vertices()
        assert npts == N

        F = np.zeros([npts,1])
        #F = np.array([[0.,1.,1.5,1.,0.,1.5]]).T

        Finit = np.ones([npts])
        Finit[0:npts:2] = np.zeros([int(npts/2)])
        Finit = np.array([0., 1., 2., 1., 0., 2.], dtype=np.float32)
        Finit = Finit + np.random.uniform(-1e-4, 1e-4, size=[npts])

    elif model == 'templateRegistration':

        ### Observation ###
        N_obs = 60  # number of points in the big circle, for now we don't add messy edges

        for i in range(N_obs):
            stbase_obs.insert([i,i+1])
        for i in range(N_obs+1,2*N_obs-1):
            stbase_obs.insert([i,i+1])
        stbase_obs.insert([N_obs,N_obs+1])
        stbase_obs.insert([2*N_obs-1,0])

        npts = stbase_obs.num_vertices()
        assert npts == 2*N_obs

        # We build the observation
        # Note: we remove final endpoint for circle structure
        a,b,c,d,e,f = 0, 1, 0.05, 0.35, 0.1, 0.8
        F_obs = np.concatenate([np.linspace(a, b, int(N_obs/2)), np.linspace(b, c, int(N_obs/4)),
                                np.linspace(c, d, int(N_obs/4)), np.linspace(d, e, int(N_obs/4)),
                                np.linspace(e, f, int(N_obs/4)),
                                np.linspace(f, a, int(N_obs/2), endpoint=False)
                               ])
        eps = 0.1

        F_obs = (F_obs + eps*np.random.rand(2*N_obs))
        F = F_obs #np.hstack([F_obs[:,np.newaxis]])

        ### Template ###
        N_template = N

        for i in range(N_template-1):
            stbase.insert([i,i+1])
        stbase.insert([0, N_template-1])

        npts = stbase.num_vertices()
        assert npts == N_template

        #Finit = 0.8 * np.random.rand(N) #np.array([0.3, 0.8, 0.4, 0.7])
        if N == 4:
            Finit = np.array([0.33, 0.61, 0.31, 0.72])
        else:
            Finit = np.random.rand(N)
        print(Finit)

    elif model == 'random':

        points = np.random.uniform(size=[N,2])
        rips_complex = gd.RipsComplex(points=points, max_edge_length=10.)
        stbase = rips_complex.create_simplex_tree(max_dimension=5)
        stbase = barycentric_subdivision(stbase)
        print('dimension = ' + str(stbase.dimension()))
        F = np.zeros([stbase.num_vertices(),1])
        Finit = np.random.uniform(0,1,size=[stbase.num_vertices()])

    else:
        raise Exception('Model %s unknown' %model)

    return F, Finit, stbase, stbase_obs


def compute_diagram(stbase, F, dim, ext=False):
    st = gd.SimplexTree()
    for (s,_) in stbase.get_filtration():
        st.insert(s, -1e10)
    for i in range(len(F)):
        st.assign_filtration([i], F[i])
    st.make_filtration_non_decreasing()

    #TODO This code is duplicated with the same in utils_tda. Could we merge these?
    if ext:
        nvert = st.num_vertices()
        st.extend_filtration()
        dgms = st.extended_persistence(min_persistence=0.)
        ppairs = st.persistence_pairs()
        pairs, regs = [], []
        for p in ppairs:
            if len(p[0]) == 0 or len(p[1]) == 0:
                continue
            else:
                p1r = (p[0][0] != nvert)
                p1 = p[0] if p1r else p[0][1:]
                p2r = (p[1][0] != nvert)
                p2 = p[1] if p2r else p[1][1:]
                #print(p1, p2)
                pairs.append((p1,p2))
                regs.append((p1r,p2r))

        # Then, loop over all simplex pairs
        dgm, indices, pers = [], [], []
        for ip, (s1, s2) in enumerate(pairs):
            # Select pairs with good homological dimension and finite lifetime
            if len(s1) == dim+1 and len(s2) > 0:
                # Get IDs of the vertices corresponding to the filtration values of the simplices
                l1, l2 = np.array(s1), np.array(s2)
                idx1 = np.argmax(F[l1]) if regs[ip][0] else np.argmin(F[l1])
                idx2 = np.argmax(F[l2]) if regs[ip][1] else np.argmin(F[l2])
                i1, i2 = l1[idx1], l2[idx2]
                f1, f2 = F[i1], F[i2]
            if f1 <= f2:
                dgm.append([f1,f2])
            else:
                dgm.append([f2,f1])
            # Compute lifetime
            pers.append(np.abs(f1-f2))

        dgm = np.array(dgm)
        # Sort vertex pairs wrt lifetime
        perm = np.argsort(pers)
        dgm = dgm[perm][::-1,:]

    else:
        dgm = st.compute_persistence()
        dgm = st.persistence_intervals_in_dimension(dim)

    if len(dgm) == 0:
        dgm = np.empty(shape=[0,2])

    dgm = np.reshape(dgm[np.argwhere(~np.isinf(dgm[:,1]))], [-1,2])
    dgm = np.array([[0.5,0.5]]) if len(dgm) == 0 else dgm
    return dgm


def get_boundary_matrix(stbase, homdim=-1):
    matrix = []
    dimensions = []
    verts = []
    pos = {}
    if homdim == -1:
        p = 0
        for s,_ in stbase.get_filtration():
            if len(s) == 1:
                matrix.append([])
            else:
                matrix.append([pos[tuple(np.sort(s[:k] + s[k+1:]))] for k in range(len(s))])
            dimensions.append(len(s)-1)
            verts.append(s)
            pos[tuple(np.sort(s))] = p
            p += 1
    else:
        p = 0
        for s,_ in stbase.get_filtration():

            if len(s) == homdim+1:
                matrix.append([])
            elif len(s) == homdim+2:
                matrix.append([pos[tuple(np.sort(s[:k] + s[k+1:]))] for k in range(len(s))])

            if len(s) == homdim+1 or len(s) == homdim+2:
                dimensions.append(len(s)-1)
                verts.append(s)
                pos[tuple(np.sort(s))] = p
                p += 1

    return matrix, dimensions, verts


def build_persistence_diagrams(F, Finit, stbase, model, dim=0, stbase_obs=None, ext=False, coeffs=[0,1]):

    L = []

    if model == 'templateRegistration':
        if stbase_obs is None:
            raise Exception('You must provide stbase_obs when using model=templateRegistration')

        #Compute dgm for the observation
        dgm_obs = compute_diagram(stbase_obs, F, dim, ext)
        L.append(dgm_obs)

        #Compute dgm for the template
        dgm_template = compute_diagram(stbase, Finit, dim, ext)
        L.append(dgm_template)

    elif model == 'Mapper':
        for idb, stboot in enumerate(stbase_obs):
            dgboot = compute_diagram(stboot, np.multiply(F[idb], np.array(coeffs)[None,:]).sum(axis=1), dim, ext)
            L.append(dgboot)
        dg = compute_diagram(stbase.mapper_, np.multiply(Finit, np.array(coeffs)[None,:]).sum(axis=1), dim, ext)
        L.append(dg)

    else:

        for fct in [F[:,i] for i in range(F.shape[1])] + [Finit]:
            dgm = compute_diagram(stbase, fct, dim)
            L.append(dgm)

    return L


#######################
### Plots functions ###
#######################


def plot_curves(losseslist, gradients, times, save=False, name='lossgrad'):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].plot(losseslist)
    axs[0].set_title('Losses')
    grad_norms = np.array([g[1] for g in gradients])
    axs[1].plot(grad_norms)
    axs[1].set_title('Gradient norms')
    axs[1].set_ylim(0)  # we know we want a grad close to 0 so this is a benchmark.
    axs[2].plot(times)
    axs[2].set_title('Computation time (s)')
    for ax in axs:
        ax.set_xlabel('Epochs')
    if save:
        plt.savefig(name)
    else:
        plt.show()


def plot_equal_pairs(F, save=False, name='pairs'):
    print(str(int(((np.abs(F[:,np.newaxis]-F[np.newaxis,:]) == 0.).sum()-len(F))/2)) + str(' degenerate pairs'))
    plt.figure()
    plt.imshow(np.abs(F[:,np.newaxis]-F[np.newaxis,:]) == 0.)
    plt.colorbar()
    if save:
        plt.savefig(name)
    else:
        plt.show()


def plot_sequence_of_diagrams(dgms, L, xmin=0, xmax=np.sqrt(2), every=10, save=False, name='dgmscv'):
    get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
    colors = get_colors(len(L))
    colors = ['red', 'blue', 'green', 'purple', 'yellow', 'orange', 'pink', 'brown', 'magenta', 'teal']
    rotmat = (np.sqrt(2)/2) * np.array([[1.,1.],[-1.,1.]])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dg = rotmat.dot(dgms[0].T).T
    ax.scatter(dg[:,0], dg[:,1], s=40, marker='x', c='turquoise', label='init diagram')
    #TODO: plot arrow ?
    for D in dgms[1:-1:every]:
        dg = rotmat.dot(D.T).T
        ax.scatter(dg[:,0], dg[:,1], s=20, marker='D', alpha=.1)
    dg = rotmat.dot(dgms[-1].T).T
    ax.scatter(dg[:,0], dg[:,1], s=40, marker='D', c='black', label='final diagram')
    for i in range(len(L)-1):
        dg = rotmat.dot(L[i].T).T
        ax.scatter(dg[:,0], dg[:,1], s=40, marker='o', c=colors[i], label='target diagram %i' %i)
    ax.plot([xmin,xmax], [0,0])
    ax.set_title('Sequence of diagrams')
    ax.legend()
    if save:
        plt.savefig(name)
    else:
        plt.show()


def plot_diagrams(L, xmin=0, xmax=1, ymin=0, ymax=1, save=False, name='dgms'):

    num_dgms = len(L)

    plt.figure()
    labs = ['dgm ' + str(i) for i in range(num_dgms-1)] + ['dgm init']
    for i, dgm in enumerate(L):
        plt.scatter(dgm[:,0], dgm[:,1], label=labs[i])
    plt.plot([xmin,xmax], [ymin,ymax])
    plt.axis('square')
    plt.legend()
    if save:
        plt.savefig(name)
    else:
        plt.show()


def plot_functions(F, model, start_epoch=-4, end_epoch=0, every=1, save=False, name='fct'):

    num_fcts = len(F)

    for i, fct in enumerate(F):

        N = len(fct)

        if model == 'circle':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if type(fct) == list:
                for j in range(start_epoch, end_epoch, every):
                    N = len(fct[j])
                    ax.plot3D(np.concatenate([ np.cos([theta for theta in np.linspace(0,-np.pi,int(N/2))]),
                                               np.cos([theta for theta in np.linspace(np.pi,0,int(N/2))]) ]),
                              np.concatenate([ np.sin([theta for theta in np.linspace(0,-np.pi,int(N/2))]),
                                               np.sin([theta for theta in np.linspace(np.pi,0,int(N/2))]) ]),
                              np.concatenate([fct[j][:int(N/2)+1],fct[j][int(N/2)+1:][::-1]]), label = 'epoch ' + str(j))
            else:
                ax.plot3D(np.concatenate([ np.cos([theta for theta in np.linspace(0,-np.pi,int(N/2))]),
                                           np.cos([theta for theta in np.linspace(np.pi,0,int(N/2))]) ]),
                          np.concatenate([ np.sin([theta for theta in np.linspace(0,-np.pi,int(N/2))]),
                                           np.sin([theta for theta in np.linspace(np.pi,0,int(N/2))]) ]),
                          np.concatenate([fct[:int(N/2)+1],fct[int(N/2)+1:][::-1]]))
            plt.title('function ' + str(i))
            plt.legend()

        elif model == 'FCcircle' or model == 'templateRegistration':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            if type(fct) == list:
                for j in range(start_epoch, end_epoch, every):
                    N = len(fct[j])
                    ax.plot3D(np.cos([theta for theta in np.linspace(0,2*np.pi,N+1)]),
                              np.sin([theta for theta in np.linspace(0,2*np.pi,N+1)]),
                              np.array(list(fct[j]) + [fct[j][0]]), label = 'epoch ' + str(j))
            else:
                ax.plot3D(np.cos([theta for theta in np.linspace(0,2*np.pi,N+1)]),
                          np.sin([theta for theta in np.linspace(0,2*np.pi,N+1)]),
                          np.array(list(fct) + [fct[0]]))
            plt.title('function ' + str(i))
            plt.legend()

        elif model == '1Drandom' or model == 'peaks':
            plt.figure()
            if type(fct) == list:
                N = len(fct[j])
                for j in range(start_epoch, end_epoch, every):
                    plt.plot(fct[j], label = 'epoch ' + str(j))
            else:
                plt.plot(fct)
            plt.legend()

        elif model == 'random':
            plt.figure()
            plt.plot(fct)

        if save:
            plt.savefig(name + str(i))
        else:
            plt.show()


def plot_diagram_distances(dgms, conv=1, save=False, name='dgmsdist'):
    ws = conv
    plt.figure()
    dbs = [gd.bottleneck_distance(dgms[i],dgms[i+1]) for i in range(len(dgms)-1)]
    dw1 = [gd.wasserstein.wasserstein_distance(dgms[i],dgms[i+1],order=1) for i in range(len(dgms)-1)]
    dw2 = [gd.wasserstein.wasserstein_distance(dgms[i],dgms[i+1],order=2) for i in range(len(dgms)-1)]
    win = np.repeat([1.], ws)
    plt.plot(signal.convolve(dbs, win, mode='same') / sum(win), label='B')
    plt.plot(signal.convolve(dw1, win, mode='same') / sum(win), label='W1')
    plt.plot(signal.convolve(dw2, win, mode='same') / sum(win), label='W2')
    plt.title('Sequence of consecutive diagram distances (moving average length ' + str(ws) + ')')
    plt.legend()
    if save:
        plt.savefig(name)
    else:
        plt.show()


def plot_function_distances(funcs, conv=1, save=False, name='fctdist'):
    ws = conv
    plt.figure()
    dl = [np.linalg.norm(funcs[i]-funcs[i+1]) for i in range(len(funcs)-1)]
    win = np.repeat([1.], ws)
    plt.plot(signal.convolve(dl, win, mode='same') / sum(win), label='1-norm')
    plt.title('Sequence of consecutive function distances (moving average length ' + str(ws) + ')')
    plt.legend()
    if save:
        plt.savefig(name)
    else:
        plt.show()


def plot_gradients(gradients, save=False, name='grads'):
    plt.figure()
    plt.plot([gradients[i][1] for i in range(len(gradients))])
    if save:
        plt.savefig(name)
    else:
        plt.show()
