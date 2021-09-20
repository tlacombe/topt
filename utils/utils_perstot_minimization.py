import numpy as np
import gudhi as gd
import matplotlib.pyplot as plt
import random
import tensorflow as tf


def build_model(N):

    stbase = gd.SimplexTree()
    stbase_obs = gd.SimplexTree()

    for i in range(N-1):
        stbase.insert([i,i+1])

    npts = stbase.num_vertices()
    assert npts == N

    F = np.zeros([npts,1])
    Finit = np.random.uniform(0,1,size=[npts])

    return F, Finit, stbase, stbase_obs


def _compute_diagram(stbase, F, dim, ext=False):
    st = gd.SimplexTree()
    for (s,_) in stbase.get_filtration():
        st.insert(s, -1e10)
    for i in range(len(F)):
        st.assign_filtration([i], F[i])
    st.make_filtration_non_decreasing()

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


def build_persistence_diagrams(F, Finit, stbase, dim=0):

    L = []

    for fct in [F[:,i] for i in range(F.shape[1])] + [Finit]:
        dgm = _compute_diagram(stbase, fct, dim)
        L.append(dgm)

    return L[:-1], L[-1]


#######################
### Plots functions ###
#######################


def plot_curves(losseslist, gradients, times):
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].plot(losseslist)
    axs[0].set_title('Objective value')
    grad_norms = np.array([g[1] for g in gradients])
    axs[1].plot(grad_norms)
    axs[1].set_title('Gradient norms')
    axs[1].set_ylim(0)  # we know we want a grad close to 0 so this is a benchmark.
    axs[2].plot(times)
    axs[2].set_title('Computation time (s)')
    [ax.set_xlabel('Iteration $k$') for ax in axs]


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


def plot_functions(F, start_epoch=-4, end_epoch=0, every=1):

    for i, fct in enumerate(F):

        N = len(fct)

        plt.figure()
        if type(fct) == list:
            N = len(fct[j])
            for j in range(start_epoch, end_epoch, every):
                plt.plot(fct[j], label = 'epoch ' + str(j))
        else:
            plt.plot(fct)
        plt.legend()
