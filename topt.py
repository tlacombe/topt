import heapq
import sys
from time import time
import numpy as np
import tensorflow as tf
import gudhi as gd

import gudhi.wasserstein as wass

import cvxpy as cp


##############################
###   Convex optim utils   ###
##############################


def _min_norm_on_convex_hull(G):
    """
    Minimizing on convex hull
    Warning: sometimes 0 is one of the vertice of te convex hull but the solver do not get it.
             This must be handled independently (e.g. testing if 0 is in first).
    """
    nb_pts, dim = G.shape
    w = cp.Variable(nb_pts)
    U = w @ G
    prob = cp.Problem(cp.Minimize(cp.sum_squares(U)), [w >= 0, cp.sum(w) == 1])
    prob.solve()
    return U.value

#####################
###  Persistence  ###
#####################


def _STPers_G(fct, simplex, dim):
    # fct:      function values on the vertices
    # simplex:  simplex tree
    # dim:      homological dimension

    # Copy simplex in another simplex tree st
    st = gd.SimplexTree()
    [st.insert(s,-1e10) for s,_ in simplex.get_filtration()]

    # Assign new filtration values
    for i in range(st.num_vertices()):
        st.assign_filtration([i], fct[i])
    st.make_filtration_non_decreasing()

    # Compute persistence diagram (must be called, gudhi requirement)
    dgm = st.persistence(min_persistence=0.)

    # Get vertex pairs for optimization. First, get all simplex pairs
    pairs = st.persistence_pairs()

    # Then, loop over all simplex pairs
    indices, pers = [], []
    for s1, s2 in pairs:
        # Select pairs with good homological dimension and finite lifetime
        if len(s1) == dim+1 and len(s2) > 0:
            # Get IDs of the vertices corresponding to the filtration values of the simplices
            l1, l2 = np.array(s1), np.array(s2)
            i1 = l1[np.argmax(fct[l1])]
            i2 = l2[np.argmax(fct[l2])]
            indices.append(i1)
            indices.append(i2)
            # Compute lifetime
            pers.append(st.filtration(s2) - st.filtration(s1))

    # Sort vertex pairs wrt lifetime
    perm = np.argsort(pers)
    indices = np.reshape(indices, [-1,2])[perm][::-1,:].flatten()

    return indices


def _STPers_E(fct, simplex, dim, verbosity_level = 1):
    """
    Same as before but for extended persistence.

    :param fct: function values on the vertices
    :param simplex: simplex tree
    :param dim: homological dimension
    :return:
    """

    # Copy simplex in another simplex tree st
    st = gd.SimplexTree()
    [st.insert(s,-1e10) for s,_ in simplex.get_filtration()]
    nvert = st.num_vertices()

    # Assign new filtration values
    for i in range(st.num_vertices()):
        st.assign_filtration([i], fct[i])
    st.make_filtration_non_decreasing()

    st.extend_filtration()
    dgms = st.extended_persistence(min_persistence=0.)

    # Get vertex pairs for optimization. First, get all simplex pairs
    ppairs = st.persistence_pairs()
    pairs, regs = [], []
    for p in ppairs:
        if len(p[0]) == 0 or len(p[1]) == 0:
            continue
        else:
            p1r = (p[0][0] != nvert) # (nvert in p[0])
            p1 = p[0] if p1r else p[0][1:]
            p2r = (p[1][0] != nvert)
            p2 = p[1] if p2r else p[1][1:]
            pairs.append((p1,p2))
            regs.append((p1r,p2r))

    # Then, loop over all simplex pairs
    indices, pers = [], []
    for ip, (s1, s2) in enumerate(pairs):
        # Select pairs with good homological dimension and finite lifetime
        if len(s1) == dim+1 and len(s2) > 0:
            # Get IDs of the vertices corresponding to the filtration values of the simplices
            l1, l2 = np.array(s1), np.array(s2)
            idx1 = np.argmax(fct[l1]) if regs[ip][0] else np.argmin(fct[l1])
            idx2 = np.argmax(fct[l2]) if regs[ip][1] else np.argmin(fct[l2])
            i1, i2 = l1[idx1], l2[idx2]
            f1, f2 = fct[i1], fct[i2]
            if f1 <= f2:
                indices.append(i1)
                indices.append(i2)
            else:
                indices.append(i2)
                indices.append(i1)
            # Compute lifetime
            pers.append(np.abs(f1-f2))
            #if np.isinf(f1) or np.isinf(f2):
            #    print(f1,f2,i1,i2)

    # Sort vertex pairs wrt lifetime
    perm = np.argsort(pers)
    indices = np.reshape(indices, [-1,2])[perm][::-1,:].flatten()
    if verbosity_level >= 2:
        print("Permutation in extended persistence:\n", perm)
        print("Corresponding indices:\n", indices)

    return indices

##############################
###  Get strata functions  ###
##############################


def _treat(x, permutation,  dist_to_x, eps):
    """
    Get permutations around a sorted vector.
    """
    n = x.shape[0]
    x_perm = x[permutation]
    #TODO: why is perm_inv not used anymore?
    # #perm_inv = np.arange(len(permutation))[np.argsort(permutation)]
    last_to_first = (x[-1]-x_perm[0])**2+(x[0]-x_perm[-1])**2-(x[0]-x_perm[0])**2-(x[-1]-x_perm[-1])**2
    delta_cost = [((x[i]-x_perm[i+1])**2+(x[i+1]-x_perm[i])**2-(x[i]-x_perm[i])**2-(x[i+1]-x_perm[i+1])**2)
                  for i in range(n-1)]
    delta_cost.append(last_to_first)
    distance_mirror_to_x = np.sqrt(np.maximum(dist_to_x**2 + np.array(delta_cost), 0))
    idx = np.where(distance_mirror_to_x < 2 * eps)[0]
    distances = list(distance_mirror_to_x[idx])

    permutations_to_consider = []
    for i in idx:
        tmp_perm = permutation.copy()
        if i < n-1:
            tmp_perm[[i+1, i]] = tmp_perm[[i, i+1]]
        else:
            tmp_perm[[0, i]] = tmp_perm[[i, 0]]
        permutations_to_consider.append(tmp_perm)
    return permutations_to_consider, distances


def _get_strata_memoization(F, perm_curr, eps, visited, k=5, timeit=False):
    t1, n = time(), F.shape[0]
    visited_keys, delete_keys, visited_dists = [], [], []

    for permi_key in visited.keys():
        permi = np.array(permi_key)
        inv_permi = np.arange(len(permi))[np.argsort(permi)]
        Fi = tf.gather(F, perm_curr[inv_permi])
        disti = tf.norm(F-Fi)

        if disti <= 2*eps:
            visited_keys.append(permi_key)
            visited_dists.append(disti.numpy())
        else:
            delete_keys.append(permi_key)
        if len(visited_keys) == k:
            break

    if timeit:
        t2 = time()
        print('Running time of get_strata (with k = %s, n = %s, %s permutations): %.3f s'
              %(k, n, len(visited_keys), t2 - t1))

    return visited_keys, visited_dists, delete_keys


def _get_strata_dijkstra(F, eps, k=5, timeit=True):
    arbitrary = 0
    t1, n, sigma, x = time(), F.shape[0], np.argsort(F), np.sort(F)
    inv_sigma, identity_permutation = np.arange(len(sigma))[np.argsort(sigma)], np.arange(0, n)
    todo_perm = [(0.,arbitrary,identity_permutation)]
    heapq.heapify(todo_perm)
    count, done, permutations_to_consider, distances_to_consider = 0, set([]), [], []

    while todo_perm:
        newperm = heapq.heappop(todo_perm)
        perm, dist_to_x = newperm[2], newperm[0]
        if not (tuple(perm) in done):
            permutations_to_consider.append(perm)
            distances_to_consider.append(dist_to_x)
            count += 1
            if count >= k:
                if timeit:
                    t2 = time()
                    print('Running time of get_strata (with k = %s and n = %s): %.2f s' %(k, n, t2 - t1))
                return permutations_to_consider, distances_to_consider

            permutations, distances = _treat(x, perm, dist_to_x, eps)
            for u in range(len(distances)):
                arbitrary += 1
                heapq.heappush(todo_perm, (distances[u], arbitrary, permutations[u]))
            done.add(tuple(perm))

    if timeit:
        t2 = time()
        print('Running time of get_strata (with k = %s (not reached), n = %s, %s permutations): %.3f s'
              %(k, n, len(permutations_to_consider), t2 - t1))

    return permutations_to_consider, distances_to_consider


def _get_strata_diffusion(F, eps, k=5, timeit=True):
    t1, n = time(), F.shape[0]
    permutations_to_consider, distances_to_consider = [], []
    x = np.sort(F)
    dist_curr = 0.
    perm_curr = np.arange(0, len(x))
    for _ in range(k):
        good_perm, good_dist = perm_curr, dist_curr
        permutations_to_consider.append(tuple(good_perm))
        distances_to_consider.append(good_dist)
        permutations, distances = _treat(x, perm_curr, dist_curr, eps)
        #TODO: should we remove this, or add a in_place=False default arg?
        ## Uncomment if you want to allow to stay in place
        #permutations.append(perm_curr)
        #distances.append(0.)
        #probas = distances/np.array(distances).sum()           # Probabilities are proportional to distances
        if len(distances):
            probas = np.ones([len(distances)])/len(distances)       # Uniform probability distribution
            u = np.random.choice(range(len(probas)),p=probas)
            perm_curr = permutations[u]
            dist_curr = np.sqrt(np.square(x-x[perm_curr]).sum())
    if timeit:
        t2 = time()
        print('Running time of get_strata (with k = %s, n = %s, %s permutations): %.3f s'
              %(k, n, len(permutations_to_consider), t2 - t1))

    return permutations_to_consider, distances_to_consider


def _get_strata_padded(heuristic='diffusion'):
    """
    Tensorflow padded version
    """
    def gs(F, eps, card_strata):
        if heuristic == 'diffusion':
            outp = _get_strata_diffusion(F=F, eps=eps, k=card_strata, timeit=False)
        elif heuristic == 'dijkstra':
            outp = _get_strata_dijkstra(F=F, eps=eps, k=card_strata, timeit=False)
        else:
            print('no heuristic provided! using dijkstra by default')
            outp = _get_strata_dijkstra(F=F, eps=eps, k=card_strata, timeit=False)
        elems, dists = outp[0], outp[1]
        num = len(elems)
        if num < card_strata:
            [elems.append(elems[0]) for _ in range(card_strata - num)]
            [dists.append(dists[0]) for _ in range(card_strata - num)]
        tmp, tmpd = elems[:card_strata], dists[:card_strata]

        res = [np.int32(min(card_strata, num))] + [np.int32(x) for s in tmp for x in s] + [np.float32(d) for d in tmpd]
        return res
    return gs


##################
# ************** #
##################

#TODO: should get_strata take topomean as input instead? (from which we can get self.eps, self.card_strata, etc.)
def _get_strata(topomean, STPersTF, perm_curr, epoch):

    F = topomean.F
    heuristic = topomean.heuristic
    complementary_heuristic = topomean.complementary_heuristic
    eps = topomean.epsilon
    card_strata = topomean.card_strata
    use_memoization = topomean.use_memoization
    verbosity_level = topomean.verbosity_level
    max_dict_size = topomean.max_dict_size
    visited = topomean.visited

    inv_perm_curr = np.argsort(perm_curr)
    nvertex = F.numpy().shape[0]

    if heuristic == 'diffusion' or heuristic == 'dijkstra':

        StrataTF = _get_strata_padded(heuristic)
        inds = tf.cast(tf.stop_gradient(StrataTF(F.numpy(), eps, card_strata)), tf.int32)
        num = inds[0].numpy()
        perms = tf.reshape(inds[1:nvertex*num+1], [num, nvertex])
        dists = tf.reshape(inds[nvertex*card_strata+1:nvertex*card_strata+1+num], [num])

        already = 0
        keys_to_try, dists_to_try = [], []

        for i in range(num):

            permi = perms[i].numpy()
            disti = dists[i].numpy()
            inv_permi = np.argsort(permi)
            permi_sort = perm_curr[inv_permi]
            permi_key = tuple(permi_sort)

            if use_memoization:
                try:
                    indsi = visited[permi_key]
                    already += 1
                except KeyError:
                    Fi = tf.gather(F, perm_curr[permi[inv_perm_curr]])
                    indsi = tf.cast(tf.stop_gradient(STPersTF(Fi.numpy())), tf.int32)
                    visited[permi_key] = indsi
            else:
                Fi = tf.gather(F, perm_curr[permi[inv_perm_curr]])
                indsi = tf.cast(tf.stop_gradient(STPersTF(Fi.numpy())), tf.int32)
                visited[permi_key] = indsi

            keys_to_try.append(permi_key)
            dists_to_try.append(disti)

        if verbosity_level >= 1:
            print('Found %s new permutations, %s already recorded permutations, '
                  'among %s permutations in the permutahedron obtained with heuristic %s'
                  %(num-already, already, num, heuristic))
            print('Number of recorded permutations is now %s' %len(visited.keys()))

        if len(visited) > max_dict_size:
            num_keys_to_delete = len(visited) - max_dict_size
            deleted_keys, curr_key, list_keys = 0, 0, list(visited.keys())
            while deleted_keys < num_keys_to_delete and curr_key < len(list_keys):
                if list_keys[curr_key] not in keys_to_try:
                    del visited[list_keys[curr_key]]
                    deleted_keys += 1
                curr_key += 1
            if verbosity_level >= 1:
                print('Number of recorded permutations is now %s after trimming' %len(visited))

    elif heuristic == 'memoization':

        visited_keys, visited_dists, delete_keys = _get_strata_memoization(F, perm_curr, eps, visited, card_strata)
        keys_to_try = visited_keys
        dists_to_try = visited_dists
        num_visited_keys = len(visited_keys)
        num_missing_keys = card_strata-num_visited_keys
        new_keys, new_dists = [], []
        if verbosity_level >= 1:
            print('Found %s recorded permutations at distance 2 * epsilon' %num_visited_keys)

        identity_perm = tuple(perm_curr)
        try:
            _ = visited[identity_perm]
            identity_memo = True
        except KeyError:
            identity_memo = False

        check_identity = True if identity_perm in visited_keys else False

        if identity_memo == True and check_identity == False:
            visited_keys[0] = identity_perm
            visited_dists[0] = 0.
            check_identity = True

        if num_visited_keys < card_strata or check_identity == False:
            if verbosity_level >= 1:
                if num_visited_keys < card_strata:
                    print('Lets try to find new permutations')

                if check_identity == False and epoch > 0:
                    print('Adding current permutation to the list')

            if num_visited_keys == card_strata:
                num_missing_keys = 1
                delete_keys = list(visited.keys())[:1]
                visited_keys = visited_keys[1:]
                visited_dists = visited_dists[1:]
                card_strata = 1

            num_to_visit = num_missing_keys
            #num_to_visit = card_strata

            StrataTF = _get_strata_padded(complementary_heuristic)
            inds = tf.stop_gradient(StrataTF(F.numpy(), eps, num_to_visit))
            num = inds[0].numpy()[0]
            perms = tf.reshape(inds[1:nvertex*num+1], [num, nvertex])
            dists = tf.reshape(inds[nvertex*num_missing_keys+1:nvertex*num_missing_keys+1+num], [num])

            already = 0
            diags_todo = []

            for i in range(num):

                permi = perms[i].numpy()
                disti = dists[i].numpy()
                inv_permi = np.argsort(permi)
                permi_sort = perm_curr[inv_permi]
                permi_key = tuple(permi_sort)

                try:
                    indsi = visited[permi_key]
                    already += 1
                except KeyError:
                    diags_todo.append(np.argsort(permi_sort)[None,:])
                    Fi = tf.gather(F, perm_curr[permi[inv_perm_curr]])
                    indsi = tf.stop_gradient(STPersTF(Fi.numpy()))
                    visited[permi_key] = indsi

                    new_keys.append(permi_key)
                    new_dists.append(disti)

                if len(new_keys) == num_missing_keys:
                    break

            if verbosity_level >= 1:
                print('Found %s new permutations, %s already recorded permutations, among %s permutations '
                      'in the permutahedron obtained with heuristic %s'
                      %(len(new_keys), already, num, complementary_heuristic))
                print('Number of recorded permutations is %s' %len(visited.keys()))

            if len(visited) > max_dict_size:
                num_keys_to_delete = len(visited) - max_dict_size
                for k in range(num_keys_to_delete):
                    del visited[delete_keys[k]]
                if verbosity_level >= 1:
                    print('Number of recorded permutations is %s after trimming' %len(visited))

        keys_to_try = keys_to_try + new_keys
        dists_to_try = dists_to_try + new_dists

    return keys_to_try, dists_to_try


##########################
###  TensorFlow model  ###
##########################

class TopoMeanModel(tf.keras.Model):
    """
        L: list of given persistence diagrams---the model is supposed to compute their Fréchet mean
        F: function values on the vertices of stbase
        params: param for the TopoMeanModel, see the corresponding cell
    """

    def __init__(self, F, diagrams, simplex,
             dim = 0,
             mode = 'vanilla',
             card_strata = 24,
             card_dgm_max=50,  # Note : not used while long range strata is not provided.
             max_dict_size=500,
             heuristic = 'dijkstra',
             complementary_heuristic = 'dijkstra',
             epsilon=0.1,
             eta = 1e-2,
             beta = 0.5,
             gamma=0.5,
             lipschitz = 1,
             order = 2.,
             internal_p = 2.,
             use_memoization = True,
             extended = True,
             verbosity_level = 0,
             normalize_gradient=True,
             vanilla_decay = False
             ):
        # Short preprocessing to make everything comparable
        if (mode == "gradient_sampling") | (mode == "vanilla"):
            epsilon = epsilon / 2  # To make strata and gradient_sampling comparable

        super().__init__(dynamic=True)

        self.F, self.simplex = F, simplex
        if isinstance(diagrams, list):
            self.L = diagrams
        else:
            self.L = [diagrams]

        self.dim, self.mode = dim, mode
        self.epsilon, self.card_strata = epsilon, card_strata
        self.heuristic, self.complementary_heuristic = heuristic, complementary_heuristic
        self.eta, self.beta, self.gamma, self.lipschitz = eta, beta, gamma, lipschitz
        self.order, self.internal_p = order, internal_p
        self.use_memoization, self.max_dict_size = use_memoization, max_dict_size
        self.extended = extended

        self.verbosity_level = verbosity_level

        # We only need 'normalize_gradient' for vanilla and GS, Strata is *always* normalized.
        if self.mode == "vanilla" or self.mode == "gradient_sampling":
            self.normalize_gradient = normalize_gradient

        if self.mode == 'vanilla':
            self.vanilla_decay = vanilla_decay

        self.dgm, self.visited, self.curr_visited = None, {}, {}
        self.losseslist = []
        self.times = []

    def loss(self, STPersTF, epoch):

        if self.mode == 'vanilla' or self.mode == 'gradient_sampling':
            start = time()
            perm = tuple(np.argsort(self.F.numpy()))

            # Don't try to compute gradients for the vertex pairs
            try:
                inds = self.visited[perm]
            except KeyError:
                inds = tf.stop_gradient(STPersTF(self.F.numpy()))
                self.visited[perm] = inds

            # Get persistence diagram
            if len(inds) > 0:
                self.dgm = tf.reshape(tf.gather(self.F, inds), [-1, 2])
            else:
                self.dgm = tf.reshape(tf.gather(self.F, [0,0]), [-1,2])

            # Compute the loss of this dgm.
            try: # Trick to handle Gudhi 3.5 not available
                loss = tf.add_n([wass.wasserstein_distance(self.dgm, tf.constant(D),
                                                           order=self.order,
                                                           internal_p=self.internal_p,
                                                           enable_autodiff=True,
                                                           keep_essential_parts=False)**self.order
                                 for D in self.L])
            except:
                loss = tf.add_n([wass.wasserstein_distance(self.dgm, tf.constant(D),
                                                           order=self.order,
                                                           internal_p=self.internal_p,
                                                           enable_autodiff=True)**self.order
                                 for D in self.L])

            end = time()
            if self.verbosity_level >= 2:
                print('Computing all persistence diagrams and losses took %.3f secs' %(end-start))
            self.times.append(end-start)

            return loss

        elif self.mode == 'strata':
            start = time()

            perm_curr = np.argsort(self.F.numpy()).ravel()
            keys_to_try, dists_to_try = _get_strata(self, STPersTF, perm_curr, epoch)

            losses = []
            self.curr_visited = len(keys_to_try)
            rebuild_grads = []
            if self.verbosity_level >= 1:
                print('Computing gradient over ' + str(self.curr_visited) + ' strata')

            for i in range(self.curr_visited):
                permi_key = keys_to_try[i]
                indices_i = self.visited[permi_key]
                perm_i = np.array(permi_key)
                inv_perm_i = np.argsort(perm_i)
                perm_F_to_Fi = perm_curr[inv_perm_i]
                F_i = tf.gather(self.F, perm_F_to_Fi)
                assert np.linalg.norm(F_i.numpy()-self.F.numpy()) <= 2 * self.epsilon
                rebuild_grads.append(perm_F_to_Fi)

                # Get persistence diagram
                dgm_i = tf.reshape(tf.gather(F_i, indices_i), [-1,2]) if len(indices_i) > 0 \
                    else tf.reshape(tf.gather(F_i, [0,0]), [-1,2])

                if self.verbosity_level >= 2:
                    print(indices_i)
                    print("Function %s:\n" %i, F_i.numpy())
                    print("Diagram %s:\n" %i, dgm_i.numpy())

                if np.all(perm_i == perm_curr):
                    self.dgm = dgm_i

                # Loss is given as the sum of the Wasserstein distances (**order) between the current
                # persistence diagram and all other persistence diagrams in the list L
                # Note: in some expe (min perstot, registration, L is of size 1)
                try: # Trick when gudhi 3.5 is not officially released, to be removed.
                    loss_i = tf.add_n([wass.wasserstein_distance(dgm_i, tf.constant(D),
                                                                 order=self.order,
                                                                 internal_p=self.internal_p,
                                                                 enable_autodiff=True,
                                                                 keep_essential_parts=False)**self.order
                                      for D in self.L])
                except:
                    loss_i = tf.add_n([wass.wasserstein_distance(dgm_i, tf.constant(D),
                                                                 order=self.order,
                                                                 internal_p=self.internal_p,
                                                                 enable_autodiff=True)**self.order
                                      for D in self.L])

                losses.append(loss_i)

            end = time()
            if self.verbosity_level >= 2:
                print('Computing all persistence diagrams and losses took %.3f secs' %(end-start))
            self.times.append(end-start)

            return losses, rebuild_grads, dists_to_try #keys_to_try

    def call(self, epoch):

        if self.extended:
            STPersTF = lambda fct: _STPers_E(fct, self.simplex, self.dim, verbosity_level=self.verbosity_level)
        else:
            STPersTF = lambda fct: _STPers_G(fct, self.simplex, self.dim)

        return self.loss(STPersTF, epoch=epoch)


def _compute_single_gradient(grads, verbosity_level):
    """
    :param grads: A list of grads obtained from a sampling procedure (typically either random, or stratified).
    :param verbosity_level: 0 : silent, 1 : standard, 2 : debug.
    :return: Gradient (as numpy array), Gradient (as tf object), and its norm. It is obtaind as a reduction over the
             gradients in the grads list using minimum on convex hull.
    """

    # We first check if 0 is in the convex hull (as our cvx optimizer is not reliable...)
    # If so, the single_grad we output is simply 0
    if len(np.argwhere(np.linalg.norm(grads,axis=1)==0)) > 0:
        single_grad = [tf.convert_to_tensor(np.zeros(len(grads[0])))]
    # Otherwise, it is the minimum on convex hull of the grads in list computed using cvx optimizer.
    else:
        single_grad = [tf.convert_to_tensor(_min_norm_on_convex_hull(grads))]

    try:
        G = single_grad[0].numpy()
    except AttributeError:
        G = np.array(single_grad[0].values)
    norm_grad = np.linalg.norm(G)

    if verbosity_level >= 2:
        print("List of grads to be considered:\n", grads)
        print("\nResulting grad:\n", single_grad[0].numpy())
        print("\nIts norm: ", norm_grad)

    return G, single_grad, norm_grad


def _reduce_gradient(topomean,
                     grads, single_grad, norm_grad,
                     loop_epsilon,
                     dists=None,
                     epoch=0):
    """
    Given a current topomean state and the corresponding grads, signle_grad, norm_grad, compute parameters to perform
    a gradient step. For instance, in topomean.mode=strata, it reduces the epsilon parameter and then take
    (instantly, as strata are ordered) the subset of grads that belong to the B(topomean.F, new_epsilon).
    In topomean.mode=gradient_sampling, it simply computes optimal parameter epsilon (reduction until decent direction).

    :param topomean: The current topomean state (store vertex values, provides gradient, etc.)
    :param grads: list of grads around the current point (topomean.F), computed wrt topomean.mode
    :param single_grad: The reduction of grads (typically, min_norm_on_convex_hull).
    :param norm_grad: The norm of single_grad
    :param loop_epsilon:
    :param dists: Distances to strata (useful only in topomean.mode=strata). None otherwise.
    :param epoch: The current epoch.
    :return:
    """

    mode            = topomean.mode
    epsilon         = topomean.epsilon
    beta            = topomean.beta
    lipschitz       = topomean.lipschitz
    eta             = topomean.eta
    gamma           = topomean.gamma
    verbosity_level = topomean.verbosity_level

    if mode == 'strata':
        good_single_grad = single_grad
        good_norm_grad = norm_grad
        good_epsilon = epsilon

        constant = (1-beta) / (2*lipschitz)
        epstimes = 0

        if verbosity_level >= 2:
            print('dists:', np.array(dists))

        start = time()
        # In the following loop, we check a criterion to ensure loss decrease.
        # We first check a stopping criterion,
        # If not, we reduce our espilon, and select a subset of the gradients of interest in B(x_k, new_epsilon)
        # It can be done efficiently because we select distances in increasing order. So we already computed all
        # gradients of interest.
        while True:
            if good_epsilon <= constant*good_norm_grad or good_norm_grad <= eta:
                break
            else:
                epstimes += 1
                if loop_epsilon:
                    # reduce epsilon
                    good_epsilon *= gamma
                    # Take subset of gradients
                    good_grads = grads[np.argwhere(np.array(dists) <= good_epsilon).ravel()]
                    if verbosity_level >= 2:
                        print("good_epsilon = ", good_epsilon)
                        print('good grads (those selected in range good_epsilon):\n', good_grads)
                    # Compute the new generalized gradient
                    good_single_grad = [tf.convert_to_tensor(_min_norm_on_convex_hull(good_grads))]
                    try:
                        G = good_single_grad[0].numpy()
                    except AttributeError:
                        G = np.array(good_single_grad[0].values)
                    good_norm_grad = np.linalg.norm(G)
                else:
                    good_epsilon = constant*good_norm_grad
                    break
        end = time()
        if verbosity_level >= 1:
            print('Had to reduce epsilon %s times, which took %.3f secs' %(epstimes, end-start))

    elif mode == 'gradient_sampling':

        good_single_grad = single_grad
        good_norm_grad = norm_grad
        good_epsilon = epsilon

        # Current state of topomean (simplex values and corresponding loss)
        curr_F = topomean.F
        curr_loss = topomean.call(epoch=epoch)

        # The loss we want to reach
        target_loss = curr_loss - beta * good_epsilon * good_norm_grad**2

        # Now, we look at what happen if we do one gradient step with the current value of good_epsilon
        topomean.F = curr_F - good_epsilon*good_single_grad[0]  # The new simplex values
        loss = topomean.call(epoch=epoch)  # the corresponding loss

        counter = 0
        if verbosity_level >= 2:
            print("\nLoss before update (curr_loss) and before the eps-reduction (possibly) starts: %5f"
                  %curr_loss.numpy())
            print("Loss after update (loss) before the eps-reduction (possibly) starts: %5f" %loss.numpy())
            print("Corresponding target loss: %5f" %target_loss.numpy())
            print("Difference between the two losses (if positive, enter the loop): %5f"
                  %(loss.numpy() - target_loss.numpy()))
            print("(Recall) Initial value for epsilon is: %5f \n" %epsilon)

        while loss >= target_loss:
            counter += 1
            if verbosity_level >= 2:
                print("\nI'm in the gradient sampling `while` loop since %s step." %counter)
            if counter > 100:
                sys.exit('counter reached max value (100), something is weird.')
            # We decrease epsilon (step size) as previous value was too large
            good_epsilon *= gamma
            # We compute the new target loss (should be larger than the previous target)
            target_loss = curr_loss - beta * good_epsilon * good_norm_grad**2
            # We compute the new simplex values using updated epsilon
            topomean.F = curr_F - good_epsilon * good_single_grad[0]
            # And the corresponding loss.
            loss = topomean.call(epoch=epoch)

            if verbosity_level >= 2:
                print("Reference state (curr_F) is:", curr_F.numpy())
                print("Gradient at curr_F is:", good_single_grad[0].numpy())
                print("Its norm is: %5f" %good_norm_grad)
                print("good_epsilon is:", good_epsilon)
                print("Updated state (curr_F - eps * grad) is:", topomean.F.numpy())
                print("Updated loss (loss) after step is: %5f" %loss.numpy())
                print("Target loss is: %5f" %target_loss.numpy())
                print("Difference between loss and target (if positive, loop continues): %5f"
                      %(loss.numpy() - target_loss.numpy()))

        topomean.F = curr_F

    return good_epsilon, good_single_grad, good_norm_grad


def _sample_noise_ball(num_pts, dim):
    """
       sample num_pts points uniformly on a dim-ball.
    """
    x = np.random.randn(num_pts, dim)  # uncorrelated multivariate normal distrib
    s = np.divide(x, np.linalg.norm(x, axis=1)[:,np.newaxis])  # Uniform distrib on dim-sphere
    r = np.random.rand(num_pts)**(1/dim)  # scaling for radius
    res = np.multiply(r[:,np.newaxis], s)
    return res


def compute_gradient(epoch, dgms, gradients, funcs, topomean, loop_epsilon=True):
    """
    One of the core function of the work.

    :param epoch: The current epoch we are in (for print purpose)
    :param dgms:
    :param gradients:
    :param funcs:
    :param topomean: The topomean state.
    :param loop_epsilon:
    :param verbosity_level: Verbosity level, 0 silent, 1 standard, 2 debug.
    :return:
    """
    continue_iterations = True

    verbosity_level = topomean.verbosity_level

    if verbosity_level >= 1:
        print('Epoch %s ' %epoch)

    # stopping_criterion: Trick to compare with Vanilla "if we had access to strata"
    if topomean.mode == 'strata':

        #mode_tmp = topomean.mode
        topomean.mode = 'strata'

        with tf.GradientTape(persistent=True) as losstape:
            losses, perms, dists = topomean.call(epoch=epoch)

        if verbosity_level >= 3:  #TODO why is losses[0] containing a single value? Should we rename it loss?
            print("All losses", losses[0])

        grads = [losstape.gradient(losses[i], topomean.trainable_variables)[0]
                 for i in range(topomean.curr_visited)]

        try:
            grads = np.vstack([g.numpy()[None,:] for g in grads])
        except AttributeError:
            for g in grads:
                unique_indices, new_index_positions = tf.unique(g.indices)
                summed_values = tf.math.unsorted_segment_sum(g.values, new_index_positions, tf.shape(unique_indices)[0])
                #TODO : weird, are we really modifying the g here ?...
                # Seems so as the result is correct, but must check.
                g = tf.IndexedSlices(indices=unique_indices, values=summed_values, dense_shape=g.dense_shape)
            grads = np.vstack([tf.sparse.to_dense(
                tf.sparse.reorder(
                    tf.sparse.SparseTensor(
                        tf.cast(g.indices[:,None], tf.int64),
                                g.values,
                                tf.convert_to_tensor(g.shape, tf.int64)
                    )
                ),
                validate_indices=False).numpy()[np.newaxis,:]
            for g in grads])

        grads = np.array([g[perms[i]] for i,g in enumerate(grads)])

        G, single_grad, norm_grad = _compute_single_gradient(grads, verbosity_level=verbosity_level)
        good_epsilon, good_single_grad, good_norm_grad = _reduce_gradient(topomean,
                                                                          grads, single_grad, norm_grad,
                                                                          loop_epsilon, dists, 0)

        if good_norm_grad > topomean.eta:
            a = 2  # parameter to handle that we do not have access to the distance to strata, but an upper bound
            good_single_grad[0] = tf.multiply(good_single_grad[0], good_epsilon/(a*good_norm_grad))
        else:
            continue_iterations = False


    if topomean.mode == 'gradient_sampling':
        nb_pts_sample = topomean.card_strata  # number of point we sample around the current state
        # We randomly sample nb_pts_sample noise values in (point in B(0, 1))
        noisex = _sample_noise_ball(nb_pts_sample, topomean.F.shape[0])
        grads = []
        curr_F = topomean.F.numpy()
            
        for ic in range(1+nb_pts_sample):

            # We rescale the noise values to put them in B(0, epsilon)
            # The first noise is 0 so that we keep the current state as a gradient
            if ic == 0:
                noise = tf.constant(np.array(np.zeros(shape=topomean.F.shape)))
            else:
                noise = tf.constant(topomean.epsilon*noisex[ic-1])

            # We update topomean current state (temporary) by adding this small noise
            if topomean.verbosity_level >= 2:
                print("Curr_F + noise:", (curr_F + noise).numpy())
            topomean.F = tf.Variable(curr_F + noise)

            # We compute the loss at the new position (will make gradient available)
            with tf.GradientTape() as losstape:
                loss = topomean.call(epoch=epoch)

            # We get the corresponding gradient and add it to our grads list.
            grad_at_point_plus_noise = losstape.gradient(loss, topomean.trainable_variables)[0]
            unique_indices, new_index_positions = tf.unique(grad_at_point_plus_noise.indices)
            summed_values = tf.math.unsorted_segment_sum(grad_at_point_plus_noise.values,
                                                         new_index_positions,
                                                         tf.shape(unique_indices)[0]
                            )
            grad_at_point_plus_noise = tf.IndexedSlices(indices=unique_indices,
                                                        values=summed_values,
                                                        dense_shape=grad_at_point_plus_noise.dense_shape)
            grads.append(grad_at_point_plus_noise)

            if topomean.verbosity_level >= 2:
                print("Grad at curr_F + noise:", grad_at_point_plus_noise)

        # We set vertices values back to their original values (and then add the subsequent noise).
        topomean.F = tf.Variable(curr_F)

        try:
            grads = np.vstack([g.numpy()[None,:] for g in grads])
        except AttributeError:
            grads = np.vstack([
                tf.sparse.to_dense(
                    tf.sparse.reorder(
                        tf.sparse.SparseTensor(
                            tf.cast(g.indices[:,None],
                                    tf.int64),
                                    g.values,
                                    tf.convert_to_tensor(g.shape,
                                                         tf.int64)
                        )
                    ),
                validate_indices=False).numpy()[np.newaxis,:]
            for g in grads])


        # Now, we reduce the grads list in a single_grad object (min on convex hull).
        G, single_grad, norm_grad = _compute_single_gradient(grads, topomean.verbosity_level)

        # We first check if we reached the stopping criterion (in particular if grad = 0, we can't reduce epsilon).
        if norm_grad < topomean.eta:
            continue_iterations = False
            good_single_grad = single_grad
            good_norm_grad = norm_grad
        else:
            # We obtain updated parameters (i.e. epsilon that ensures the loss decreases).
            good_epsilon, good_single_grad, good_norm_grad = _reduce_gradient(topomean,
                                                                              grads, single_grad, norm_grad,
                                                                              loop_epsilon=loop_epsilon, epoch=epoch)
            if topomean.normalize_gradient:
                good_single_grad[0] = tf.multiply(good_single_grad[0], good_epsilon / good_norm_grad)
            else:
                good_single_grad[0] = tf.multiply(good_single_grad[0], good_epsilon)
            # We reached the stopping criterion (for gradient sampling).
            if good_norm_grad < topomean.eta:
                continue_iterations = False

    elif topomean.mode == 'vanilla':

        with tf.GradientTape() as losstape:
            loss = topomean.call(epoch=epoch)
        good_single_grad = losstape.gradient(loss, topomean.trainable_variables)
        try:
            G = good_single_grad[0].numpy()
        except AttributeError:
            G = np.array(good_single_grad[0].values)
        good_norm_grad = np.linalg.norm(G)
        if topomean.normalize_gradient:
            good_single_grad[0] = tf.multiply(good_single_grad[0], 1 / good_norm_grad)
        if topomean.vanilla_decay:
            good_single_grad[0] = tf.multiply(good_single_grad[0], 1/(1 + epoch))

        if verbosity_level >= 2 :
            print(good_single_grad[0])

        if good_norm_grad < topomean.eta:
            continue_iterations = False


    # Store the list of diagrams, functions, (reduced) gradients, losses, crossed at each steps.
    # Used for plotting the optimisation history.
    dgms.append(topomean.dgm.numpy())
    funcs.append(topomean.F.numpy())
    gradients.append((G, good_norm_grad))

    try:  # dirty trick while gudhi 3.5 is not available on conda-forge
        loss = tf.add_n([wass.wasserstein_distance(topomean.dgm, tf.constant(D),
                                                   order=topomean.order,
                                                   internal_p=topomean.internal_p,
                                                   enable_autodiff=True,
                                                   keep_essential_parts=False)**topomean.order
                         for D in topomean.L]).numpy()
    except TypeError:  # if keep_essential_parts is not found, i.e. gudhi.__version__ < 3.5.
        loss = tf.add_n([wass.wasserstein_distance(topomean.dgm, tf.constant(D),
                                                   order=topomean.order,
                                                   internal_p=topomean.internal_p,
                                                   enable_autodiff=True)**topomean.order
                         for D in topomean.L]).numpy()
    topomean.losseslist.append(loss)

    if verbosity_level >= 1:
        print('\n*****\n')
        print("After epoch %s:" %epoch)
        print('Loss = %.5f' %loss)
        print('Gradient norm = %.5f' %good_norm_grad)
        print('\n*****\n')

    return continue_iterations, good_single_grad


def apply_gradient(gradient, optimizer, topomean):
    optimizer.apply_gradients(zip(gradient, topomean.trainable_variables))
