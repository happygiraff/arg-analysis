import numpy as np
from scipy.linalg import expm

# explicit calculation of the transition matrix for a continuous-time Markov chain
def calculate_transition_matrix(mu, num_states = 2):
    """Calculate the transition matrix."""
    Q = np.full((num_states, num_states), mu)
    np.fill_diagonal(Q, -(num_states - 1) * mu)
    return Q

# get the probability matrix from the transition matrix Q
def probability_matrix(Q, t):
    return expm(Q * t)


#jukes cantor model probabilities
def calculate_jukes_cantor_prob(t_branch):
    """Calculate the transition matrix for a Jukes-Cantor model."""
    p_same = 1/4 + 3/4 * np.exp(-4*t_branch/3)
    p_diff = 1/4 - 1/4 * np.exp(-4*t_branch/3)

    P = np.full((4, 4), p_diff)
    np.fill_diagonal(P, p_same)

    return P

# felsenstien jukes cantor w ACTG (this is just a specific instance of the general felsenstein later)
def felsenstein_backwards_jc(tree, tip_states):
    "felsenstein algorithm to calculate the likelihood of the tree given the tip states."

    bases = ["A", "C", "G", "T"]
    base_index = {base: i for i, base in enumerate(bases)}

    likelihoods = {}  # Dictionary to store likelihoods for each node
    best = {}  # joint: probability of the best assignment below each node
    choice = {}  # joint: choice[child][i] = best state for the child if its parent is in state i


    for u in tree.nodes(order="postorder"):
            
            # set likelihoods for the tips
            if tree.is_leaf(u):
                likelihoods[u] = np.zeros(4)
                likelihoods[u][base_index[tip_states[u]]] = 1.0
                best[u] = likelihoods[u].copy()
                continue
            
            likelihoods[u] = np.ones(4)  # Initialize internal nodes with uniform likelihoods, 1 since we will multiply in the children's contributions
            best[u] = np.ones(4)

            # go through internal nodes
            for child in tree.children(u):
                branch_length = tree.branch_length(child)
                # later we can upgrade this so that we can save the transition matrix for each branch length, but for now we will just calculate it on the fly
                P = calculate_jukes_cantor_prob(branch_length)
                # sum over all possible states of the child, weighted by the transition probabilities
                child_states_sum = P @ likelihoods[child]
                # multiply the parent's likelihoods by the sum of the child's states
                likelihoods[u] *= child_states_sum
                # print("likelihoods:", likelihoods[u])


                # joint: same products, but keep the largest
                # scores[i, j] = P[i, j] * best[child][j]   (parent in state i, child in state j)
                scores = P * best[child]
                # for each parent state i, remember which child state j was best
                choice[child] = scores.argmax(axis=1)
                # multiply the parent's best by the child's best score
                best[u] *= scores.max(axis=1)

    # joint: the root picks its own state (jc: every base has frequency 1/4)
    root = tree.root
    root_scores = 0.25 * best[root]
    path_probability = root_scores.max()

    # joint: traceback from the root to the leaves, reading off the stored choices
    states = {root: bases[root_scores.argmax()]}
    num_mutations = 0
    for u in tree.nodes(order="preorder"):
        if u != root:
            parent_index = base_index[states[tree.parent(u)]]
            states[u] = bases[choice[u][parent_index]]
            if states[u] !=states[tree.parent(u)]:
                num_mutations +=1


            
    return likelihoods, states, path_probability, num_mutations


# general felsenstein function that can handle any number of states, including 0/1 and ACTG. We can also impose the root to be the ancestral state
def felsenstein_backwards_general(tree, tip_states, bases=(0, 1), mu= None, root_state = None):
    "felsenstein algorithm to calculate the likelihood of the tree given the tip states"

    num_states = len(bases) 
    base_index = {b: i for i, b in enumerate(bases)}   # label -> row/column in P
    if mu is None:
        mu = 1 / (num_states - 1)
    Q = calculate_transition_matrix(mu, num_states)

    likelihoods = {}  # Dictionary to store likelihoods for each node
    best = {}  # joint: probability of the best assignment below each node
    choice = {}  # joint: choice[child][i] = best state for the child if its parent is in state i


    for u in tree.nodes(order="postorder"):
        #set likelihoods for the tips
        if tree.is_leaf(u):
            likelihoods[u] = np.zeros(num_states)
            likelihoods[u][base_index[tip_states[u]]] = 1.0
            best[u] = likelihoods[u].copy()
            continue
        
        likelihoods[u] = np.ones(num_states)  # Initialize internal nodes with uniform likelihoods, 1 since we will multiply in the children's contributions
        best[u] = np.ones(num_states)
                
        for child in tree.children(u):
            branch_length = tree.branch_length(child)
            # later we can upgrade this so that we can save the transition matrix for each branch length, but for now we will just calculate it on the fly
            P = probability_matrix(Q, branch_length)
            # sum over all possible states of the child, weighted by the transition probabilities
            child_states_sum = P @ likelihoods[child]
            # multiply the parent's likelihoods by the sum of the child's states
            likelihoods[u] *= child_states_sum
            # print("likelihoods:", likelihoods[u])


            # joint: same products, but keep the largest
            # scores[i, j] = P[i, j] * best[child][j]   (parent in state i, child in state j)
            scores = P * best[child]
            # for each parent state i, remember which child state j was best
            choice[child] = scores.argmax(axis=1)
            # multiply the parent's best by the child's best score
            best[u] *= scores.max(axis=1)

    # joint: the root picks its own state, or is fixed to root_state
    root = tree.root
    if root_state is None:
        root_scores = (1/num_states) * best[root]
    else:
        root_scores = np.zeros(num_states)
        root_scores[base_index[root_state]] = best[root][base_index[root_state]]
    path_probability = root_scores.max()

    # joint: traceback from the root to the leaves, reading off the stored choices
    states = {root: bases[root_scores.argmax()]}
    num_mutations = 0
    for u in tree.nodes(order="preorder"):
        if u != root:
            parent_index = base_index[states[tree.parent(u)]]
            states[u] = bases[choice[u][parent_index]]
            if states[u] !=states[tree.parent(u)]:
                num_mutations +=1


            
    return likelihoods, states, path_probability, num_mutations