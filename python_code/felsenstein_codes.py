import numpy as np
from scipy.linalg import expm
from collections import defaultdict
from python_code.data_utils import haplotype_columns

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
def felsenstein_backwards_general(tree, tip_states, bases=(0, 1), mutation_rate= 1.0, root_state = None):
    """
    Calculate the likelihood of the tip states using Felsenstein's algorithm.

    Parameters
    ----------
    tree : tskit.Tree
        Tree on which to calculate the likelihood.
    tip_states : dict
        Mapping from tip node IDs to observed states.
    bases : sequence, default (0, 1). can use ACTG etc
        Possible states, in the order used by the transition matrix.
    mutation_rate : float, optional
        Mutation rate. This will be divided by number of states -1 to get mu, transition of switching from state to state
    root_state : optional
        If None, average over root states using the root prior.
        Otherwise, condition on this root state; must belong to bases.

    Returns
    -------
    likelihoods, states, path_probability, num_mutations
    """

    num_states = len(bases) 
    base_index = {b: i for i, b in enumerate(bases)}   # label -> row/column in P
    mu = mutation_rate / (num_states - 1)
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
        root_index = root_scores.argmax()                    
    else:
        root_index = base_index[root_state]                  
        root_scores = np.zeros(num_states)
        root_scores[root_index] = best[root][root_index]
    path_probability = root_scores[root_index]               

    # joint: traceback from the root to the leaves, reading off the stored choices
    states = {root: bases[root_index]}                       # CHANGED (was root_scores.argmax())
    num_mutations = 0
    for u in tree.nodes(order="preorder"):
        if u != root:
            parent_index = base_index[states[tree.parent(u)]]
            states[u] = bases[choice[u][parent_index]]
            if states[u] !=states[tree.parent(u)]:
                num_mutations +=1


            
    return likelihoods, states, path_probability, num_mutations


def snps_by_tree_index(ts, snp_positions):
    """
    Map SNP positions to their tree indices, excluding invalid or empty trees. This is used to cut down calculation time on Felsenstein since multiple SNPs may 
    be associated to the same tree.

    Parameters
    ----------
    ts : tskit.TreeSequence
        Tree sequence containing the trees and sequence boundaries.
    snp_positions : array-like
        Genomic positions of SNPs to assign to trees.

    Returns
    -------
    dict
        Mapping from tree index to the list of SNP positions in that tree.
    """

    positions = np.asarray(snp_positions)

    breakpoints = ts.breakpoints(as_array=True)
    tree_indices = np.searchsorted(breakpoints, positions, side="right") - 1

    # skip SNPs outside the sequence, or on a tree with no edges (e.g. before Relate's first tree, in masked relate regions, or snps that are left out at the end of the chromosome)
    inside = (positions >= 0) & (positions < ts.sequence_length)
    empty = np.array([tree.num_edges == 0 for tree in ts.trees()])
    keep = inside.copy()
    keep[inside] = ~empty[tree_indices[inside]]
    if not keep.all():
        print(f"skipped {(~keep).sum()} SNPs outside the sequence or on empty trees")

    result = defaultdict(list)
    for tree_index, pos in zip(tree_indices[keep].tolist(), positions[keep].tolist()):
        result[tree_index].append(pos)

    return dict(result)


def tip_states_at(df, pos):
    """
    Get the 0/1 tip states for the SNP at some positon `pos`. This is fed into the Felsensetin general alg.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing an "SNP position" column and haplotype columns.
    pos : float or int
        Genomic position of the SNP to retrieve.

    Returns
    -------
    dict
        Mapping from sample/haplotype node number to its 0/1 state.
    """
    
    hap_cols = haplotype_columns(df) # get only the 0/1 haplotype columns
    row = df.loc[df["SNP position"] == pos, hap_cols]
    if len(row) != 1:
        raise ValueError(f"{len(row)} rows at position {pos}")
    haps = row.to_numpy().ravel().astype(int)
    return dict(enumerate(haps))      # {hap 0: state, hap 1: state, ...}