import numpy as np
from scipy.linalg import expm
from collections import defaultdict
from python_code.data_utils import haplotype_columns
from functools import lru_cache

# explicit calculation of the transition matrix for a continuous-time Markov chain
def calculate_transition_matrix(mu, num_states = 2):
    """Calculate the transition matrix."""
    Q = np.full((num_states, num_states), mu)
    np.fill_diagonal(Q, -(num_states - 1) * mu)
    return Q

# get the probability matrix from the transition matrix Q
def probability_matrix(Q, t):
    return expm(Q * t)

# P cached for branch lengths
@lru_cache(maxsize=200_000)
def cached_probability_matrix(mu, num_states, t):
    """P = expm(Q t) for the Mk rate matrix, remembered for each (mu, num_states, t)."""
    return probability_matrix(calculate_transition_matrix(mu, num_states), t)


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
def felsenstein_backwards_general(tree, tip_states, bases=(0, 1), mutation_rate=1.0, root_state=None):
    """
    Felsenstein's pruning algorithm on one tree for one site: the marginal
    log-likelihood, and the best (joint) reconstruction of internal states.

    Parameters
    ----------
    tree : tskit.Tree
        Tree to run on. Branch lengths are in generations.
    tip_states : dict
        Mapping from tip node IDs to observed states (values must be in `bases`).
    bases : sequence, default (0, 1)
        Possible states, in the order used by the transition matrix. Can be e.g. ("A", "C", "G", "T").
    mutation_rate : float, default 1.0
        Total mutation rate per site per generation. Divided by (number of states - 1)
        to give mu, the rate of switching to each other state.
    root_state : optional
        If None, the root is free: the marginal averages over root states (1/k each),
        and the joint picks the best root state.
        Otherwise, the root is fixed to this state; must belong to `bases`.

    Returns
    -------
    log_marginal : float
        Log-likelihood of the tip states, summed over all histories
        (given the root state, if root_state was set).
    states : dict
        The best (joint) reconstruction: node ID -> state.
    log_path_probability : float
        Log probability of that best reconstruction together with the tip states.
    num_mutations : int
        Number of branches where the best reconstruction changes state.

    Notes
    -----
    Values are rescaled at every node, so they do not underflow on large trees
    or sites needing many changes.
    """

    num_states = len(bases)
    base_index = {b: i for i, b in enumerate(bases)}   # label -> row/column in P
    mu = mutation_rate / (num_states - 1)

    likelihoods = {}  # marginal: partial likelihoods for each node (rescaled, see below)
    best = {}  # joint: probability of the best assignment below each node (rescaled)
    choice = {}  # joint: choice[child][i] = best state for the child if its parent is in state i

    # rescaling: after each internal node, divide its vector by its largest entry so the
    # numbers never get tiny, and add log(that factor) to a running total. Every node's
    # factor multiplies into the root exactly once, so the true root value is
    # (rescaled root value) * exp(running total).
    log_scale_marginal = 0.0
    log_scale_joint = 0.0

    for u in tree.nodes(order="postorder"):
        # set likelihoods for the tips
        if tree.is_leaf(u):
            likelihoods[u] = np.zeros(num_states)
            likelihoods[u][base_index[tip_states[u]]] = 1.0
            best[u] = likelihoods[u].copy()
            continue

        likelihoods[u] = np.ones(num_states) # Initialize internal nodes with uniform likelihoods, 1 since we will multiply in the children's contributions
        best[u] = np.ones(num_states)

        for child in tree.children(u):
            branch_length = tree.branch_length(child)
            # this cached version means if we have calculated a branch w this length before we can speed up 
            P = cached_probability_matrix(mu, num_states, branch_length)
            # marginal: sum over all possible states of the child, weighted by the transition probabilities
            child_states_sum = P @ likelihoods[child]
            # multiply the parent's likelihoods by the sum of the child's states
            likelihoods[u] *= child_states_sum

            # joint: same products, but keep the largest
            scores = P * best[child]
            # for each parent state i, remember which child state j was best
            choice[child] = scores.argmax(axis=1)
            # multiply the parent's best by the child's best score
            best[u] *= scores.max(axis=1)

        # rescale: divide this node's vectors by their largest entry so the numbers stay
        # near 1 instead of shrinking towards 0 as we go up the tree. We keep the log of
        # each factor we divide by, and add them all back at the root, so the final
        # answer is unchanged.
        # marginal
        s = likelihoods[u].max()
        if s > 0:                                  # all zeros means the data are impossible; leave as is
            likelihoods[u] /= s
            log_scale_marginal += np.log(s)        # remember what we divided out
        # joint
        s = best[u].max()
        if s > 0:
            best[u] /= s
            log_scale_joint += np.log(s)

    root = tree.root
    with np.errstate(divide="ignore"):             # log(0) gives -inf instead of a warning (impossible data)

        # marginal: probability of the tips summed over all histories
        # true root value = rescaled root value * exp(total of the logs we divided out)
        if root_state is None:
            # root free: average over the k root states, each with probability 1/k
            log_marginal = np.log(likelihoods[root].mean()) + log_scale_marginal
        else:
            # root fixed: take the entry for that state
            log_marginal = np.log(likelihoods[root][base_index[root_state]]) + log_scale_marginal

        # joint: probability of the single best history
        if root_state is None:
            # root free: pick the root state with the best score, each state weighted by 1/k
            root_scores = (1 / num_states) * best[root]
            root_index = root_scores.argmax()
        else:
            # root fixed: use that state (read its index directly, don't argmax)
            root_index = base_index[root_state]
            root_scores = np.zeros(num_states)
            root_scores[root_index] = best[root][root_index]
        # add back what we divided out on the way up
        log_path_probability = np.log(root_scores[root_index]) + log_scale_joint

    # joint: traceback from the root to the leaves, reading off the stored choices
    states = {root: bases[root_index]}
    num_mutations = 0
    for u in tree.nodes(order="preorder"):
        if u != root:
            parent_index = base_index[states[tree.parent(u)]]
            states[u] = bases[choice[u][parent_index]]
            if states[u] != states[tree.parent(u)]:
                num_mutations += 1

    return log_marginal, states, log_path_probability, num_mutations

import math

# two state felsenstein
def felsenstein_backwards_two_state(tree, tip_states, bases=(0, 1), mutation_rate=1.0, root_state=None):
    """
    Same as felsenstein_backwards_general, for exactly two states, written with
    plain Python numbers instead of numpy arrays (much faster for 2 states).
    Every step matches the general version line for line.
    """
    if len(bases) != 2:
        raise ValueError("felsenstein_backwards_two_state needs exactly two states")
    num_states = 2
    base_index = {b: i for i, b in enumerate(bases)}   # label -> 0 or 1
    mu = mutation_rate / (num_states - 1)               # with 2 states, mu = mutation_rate

    # each node's vector is a pair (value for state 0, value for state 1)
    likelihoods = {}  # marginal: partial likelihoods for each node (rescaled)
    best = {}  # joint: probability of the best assignment below each node (rescaled)
    choice = {}  # joint: choice[child][i] = best state for the child if its parent is in state i

    # rescaling: same as the general version
    log_scale_marginal = 0.0
    log_scale_joint = 0.0

    for u in tree.nodes(order="postorder"):
        # set likelihoods for the tips: 1 for the observed state, 0 for the other
        if tree.is_leaf(u):
            i = base_index[tip_states[u]]
            likelihoods[u] = (1.0, 0.0) if i == 0 else (0.0, 1.0)
            best[u] = likelihoods[u]
            continue

        L0, L1 = 1.0, 1.0   # likelihoods[u], start at 1 since we multiply in the children's contributions
        B0, B1 = 1.0, 1.0   # best[u]

        for child in tree.children(u):
            branch_length = tree.branch_length(child)
            # P = expm(Q t) for 2 states has a closed form:
            #   P[i, j] = p_same if i == j else p_diff,  p_diff = (1 - exp(-2 mu t)) / 2
            p_diff = -math.expm1(-num_states * mu * branch_length) / num_states
            p_same = 1.0 - p_diff
            c0, c1 = likelihoods[child]
            b0, b1 = best[child]

            # marginal: child_states_sum = P @ likelihoods[child]
            child_states_sum_0 = p_same * c0 + p_diff * c1    # parent in state 0
            child_states_sum_1 = p_diff * c0 + p_same * c1    # parent in state 1
            # multiply the parent's likelihoods by the sum of the child's states
            L0 *= child_states_sum_0
            L1 *= child_states_sum_1

            # joint: same products, but keep the largest
            # scores[i, j] = P[i, j] * best[child][j]   (parent in state i, child in state j)
            s00, s01 = p_same * b0, p_diff * b1     # parent 0: child 0, child 1
            s10, s11 = p_diff * b0, p_same * b1     # parent 1: child 0, child 1
            # for each parent state i, remember which child state j was best  (= scores.argmax(axis=1))
            choice[child] = (0 if s00 >= s01 else 1, 0 if s10 >= s11 else 1)
            # multiply the parent's best by the child's best score              (= scores.max(axis=1))
            B0 *= max(s00, s01)
            B1 *= max(s10, s11)

        # rescale: divide this node's vectors by their largest entry so the numbers stay
        # near 1, keep the log of what we divided out, add it back at the root
        # marginal
        s = max(L0, L1)
        if s > 0:                                  # all zeros means the data are impossible; leave as is
            L0, L1 = L0 / s, L1 / s
            log_scale_marginal += math.log(s)      # remember what we divided out
        # joint
        s = max(B0, B1)
        if s > 0:
            B0, B1 = B0 / s, B1 / s
            log_scale_joint += math.log(s)

        likelihoods[u] = (L0, L1)
        best[u] = (B0, B1)

    root = tree.root
    log = lambda v: math.log(v) if v > 0 else -math.inf   # log(0) gives -inf (impossible data)

    # marginal: probability of the tips summed over all histories
    if root_state is None:
        # root free: average over the 2 root states, each with probability 1/2
        log_marginal = log(sum(likelihoods[root]) / num_states) + log_scale_marginal
    else:
        # root fixed: take the entry for that state
        log_marginal = log(likelihoods[root][base_index[root_state]]) + log_scale_marginal

    # joint: probability of the single best history
    if root_state is None:
        # root free: pick the root state with the best score, each state weighted by 1/2
        root_scores = (best[root][0] / num_states, best[root][1] / num_states)
        root_index = 0 if root_scores[0] >= root_scores[1] else 1
    else:
        # root fixed: use that state
        root_index = base_index[root_state]
        root_scores = (best[root][0], best[root][1])
    # add back what we divided out on the way up
    log_path_probability = log(root_scores[root_index]) + log_scale_joint

    # joint: traceback from the root to the leaves, reading off the stored choices
    states = {root: bases[root_index]}
    num_mutations = 0
    for u in tree.nodes(order="preorder"):
        if u != root:
            parent_index = base_index[states[tree.parent(u)]]
            states[u] = bases[choice[u][parent_index]]
            if states[u] != states[tree.parent(u)]:
                num_mutations += 1

    return log_marginal, states, log_path_probability, num_mutations



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