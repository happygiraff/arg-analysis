import pandas as pd
import json
import tskit
import numpy as np

def haplotype_columns(df):
    """Columns named with a number (5, 6, ..., or "5", "6", ...), in numeric order."""
    cols = [c for c in df.columns if str(c).isdigit()]
    return sorted(cols, key=int)

def mutation_rate_from_provenance(ts):
    """The rate passed to msprime.sim_mutations, read from the provenance records."""
    rates = []
    for p in ts.provenances():
        record = json.loads(p.record)
        params = record.get("parameters", {})
        if params.get("command") == "sim_mutations":
            rates.append(params.get("rate"))
    return rates

def mutation_branches(tree, states):
    """Branches where the reconstruction changes state: (child node, parent node)."""
    return [(u, tree.parent(u)) for u in states
            if tree.parent(u) != -1 and states[u] != states[tree.parent(u)]]


def clade_membership(tree, *, include_singletons=True, include_root=True):
    """Return node IDs and a binary sample-membership matrix for a tree.

    Parameters
    ----------
    tree : tskit.Tree
        Tree whose clades will be represented.
    include_singletons : bool, optional
        Include nodes with exactly one descendant sample. Defaults to True.
    include_root : bool, optional
        Include the root node as a clade. Defaults to True.

    Returns
    -------
    node_ids : numpy.ndarray
        IDs of the nodes represented by the rows.
    sample_ids : numpy.ndarray
        IDs of the samples represented by the columns.
    membership : numpy.ndarray
        Binary matrix where each row indicates which samples belong to a node.
    """
    sample_ids = tree.tree_sequence.samples()
    column = {int(s): j for j, s in enumerate(sample_ids)}

    node_ids = np.array([
        u for u in tree.nodes(order="postorder")
        if (include_singletons or tree.num_samples(u) >= 2)
        and (include_root or not tree.is_root(u))
    ], dtype=np.int32)

    membership = np.zeros(
        (len(node_ids), len(sample_ids)), dtype=np.uint8
    )

    for i, u in enumerate(node_ids):
        columns = [column[s] for s in tree.samples(int(u))]
        membership[i, columns] = 1

    return node_ids, sample_ids, membership


def correlate_clades(membership, carrier):
    """Correlate each clade's sample membership with binary carrier status.

    Returns a Pearson correlation for every row of ``membership``. Rows that
    have no variation across samples receive ``NaN``.
    """

    X = np.asarray(membership, dtype=float)
    y = np.asarray(carrier, dtype=float)

    if X.ndim != 2 or y.shape != (X.shape[1],):
        raise ValueError("Provide one carrier value per membership column.")
    if not np.isin(X, [0, 1]).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Membership and carrier arrays must contain only 0 and 1.")

    X_centered = X - X.mean(axis=1, keepdims=True)
    y_centered = y - y.mean()

    if np.linalg.norm(y_centered) == 0:
        raise ValueError("Carrier array must contain both carriers and non-carriers.")

    denominator = (
        np.linalg.norm(X_centered, axis=1)
        * np.linalg.norm(y_centered)
    )

    correlations = np.full(X.shape[0], np.nan)
    np.divide(
        X_centered @ y_centered,
        denominator,
        out=correlations,
        where=denominator > 0,
    )
    return correlations

def clade_correlations(tree, carrier, *, include_singletons=True, include_root=True):
    """
    Correlation between a SNP and every clade of a tree, without building the
    membership matrix. Gives the same result as
    correlate_clades(*clade_membership(tree, ...)[::2], carrier) but much faster.

    Parameters
    ----------
    tree : tskit.Tree
    carrier : array-like
        0/1 value per sample, in tree.tree_sequence.samples() order.
    include_singletons, include_root : bool
        Same meaning as in clade_membership.

    Returns
    -------
    node_ids : np.ndarray
        Nodes in postorder (same order as clade_membership).
    correlations : np.ndarray
        Pearson (phi) correlation between the carrier vector and "is below this node";
        NaN where it is undefined (e.g. the root, whose clade is every sample).
    """
    sample_ids = tree.tree_sequence.samples()
    y = np.asarray(carrier)
    if y.shape != (len(sample_ids),):
        raise ValueError("Provide one carrier value per sample.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Carrier array must contain only 0 and 1.")
    n, c = len(y), int(y.sum())
    if c == 0 or c == n:
        raise ValueError("Carrier array must contain both carriers and non-carriers.")

    is_carrier = dict(zip(sample_ids.tolist(), y.tolist()))
    carriers_below = {}
    node_ids, correlations = [], []
    for u in tree.nodes(order="postorder"):
        # carriers below u = u itself (if it is a carrier sample) + carriers below each child
        carriers_below[u] = is_carrier.get(u, 0) + sum(carriers_below[ch] for ch in tree.children(u))
        b = tree.num_samples(u)                       # samples below u
        if not (include_singletons or b >= 2) or not (include_root or not tree.is_root(u)):
            continue
        a = carriers_below[u]
        denom = np.sqrt(b * (n - b) * c * (n - c))
        node_ids.append(u)
        correlations.append((a * n - b * c) / denom if denom > 0 else np.nan)
    return np.array(node_ids, dtype=np.int32), np.array(correlations)