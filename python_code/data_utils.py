import pandas as pd
import json
import tskit
import numpy

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