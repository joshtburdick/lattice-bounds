import pytest
from lattice_bounds.lattice import CliqueLattice

def test_hypergraph_count():
    n = 5
    k = 3
    lattice = CliqueLattice(n, k)
    
    num_cliques = len(lattice.cliques)
    expected_hypergraph_count = 2 ** num_cliques
    
    assert len(lattice.hypergraphs) == expected_hypergraph_count
