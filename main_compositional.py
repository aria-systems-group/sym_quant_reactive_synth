"""
 In this script we write the main for constructing the abstraction for FrankaDyanmic (Game) in Compositional Manner.  
"""

from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFA

from cudd import Cudd, ADD

if __name__ == "__main__":
    # testing things
    manager = Cudd()
    dfa_handle = SymbolicPartitionedDFA(formula='F(p01)', manager=manager, ltlf_flag=True)

    print("****************** qVars Map: ********************")
    for k, v in dfa_handle.qVar_map_sym.items():
        print(f"{k}: {v}")
