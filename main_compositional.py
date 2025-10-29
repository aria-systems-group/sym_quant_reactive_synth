"""
 In this script we write the main for constructing the abstraction for FrankaDyanmic (Game) in Compositional Manner.  
"""

from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot
from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame

from cudd import Cudd, ADD

if __name__ == "__main__":
    # setting things up
    boxes = 1
    locs = 2
    ratio = 1

    # init = ['ready l2', 'b0 l2', 'b1 l4']
    # goal = [['b0 l1']]
    init = ['ready l2', 'b0 l2']
    goal = [['b0 l1']]

    human_locs = range(1, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = []

    formula = 'F(p01)'

    dfa_game = SymbolicPartitionedDFAGame(boxes=boxes, locs=locs,
                                          ratio=ratio, init=init,
                                          goal=goal, formula=formula, 
                                          restricted_human_locs=human_locs,
                                          ltlf_flag=True)

    # print Game Info?

    # print DFA Info?
