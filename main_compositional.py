"""
 In this script we write the main for constructing the abstraction for FrankaDyanmic (Game) in Compositional Manner.  
"""
import time
import math

from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame

from cudd import Cudd, ADD

if __name__ == "__main__":
    # setting things up
    boxes = 2
    locs = 3
    ratio = 1

    init = ['ready l2', 'b0 l2', 'b1 l3']
    # init = ['ready l2', 'b0 l2']
    # goal = [['b0 l1']]
    goal = []

    # human_locs = range(1, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    human_locs = [3]

    formula = 'F(p01 & F(p02 & F(p01)))'

    dfa_game = SymbolicPartitionedDFAGame(boxes=boxes, locs=locs,
                                          ratio=ratio, init=init,
                                          goal=goal, formula=formula, 
                                          restricted_human_locs=human_locs,
                                          ltlf_flag=True)

    # print Game Info?
    print("*****************Printing Game Info*****************")
    print('****************xVars Map:****************')
    for k, v in dfa_game.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************rAction Map:****************')
    for k, v in dfa_game.rAction_map.items():
        print(f"{k} : {v}")

    print('****************eAction Map:****************')
    for k, v in dfa_game.eAction_map.items():
        print(f"{k} : {v}")

    print("*****************Ratio Map:*****************")
    for k, v in dfa_game.kVar_map.items():
        print(f"{k} : {v}")


    # print the number of explicit states
    sys_states = (ratio + 1)*(pow(locs + 1, 3) + boxes)*(math.factorial(locs+1) // math.factorial(locs+1 - boxes))
    env_states = (ratio + 1)*(pow(locs + 1, 2) + boxes*(locs+1))*(math.factorial(locs+1) // math.factorial(locs+1 - boxes))
    print("Total num of explicit states in Game: ", env_states + sys_states)

    # print DFA Info?
    print("*****************Printing Game Info*****************")
    for k, v in dfa_game.dfa_handle.qVar_map.items():
        print(f"{k} : {v}")


    # print DFA Game Info?
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(dfa_game.latches) + len(dfa_game.qVars))
    print("Total num of prime latches: ", len(dfa_game.prime_latches) + len(dfa_game.prime_qVars))
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.prime_latches) + len(dfa_game.qVars) + len(dfa_game.prime_qVars))

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # dfa_game.test_pre_image()

    tic = time.time()
    strategy = dfa_game.solve(verbose=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # if strategy is not None:
    #     dfa_game.roll_out_strategy(strategy=strategy, verbose=True)
