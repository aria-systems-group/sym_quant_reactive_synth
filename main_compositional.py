"""
 In this script we write the main for constructing the abstraction for FrankaDyanmic (Game) in Compositional Manner.  
"""
import sys
import time
import math

from cudd import Cudd, ADD

from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame
from src.compositional_graphs.test_frankadynamic_ratio_else_tb import FrankaWorldDynamicRatioTurnBasedElse
from src.compositional_graphs.test_frankadynamic_ratio_tb import FrankaWorldDynamicRatioTurnBased

# regret script imports
from src.compositional_graphs.symbolic_partitioned_regret_dfa_game import SymbolicPartitionedRegretDFAGame


def Regret_DFA_Game_Main():
    # setting things up
    boxes = 1
    locs = 2
    ratio = 1
    budget = 4

    cooperative_game = True
    enable_reordering = False
    ltlf_flag = True

    # init = ['ready l6', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5']
    # init = ['ready l5', 'b0 l2', 'b1 l3', 'b2 l4']
    init = ['ready l3', 'b0 l2']
    # goal = [['b0 l1']]
    goal = []

    human_locs = range(1, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = [3]

    # formula = 'F(p01 & F(p02 & F(p01)))'
    # formula = 'F(p01 & p12)'
    # formula = 'F(p01 & F(p02))'
    formula = 'F(p01)'

    dfa_game = SymbolicPartitionedRegretDFAGame(boxes=boxes, locs=locs,
                                                ratio=ratio, init=init,
                                                goal=goal, formula=formula, 
                                                restricted_human_locs=human_locs,
                                                ltlf_flag=ltlf_flag, budget=budget,
                                                enable_reordering=enable_reordering)
    

    # print Game Info
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

    print("*****************Utility Value Map:*****************")
    for k, v in dfa_game.uVar_map.items():
        print(f"{k} : {v}")
    
    print("|xVars|: ", len(dfa_game.xVars))
    print("|rAct|: ", len(dfa_game.oVars))
    print("|eAct|: ", len(dfa_game.iVars))
    print("|kVars|: ", len(dfa_game.kVars))
    print("|uVars|: ", len(dfa_game.uVars))


    # print the number of explicit states
    sys_states, env_states = dfa_game.get_number_of_states(verbose=True)

    # print DFA Info
    print("*****************Printing Game Info*****************")
    for k, v in dfa_game.dfa_handle.qVar_map.items():
        print(f"{k} : {v}")


    # print DFA Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(dfa_game.latches) + len(dfa_game.qVars))
    print("Total num of prime latches: ", len(dfa_game.prime_latches) + len(dfa_game.prime_qVars))
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.prime_latches) + len(dfa_game.qVars) + len(dfa_game.prime_qVars))

    print(f"Total num of explicit states in DFA Game: {dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")
    print(f"Total num of explicit states in Graph of Utility DFA Game: {budget * dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_gou_trns)

    tic = time.time()
    strategy = dfa_game.regret_solver(verbose=False)
    toc = time.time()
    print(f"Time to synthesize Regret-Minimizing strategy: {toc - tic} seconds")

    if strategy is not None:
        dfa_game.gobr_roll_out_strategy(strategy=strategy, verbose=True)


def DFA_Game_Main():
    # setting things up
    boxes = 1
    locs = 2
    ratio = 1

    cooperative_game = True
    enable_reordering = False
    ltlf_flag = True

    # init = ['ready l2', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5', 'b4 l6', 'b5 l7']
    init = ['ready l2', 'b0 l2']
    # goal = [['b0 l1']]
    goal = []

    human_locs = range(1, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = [3]

    # formula = 'F(p01 & F(p02 & F(p01)))'
    # formula = 'F(p01 & p12)'
    formula = 'F(p01)'

    dfa_game = SymbolicPartitionedDFAGame(boxes=boxes, locs=locs,
                                          ratio=ratio, init=init,
                                          goal=goal, formula=formula, 
                                          restricted_human_locs=human_locs,
                                          ltlf_flag=ltlf_flag,
                                          enable_reordering=enable_reordering)
    
    # print Game Info
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
    sys_states, env_states = dfa_game.get_number_of_states(verbose=True)

    # print DFA Info
    print("*****************Printing Game Info*****************")
    for k, v in dfa_game.dfa_handle.qVar_map.items():
        print(f"{k} : {v}")


    # print DFA Game Info
    print("*****************Printing DFA Game Info*****************")
    print(f"Total num of latches: ", len(dfa_game.latches) + len(dfa_game.qVars))
    print("Total num of prime latches: ", len(dfa_game.prime_latches) + len(dfa_game.prime_qVars))
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.prime_latches) + len(dfa_game.qVars) + len(dfa_game.prime_qVars))

    print(f"Total num of explicit states in DFA Game: {dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_dfa_game_trns)
    # sys.exit(-1)

    # dfa_game.test_pre_image()

    tic = time.time()
    strategy = dfa_game.solve(verbose=False, cooperative_game=cooperative_game)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        dfa_game.roll_out_strategy(strategy=strategy, verbose=True)



def Game_Main():
    # setting things up
    boxes = 1
    locs = 2
    ratio = 1

    cooperative_game = True
    enable_reordering = False

    # init = ['ready l2', 'b0 l2', 'b1 l3']
    init = ['ready l2', 'b0 l2']
    goal = [['b0 l1']]

    human_locs = range(1, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = [3]

    
    game = FrankaWorldDynamicRatioTurnBasedElse(boxes=boxes, locs=locs,
                                                ratio=ratio, init=init,
                                                goal=goal, enable_reordering=enable_reordering,
                                                restricted_human_locs=human_locs)
    
    # game = FrankaWorldDynamicRatioTurnBased(boxes=boxes, locs=locs,
    #                                         ratio=ratio, init=init,
    #                                         goal=goal, enable_reordering=enable_reordering,
    #                                         restricted_human_locs=human_locs)

    # print Game Info
    print("*****************Printing Game Info*****************")
    print('****************xVars Map:****************')
    for k, v in game.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************rAction Map:****************')
    for k, v in game.rAction_map.items():
        print(f"{k} : {v}")

    print('****************eAction Map:****************')
    for k, v in game.eAction_map.items():
        print(f"{k} : {v}")

    print("*****************Ratio Map:*****************")
    for k, v in game.kVar_map.items():
        print(f"{k} : {v}")


    game.get_number_of_states(verbose=True)


    # print Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(game.latches))
    print("Total num of prime latches: ", len(game.prime_latches))
    print("Total boolean vars: ", len(game.latches) + len(game.prime_latches))

    # create the game's transition relation
    tic = time.time()
    game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    game.assert_one_s_prime_s_relation(dd_full_trans_rel=game.monolithic_valid_state_robot_actions_prime_state)
    sys.exit(-1)

    # dfa_game.test_pre_image()

    tic = time.time()
    strategy = game.solve(verbose=False, cooperative_game=cooperative_game)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        game.roll_out_strategy(strategy=strategy, verbose=True)



if __name__ == "__main__":
    # game synthesis main function call
    # Game_Main()
    
    # dfa game synthesis main function call
    # DFA_Game_Main()

    # Regret dfa game synthesis main function call
    Regret_DFA_Game_Main()
