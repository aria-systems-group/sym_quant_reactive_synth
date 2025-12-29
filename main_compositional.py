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
    # boxes = 2
    # locs = 3
    # ratio = 1
    # budget = 4

    boxes = 3
    locs = 7
    ratio = 1
    budget = 10

    cooperative_game = False
    enable_reordering = False
    ltlf_flag = True

    # init = ['ready l6', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5']
    init = [f'ready l{locs + 1}', 'b0 l2', 'b1 l3', 'b2 l6']
    # init = ['ready l3', 'b0 l2', 'b1 l3']
    goal = []

    human_locs = range(5, locs + 1)
    # human_locs = range(3, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = [3]
    # box numbering starts with 0.
    # human_boxes = [1]
    human_boxes = range(boxes)

    # formula = 'F(p01 & F(p02 & F(p01)))'
    # formula = 'F(p01 & p12)'
    # formula = 'F(p01 & F(p02))'
    formula = 'F(p01)'

    dfa_game = SymbolicPartitionedRegretDFAGame(boxes=boxes, locs=locs,
                                                ratio=ratio, init=init,
                                                goal=goal, formula=formula, 
                                                restricted_human_locs=human_locs,
                                                restricted_human_boxes=human_boxes,
                                                ltlf_flag=ltlf_flag, budget=budget,
                                                enable_reordering=enable_reordering)
    

    # print Game Info
    print("*****************Printing Game Info*****************")
    print('****************xVars Map:****************')
    for k, v in dfa_game.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************Action Map:****************')
    for k, v in dfa_game.action_map.items():
        print(f"{k} : {v}")

    print("*****************Ratio Map:*****************")
    for k, v in dfa_game.kVar_map.items():
        print(f"{k} : {v}")

    print("*****************Utility Value Map:*****************")
    for k, v in dfa_game.uVar_map.items():
        print(f"{k} : {v}")
    
    # print the number of explicit states
    sys_states, env_states = dfa_game.get_number_of_states(verbose=True)

    # print DFA Info
    print("*****************DFA Info*****************")
    for k, v in dfa_game.dfa_handle.qVar_map.items():
        print(f"{k} : {v}")


    # print DFA Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(dfa_game.latches) + len(dfa_game.qVars))
    print("Total num of prime latches: ", len(dfa_game.prime_latches) + len(dfa_game.prime_qVars))
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.prime_latches) + len(dfa_game.qVars) + len(dfa_game.prime_qVars) + len(dfa_game.rVars))

    print(f"Total num of explicit states in DFA Game: {dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")
    print(f"Total num of explicit states in Graph of Utility DFA Game: {budget * dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    print("*****************BR Info*****************")
    for k, v in dfa_game.brVar_map.items():
        print(f"{k} : {v}")
    
    print(f"Total num of explicit states in Graph of BR DFA Game: {len(dfa_game.brVals) * budget * dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")

    print("|xVars|: ", len(dfa_game.xVars))
    print("|rAct|: ", len(dfa_game.rVars))
    print("|kVars|: ", len(dfa_game.kVars))
    print("|uVars|: ", len(dfa_game.uVars))
    print("|brVars|: ", len(dfa_game.brVars) )
    print(f"Time to create transition relation: {toc - tic} seconds")
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_gou_trns)
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_gobr_trns)
    # sys.exit(-1)
    # print("Variable ordering before calling the regret solver: ", dfa_game.manager.bddOrder())
    tic = time.time()
    strategy, reachable_rVals = dfa_game.regret_solver(verbose=False, optimized=False, only_reachable_state=True)
    toc = time.time()
    print(f"OLD: Time to synthesize Regret-Minimizing strategy: {toc - tic} seconds")

    tic = time.time()
    strategy, rVals = dfa_game.regret_solver(verbose=False, optimized=False, only_reachable_state=False)
    toc = time.time()
    
    # compare the reget values computed using different methods
    dfa_game.compare_regre_vals(reachable_dd=reachable_rVals, org_dd=rVals)
    # tic = time.time()
    # strategy = dfa_game.regret_solver(verbose=False, optimized=False, only_reachable_state=True)
    # toc = time.time()
    # print(f"With Reachable Sates: Time to synthesize Regret-Minimizing strategy: {toc - tic} seconds")
    # dfa_game.TVI_regret_solver(verbose=False)
    # tic = time.time()
    # dfa_game.TVI_utility_regret_solver()
    # dfa_game.TVI_br_regret_solver()
    # dfa_game.gou_solve(verbose=False, optimized=False, test=True)
    # toc = time.time()
    # assert dfa_game.rVals == dfa_game.test_rVals, "Regret value maps do not match between regret_solver and TVI_regret_solver!"
    # print(f"NEW: Time to synthesize Regret-Minimizing strategy: {toc - tic} seconds")
    # print("Variable ordering After calling the regret solver: ", dfa_game.manager.bddOrder())

    # if strategy is not None:
    #     dfa_game.gobr_roll_out_strategy(strategy=strategy, verbose=True)
    
    # dfa_game.debug_reachables_states()


def DFA_Game_Main():
    # setting things up
    boxes = 2
    locs = 3
    ratio = 3

    cooperative_game = True
    enable_reordering = False
    ltlf_flag = True

    # init = ['ready l2', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5', 'b4 l6', 'b5 l7']
    init = ['ready l4', 'b0 l2', 'b1 l3']#, 'b2 l9']
    # goal = [['b0 l1']]
    goal = []

    human_locs = range(1, locs + 1)
    human_boxes = [0]
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = [3]

    # formula = 'F(p01 & F(p02 & F(p01)))'
    formula = 'F(p01 & F(p02))'
    # formula = 'F(p01)'

    dfa_game = SymbolicPartitionedDFAGame(boxes=boxes, locs=locs,
                                          ratio=ratio, init=init,
                                          goal=goal, formula=formula, 
                                          restricted_human_locs=human_locs,
                                          restricted_human_boxes=human_boxes,
                                          ltlf_flag=ltlf_flag,
                                          enable_reordering=enable_reordering)
    
    # print Game Info
    print("*****************Printing Game Info*****************")
    print('****************xVars Map:****************')
    for k, v in dfa_game.xVar_map.items():
        print(f"{k} : {v}")

    print('****************Action Map:****************')
    for k, v in dfa_game.action_map.items():
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
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.prime_latches) + len(dfa_game.qVars) + len(dfa_game.prime_qVars) + len(dfa_game.rVars))
    print("********************************************************")
    print(f"Total num of explicit states in DFA Game: {dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")
    print("********************************************************")

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_dfa_game_trns)
    # sys.exit(-1)

    # dfa_game.test_pre_image()
    # return

    tic = time.time()
    # strategy = dfa_game.solve(verbose=False, cooperative_game=cooperative_game)
    strategy = dfa_game.solve_optimized(verbose=False, cooperative_game=cooperative_game)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        dfa_game.roll_out_strategy(strategy=strategy, verbose=True)



def Game_Main():
    # setting things up
    boxes = 4
    locs = 8
    ratio = 1

    cooperative_game = False
    enable_reordering = True

    init = [f'ready l{locs + 1}', 'b0 l2', 'b1 l3', 'b2 l6', 'b3 l7']#, 'b4 l4', 'b5 l8']
    # init = ['ready l2', 'b0 l2', 'b1 l3', 'b2 l6']
    # init = ['ready l1', 'b0 l2']
    goal = [['b0 l1']]

    # human_locs = range(2, locs + 1)
    human_locs = range(5, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10]
    # human_locs = [3]
    # human_boxes = range(boxes)
    human_boxes = [2, 3]#, 5]

    # Simple set-up
    boxes = 2
    locs = 3
    ratio = 1
    init = ['ready l3', 'b0 l2', 'b1 l3']
    goal = [['b0 l1']]

    # Even more simple set-up
    # boxes = 1
    # locs = 2
    # ratio = 1
    # init = ['ready l2', 'b0 l2']
    # goal = [['b0 l1']]

    human_locs = range(2, locs + 1)
    human_boxes = range(boxes)
    human_boxes = [1]

    
    game = FrankaWorldDynamicRatioTurnBasedElse(boxes=boxes, locs=locs,
                                                ratio=ratio, init=init,
                                                goal=goal, enable_reordering=enable_reordering,
                                                restricted_human_locs=human_locs,
                                                restricted_human_boxes=human_boxes)
    
    # game = FrankaWorldDynamicRatioTurnBased(boxes=boxes, locs=locs,
    #                                         ratio=ratio, init=init,
    #                                         goal=goal, enable_reordering=enable_reordering,
    #                                         restricted_human_locs=human_locs, 
    #                                         restricted_human_boxes=human_boxes)

    # print Game Info
    print("*****************Printing Game Info*****************")
    print('****************xVars Map:****************')
    for k, v in game.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************Action Map:****************')
    for k, v in game.action_map.items():
        print(f"{k} : {v}")


    print("*****************Ratio Map:*****************")
    for k, v in game.kVar_map.items():
        print(f"{k} : {v}")

    # print Game Info - # number of explicit states
    game.get_number_of_states(verbose=True)

    # print Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(game.latches))
    print("Total num of prime latches: ", len(game.prime_latches))
    print("Total boolean vars: ", len(game.latches) + len(game.prime_latches) + len(game.rVars))

    # create the game's transition relation
    tic = time.time()
    game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    # game.assert_one_s_prime_s_relation(dd_full_trans_rel=game.monolithic_valid_state_robot_actions_prime_state)

    # game.test_pre_image_restricted_human_moves()
    # sys.exit(-1)

    tic = time.time()
    strategy = game.solve(verbose=False, cooperative_game=cooperative_game, only_reachable_state=True)
    # strategy = game.solve_optimized(verbose=False, cooperative_game=cooperative_game)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        game.roll_out_strategy(strategy=strategy, verbose=True)



if __name__ == "__main__":
    # game synthesis main function call
    Game_Main()
    
    # dfa game synthesis main function call
    # DFA_Game_Main()

    # Regret dfa game synthesis main function call
    # Regret_DFA_Game_Main()
