"""
 In this script we write the main for constructing the abstraction for FrankaDyanmic (Game) in Compositional Manner.  
"""
import sys
import time
import math

from typing import Union
from cudd import Cudd, ADD

# Imports with prime latches
from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame
from src.compositional_graphs.test_frankadynamic_ratio_else_tb import FrankaWorldDynamicRatioTurnBasedElse
from src.compositional_graphs.test_frankadynamic_ratio_tb import FrankaWorldDynamicRatioTurnBased

# Imports with no prime latches
from src.compositional_graphs.frankadynamic_ratio_tb_noprime import FrankaWorldDynamicRatioTurnBasedNoPrime
from src.compositional_graphs.frankadynamic_ratio_else_tb_noprime import FrankaWorldDynamicRatioTurnBasedElseNoPrime
from src.compositional_graphs.symbolic_partitioned_dfa_game_noprime import SymbolicPartitionedDFAGameNoPrime
from src.compositional_graphs.symbolic_partitioned_regret_dfa_game_noprime import SymbolicPartitionedRegretDFAGameNoPrime

# regret script imports
from src.compositional_graphs.symbolic_partitioned_regret_dfa_game import SymbolicPartitionedRegretDFAGame


def _test_opt_state_vals_are_equal(game: Union[FrankaWorldDynamicRatioTurnBased, FrankaWorldDynamicRatioTurnBasedElse, SymbolicPartitionedRegretDFAGame],
                                   reachable_opt_sVals: ADD,
                                   opt_sVals: ADD,
                                   debug: bool = False) -> bool:
    # check that the reachable states are the same as the original states
    diff_add = reachable_opt_sVals - opt_sVals
    if reachable_opt_sVals.compare(opt_sVals, 2):
        print("******************The reachable states are the same as the original states!******************")
        return True
    elif diff_add.findMin() == game.manager.addZero() and diff_add.findMax() == game.manager.plusInfinity():
        print("******************The reachable states are the same as the original states!******************")
        return True
    else:
        print("******************The reachable states are different from the original states!******************")
        if debug:
            game.convert_cube_to_state_ADD(diff_add, action=False, verbose=True)
        return False


def Regret_DFA_Game_Main():
    # setting things up
    # boxes = 2
    # locs = 3
    # ratio = 1
    # budget = 4

    boxes = 3
    locs = 8
    ratio = 1
    budget = 25

    cooperative_game = False
    enable_reordering = False
    ltlf_flag = True
    only_reachable_states = True

    # init = ['ready l6', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5']
    init = [f'ready l{locs + 1}', 'b0 l2', 'b1 l3', 'b2 l6']
    # init = ['ready l3', 'b0 l2', 'b1 l3']
    goal = []

    human_locs = range(5, locs + 1)
    # human_locs = range(3, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = [3]
    # box numbering starts with 0.
    human_boxes = [2]
    # human_boxes = range(boxes)

    # Simple set-up
    # boxes = 2
    # locs = 3
    # ratio = 1
    # budget = 10
    # init = ['ready l3', 'b0 l2', 'b1 l3']

    # Even more simple set-up
    # boxes = 1
    # locs = 2
    # ratio = 0
    # budget = 6
    # init = ['ready l2', 'b0 l2']

    # human_locs = range(2, locs + 1)
    # human_boxes = range(boxes)
    # human_boxes = [0]

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
                                                enable_reordering=enable_reordering,
                                                only_reachable_states=only_reachable_states)
    

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
    print("|brVars|: ", len(dfa_game.brVars))

    print("Total boolean vars in GoBR: ", len(dfa_game.gobr_game_latches) + len(dfa_game.gobr_game_prime_latches) + len(dfa_game.rVars))

    print(f"Time to create transition relation: {toc - tic} seconds")
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_gou_trns)
    # dfa_game.assert_one_s_prime_s_relation(dd_full_trans_rel=dfa_game.monolithic_valid_full_gobr_trns)
    # sys.exit(-1)
    # print("Variable ordering before calling the regret solver: ", dfa_game.manager.bddOrder())
    tic = time.time()
    strategy, rVals = dfa_game.regret_solver(verbose=False, optimized=False, only_reachable_state=False)
    toc = time.time()
    print(f"OLD: Time to synthesize Regret-Minimizing strategy: {toc - tic} seconds")
    
    # tic = time.time()
    # # to avoid caching related issues
    # dfa_game.gobr_care_set = dfa_game.compute_gobr_reachable_states(verbose=False, print_states=False)
    # print("********************Done Computing GoBR Reachable States********************")
    # strategy, reachable_rVals = dfa_game.regret_solver(verbose=False, optimized=False, only_reachable_state=only_reachable_states)
    # toc = time.time()
    # print(f"NEW: Time to synthesize Regret-Minimizing strategy with reachability states: {toc - tic} seconds")
    
    # compare the reget values computed using different methods
    # dfa_game.compare_regret_vals(reachable_dd=reachable_rVals, org_dd=rVals)
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

    if strategy is not None:
        dfa_game.gobr_roll_out_strategy(strategy=strategy, verbose=True)
    
    # dfa_game.debug_reachables_states()

def Regret_DFA_Game_Main_no_prime():
    # setting things up
    # boxes = 2
    # locs = 3
    # ratio = 1
    # budget = 4

    boxes = 3
    locs = 8
    ratio = 1
    budget = 25

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
    human_boxes = [2]
    # human_boxes = range(boxes)

    # Simple set-up
    # boxes = 2
    # locs = 3
    # ratio = 1
    # budget = 10
    # init = ['ready l3', 'b0 l2', 'b1 l3']

    # Even more simple set-up
    boxes = 1
    locs = 2
    ratio = 0
    budget = 6
    init = ['ready l2', 'b0 l2']

    human_locs = range(2, locs + 1)
    human_boxes = range(boxes)
    human_boxes = [0]

    # formula = 'F(p01 & F(p02 & F(p01)))'
    # formula = 'F(p01 & p12)'
    # formula = 'F(p01 & F(p02))'
    formula = 'F(p01)'

    dfa_game = SymbolicPartitionedRegretDFAGameNoPrime(boxes=boxes, locs=locs,
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
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.qVars) + len(dfa_game.rVars))

    print(f"Total num of explicit states in DFA Game: {dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")
    print(f"Total num of explicit states in Graph of Utility DFA Game: {budget * dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    


def DFA_Game_Main():
    # setting things up
    boxes = 2
    locs = 3
    ratio = 3

    cooperative_game = False
    enable_reordering = False
    only_reachable_states = False
    ltlf_flag = False

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

    # Simple set-up
    # boxes = 2
    # locs = 3
    # ratio = 1
    # init = ['ready l3', 'b0 l2', 'b1 l3']

    # Even more simple set-up
    boxes = 1
    locs = 2
    ratio = 1
    init = ['ready l3', 'b0 l2']
    # goal = [['b0 l1']]

    human_locs = range(1, locs + 1)
    human_boxes = range(boxes)
    # human_boxes = [1]
    formula = 'F(p01)'

    dfa_game = SymbolicPartitionedDFAGame(boxes=boxes, locs=locs,
                                          ratio=ratio, init=init,
                                          goal=goal, formula=formula, 
                                          restricted_human_locs=human_locs,
                                          restricted_human_boxes=human_boxes,
                                          ltlf_flag=ltlf_flag,
                                          enable_reordering=enable_reordering,
                                          only_reachable_states=only_reachable_states)
    
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
    strategy, opt_sVals = dfa_game.solve(verbose=False, cooperative_game=cooperative_game)
    # strategy = dfa_game.solve_optimized(verbose=False, cooperative_game=cooperative_game)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    dfa_game.only_reachable_states = True
    dfa_game.care_states = dfa_game.compute_reachable_states(monolithic_trans_dd=dfa_game.monolithic_valid_full_dfa_game_trns,
                                                             latches=dfa_game.latches + dfa_game.qVars,
                                                             prime_latches=dfa_game.prime_latches + dfa_game.prime_qVars,
                                                             act_vars=dfa_game.rVars, verbose=False, print_states=False)
    strategy, reach_opt_sVals = dfa_game.solve(verbose=False, cooperative_game=cooperative_game)
    
    # check that the reachable states are the same as the original states - sanity checking
    if strategy is not None:
        vals_same: bool = _test_opt_state_vals_are_equal(game=dfa_game, reachable_opt_sVals=reach_opt_sVals, opt_sVals=opt_sVals, debug=False)

        if not vals_same:
            sys.exit(-1)

    if strategy is not None:
        dfa_game.roll_out_strategy(strategy=strategy, verbose=True)



def DFA_Game_Main_no_prime():

    # Even more simple set-up
    boxes = 1
    locs = 20
    ratio = 1
    init = ['ready l2', 'b0 l2']
    goal = []

    human_locs = range(1, locs + 1)
    # human_boxes = range(boxes)
    human_boxes = [0]
    formula = 'F(p01 & F(p02))'

    cooperative_game = True
    enable_reordering = False
    ltlf_flag = True

    dfa_game = SymbolicPartitionedDFAGameNoPrime(boxes=boxes, locs=locs,
                                                 ratio=ratio, init=init,
                                                 goal=goal, formula=formula,
                                                 restricted_human_locs=human_locs,
                                                 restricted_human_boxes=human_boxes,
                                                 ltlf_flag= ltlf_flag,
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
    print("Total boolean vars: ", len(dfa_game.latches) + len(dfa_game.qVars) + len(dfa_game.rVars))
    print("********************************************************")
    print(f"Total num of explicit states in DFA Game: {dfa_game.dfa_handle.num_of_states * (env_states + sys_states):,}")
    print("********************************************************")

    # create the game's transition relation
    tic = time.time()
    dfa_game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # dfa_game.test_pre_image()
    # return

    tic = time.time()
    strategy, opt_sVals = dfa_game.solve(verbose=False, cooperative_game=cooperative_game)
    # strategy, old_opt_sVals = dfa_game.old_solve(verbose=False, cooperative_game=cooperative_game)
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
    only_reachable_states = False

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
    # human_boxes = range(boxes)
    human_boxes = [0]

    
    # game = FrankaWorldDynamicRatioTurnBasedElse(boxes=boxes, locs=locs,
    #                                             ratio=ratio, init=init,
    #                                             goal=goal, enable_reordering=enable_reordering,
    #                                             restricted_human_locs=human_locs,
    #                                             restricted_human_boxes=human_boxes,
    #                                             only_reachable_states=only_reachable_states)
    
    game = FrankaWorldDynamicRatioTurnBased(boxes=boxes, locs=locs,
                                            ratio=ratio, init=init,
                                            goal=goal, enable_reordering=enable_reordering,
                                            restricted_human_locs=human_locs, 
                                            restricted_human_boxes=human_boxes,
                                            only_reachable_states=only_reachable_states)

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
    print("******************Printing DFA Game Info*****************")
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
    # strategy, opt_sVals = game.solve(verbose=False, cooperative_game=cooperative_game)
    strategy, opt_sVals = game.solve_optimized(verbose=False, cooperative_game=cooperative_game)

    # NOTE: I am doing this purely to test reachable state computation and non-reachable state computation in one go.
    # To just do eithe of them, jsut se the reachable states flag in the game initilization above accordingly and comment this part.
    # now override - compute reachable states variable and manually set it to True 
    game.only_reachable_states = True
    game.care_states = game.compute_reachable_states(monolithic_trans_dd=game.monolithic_state_action_prime_state,
                                                     latches=game.latches, prime_latches=game.prime_latches,
                                                     act_vars=game.rVars, verbose=False, print_states=False)
    strategy, reach_opt_sVals = game.solve_optimized(verbose=False, cooperative_game=cooperative_game)


    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # check that the reachable states are the same as the original states - sanity checking
    if strategy is not None:
        vals_same: bool = _test_opt_state_vals_are_equal(game=game, reachable_opt_sVals=reach_opt_sVals, opt_sVals=opt_sVals, debug=False)

        if not vals_same:
            sys.exit(-1)

    if strategy is not None:
        game.roll_out_strategy(strategy=strategy, verbose=True)



def Game_Main_no_prime():
    # Even more simple set-up
    boxes = 3
    locs = 15
    ratio = 1
    init = ['ready l2', 'b0 l2', 'b1 l6', 'b2 l4']
    goal = [['b0 l1']]
    human_locs = range(5, locs + 1)
    human_boxes = range(boxes)
    human_boxes = [1]

    # Even more simple set-up
    # boxes = 1
    # locs = 2
    # ratio = 1
    # init = ['ready l2', 'b0 l2']
    # goal = [['b0 l1']]

    # human_locs = range(1, locs + 1)
    # # human_boxes = range(boxes)
    # human_boxes = [0]

    cooperative_game = False
    enable_reordering = True

    # game = FrankaWorldDynamicRatioTurnBasedNoPrime(boxes=boxes, locs=locs,
    #                                                ratio=ratio, init=init,
    #                                                goal=goal, enable_reordering=enable_reordering,
    #                                                restricted_human_locs=human_locs, 
    #                                                restricted_human_boxes=human_boxes)
    
    game = FrankaWorldDynamicRatioTurnBasedElseNoPrime(boxes=boxes, locs=locs,
                                                       ratio=ratio, init=init,
                                                       goal=goal, enable_reordering=enable_reordering,
                                                       restricted_human_locs=human_locs, 
                                                       restricted_human_boxes=human_boxes)

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
    print("******************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(game.latches))
    print("Total boolean vars: ", len(game.latches) + len(game.rVars))

    tic = time.time()
    game.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    # game.assert_one_s_prime_s_relation(dd_full_trans_rel=game.monolithic_valid_state_robot_actions_prime_state)

    # game.test_pre_image_restricted_human_moves()
    # sys.exit(-1)

    # tic = time.time()
    # strategy, opt_sVals = game.solve(verbose=False, cooperative_game=cooperative_game)
    # toc = time.time()
    # print(f"Time to synthesize strategy: {toc - tic} seconds")

    tic = time.time()
    strategy, opt_sVals = game.old_solve(verbose=False, cooperative_game=cooperative_game)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        game.roll_out_strategy(strategy=strategy, verbose=True)




if __name__ == "__main__":
    # game synthesis main function call
    # Game_Main()
    # Game_Main_no_prime()
    
    # dfa game synthesis main function call
    # DFA_Game_Main()
    # DFA_Game_Main_no_prime()

    # Regret dfa game synthesis main function call
    # Regret_DFA_Game_Main()
    Regret_DFA_Game_Main_no_prime()
