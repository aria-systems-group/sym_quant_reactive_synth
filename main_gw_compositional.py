import time
import sys

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors import GridWorldDynamicDoorsGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors_dfa_game import GridWorldDynamicDoorsDFAGame
from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame

from src.compositional_graphs.gridworld.gridworld_dynamic_regret import GridWorldDynamicRegretGame


def game_main():
    # create grid dictionary
    rows = 3
    columns = 3
    goal = [(2, 2)]

    # init = [(0, 0), (2, 0)]
    init = [(0, 0), (2, 0), (0, 2), (2, 2), (0, 0)]
    # init = [(0, 0), (2, 0), (0, 2)]
    # grid = {}
    # grid = {'wall': [(0, 1)], 'goal': goal}
    grid = {'wall': [(0, 1), (2, 1)], 'goal': goal}
    door = [(1, 1)]  # list of doors and their locations
    grid = {'wall': [(0, 1), (2, 1)], 'goal': goal, 'door': door}
    # restricted_env_locs = [(1, 2)]
    players = {'sys': 1, 'env': 4}
    restricted_env_locs = []

    # simple example - 2 x 2 grid, one goal, two agents
    rows = 2
    columns = 2
    goal = [(1, 1)]
    init = [(0, 0), (1, 0)]
    grid = {'wall': [(0, 1)], 'goal': goal}
    players = {'sys': 1, 'env': 1}
    restricted_env_locs = []

    cooperative_game = True

    # testing things out
    # rows = 5
    # columns = 5
    # goal = [(2, 2)]
    # # init = []
    # # grid = {'wall': [(0, 1), (1, 1)], 'goal': goal}
    # door = [(1, 2)]  # list of doors and their locations
    # grid = {'wall': [(1, 1), (3, 1), (2, 1), (3, 2), (3, 3), (2, 3), (1, 3)], 'goal': goal, 'door': door, 's': [(2, 2)]}
    # restricted_env_locs = [(2, 2)]

    # create a gridworld of size n x m
    if 'door' in grid.keys():
        gridworld = GridWorldDynamicDoorsGame(rows=rows, columns=columns,
                                              init=init, goal=goal,
                                              grid=grid, players=players,
                                              cooperative_game=cooperative_game,
                                              restricted_env_locs=restricted_env_locs)
    else:
        gridworld = GridWorldDynamicGame(rows=rows, columns=columns,
                                         init=init, goal=goal,
                                         grid=grid, players=players,
                                         cooperative_game=cooperative_game,
                                         restricted_env_locs=restricted_env_locs)

    print('****************Sys Action Map:****************')
    for k, v in gridworld.sys_action_map.items():
        print(f"{k} : {v}")

    print('****************Env Action Map:****************')
    for k, v in gridworld.env_action_map.items():
        print(f"{k} : {v}")
    
    if 'door' in grid.keys():
        print("*****************Door Map:*****************")
        for didx in range(len(grid['door'])):
            print(f'Door{didx} Vars') 
            for k, v in gridworld.dVar_map[didx].items():
                print(f"{k} : {v}")
    
    # print the number of explicit states
    sys_states, env_states = gridworld.get_number_of_states(verbose=True)

     # print DFA Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(gridworld.latches))
    print("Total num of prime latches: ", len(gridworld.prime_latches))
    print("Total boolean vars: ", len(gridworld.latches) + len(gridworld.prime_latches) + len(gridworld.rVars))

    tic = time.time()
    gridworld.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # test preimage computation
    # gridworld.test_preimage()
    # sys.exit(-1)

    # solve
    tic = time.time()
    # strategy, opt_sval = gridworld.solve(verbose=False)
    # hybrid_strategy, hybrid_opt_sval = gridworld.hybrid_solve(verbose=False)
    bdd_strategy, bdd_opt_sval = gridworld.pure_bdd_solve(verbose=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # if strategy.compare(bdd_strategy, 2):
    #     print("The strategies from both methods are the same!")
    # if opt_sval.compare(bdd_opt_sval, 2):
    #     print("The optimal state values from both methods are the same!")
    strategy = bdd_strategy

    # debugging - print state and optimal value
    # gridworld.convert_cube_to_state_ADD(opt_sval, state_flag=True, action=False, verbose=True)
    
    if strategy is not None:
        gridworld.roll_out_strategy(strategy=strategy, verbose=True)


def dfa_game_main():
    # small wrapper that convert goal to formula
    rows = 25
    columns = 25
    goal = [(1, 1)]
    goal1 = [(2, 0)]
    # grid = {'wall': [(2, 1)], 'goal': goal, 'goal1': goal1}
    grid = {'wall': [(0, 1)], 'goal': goal}
    restricted_env_locs = [(1, 0)]

    # testing things out with doors - complex scenario
    rows = 20
    columns = 20
    goal = [(0, 0)]
    init = [(0, 0), (4, 3)]
    # grid = {'wall': [(0, 1), (1, 1)], 'goal': goal}
    door = [(1, 2)]  # list of doors and their locations
    restricted_env_locs = [*goal]
    # grid = {'wall': [(1, 1), (3, 1), (2, 1), (3, 2), (3, 3), (2, 3), (1, 3)], 'goal': goal, 'door': door, 's': [(2, 2)]}
    # grid = {'wall': [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3)], 'goal': goal, 's': [(2, 2)]}
    # grid = {'wall': [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3)]}
    # grid = {'wall': [(3, 1), (3, 2), (3, 3)], 'goal': goal, 's': [(2, 2)]}
    # restricted_env_locs = [(1, 0)]
    # formula = 'F(s & F(goal)) & G(!c)'  # p is the proposition for photographing the other agent, c is the proposition for colliding
    # formula = 'F(p & F(s & F(goal)))'
    # formula = 'F(p & F(s & F(goal))) & G(!c)'
    grid = {'g0': goal, 'g1': [(rows - 1, 0)], 'g2': [(0, columns - 1)], 'g3': [(rows - 1, columns - 1)]}
    formula = 'F(g0) & F(g1) & F(g2) & F(g3)'  # visit all four corners

    # testing things out with doors - relatively simple scenario
    rows = 5
    columns = 5
    goal = [(4, 4)]
    goal1 = [(4, 0)]
    # init = [(2, 0), (0, 4), (1, 1)]
    init = [(3, 0), (1, 1), (0, 3)]
    # grid = {'wall': [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], 'goal0': goal, 'goal1': goal1}
    # grid = {'goal0': goal, 'goal1': goal1}
    players = {'sys': 2, 'env': 1}
    door = [(2, 2)]  # list of doors and their locations
    # grid = {'wall': [(0, 1), (2, 1)], 'goal': goal, 'door': door}
    grid = {'wall': [(2, 0), (2, 1), (2, 3), (2, 4)], 'goal0': goal,  'goal1': goal1, 'door': door}
    # formula = 'F(p & F(goal0)) & F(goal1) & G(!c)'  # p is the proposition for photographing the other agent, c is the proposition for colliding
    # formula = 'F(p & F(goal0)) & F(goal1) & G!c'
    formula = 'F(p & F(goal0)) & F(goal1) & G!c'
    # restricted_env_locs = [(2, 1)]
    restricted_env_locs = [*goal, *goal1]

    # simple 2x2 gridworld
    rows = 3
    columns = 3
    goal = [(2, 2)]
    init = [(0, 0), (0, 2), (2, 0)]
    # grid = {'wall': [(0, 1), (2, 1)], 'goal': goal}
    grid = {'wall': [], 'goal': goal}
    players = {'sys': 2, 'env': 1}
    restricted_env_locs = []
    # formula = 'F(goal)'
    formula = 'F(goal & X(!goal))'

    ltlf_flag = True
    cooperative_game = False

    # create a gridworld of size n x m
    if 'door' in grid.keys():
        gridworld =  GridWorldDynamicDoorsDFAGame(rows=rows, columns=columns,
                                                  init=init,
                                                  grid=grid, goal=goal,
                                                  camera=False,
                                                  players=players,
                                                  cooperative_game=cooperative_game,
                                                  restricted_env_locs=restricted_env_locs,
                                                  formula=formula, ltlf_flag=ltlf_flag)
    else:
        gridworld = GridWorldDynamicDFAGame(rows=rows, columns=columns,
                                            init=init,
                                            grid=grid, goal=goal,
                                            camera=False,
                                            players=players,
                                            cooperative_game=cooperative_game,
                                            restricted_env_locs=restricted_env_locs,
                                            formula=formula, ltlf_flag=ltlf_flag)

    print('****************Sys Action Map:****************')
    for k, v in gridworld.sys_action_map.items():
        print(f"{k} : {v}")

    print('****************Env Action Map:****************')
    for k, v in gridworld.env_action_map.items():
        print(f"{k} : {v}")
    
    if 'door' in grid.keys():
        print("*****************Door Map:*****************")
        for didx in range(len(grid['door'])):
            print(f'Door{didx} Vars') 
            for k, v in gridworld.dVar_map[didx].items():
                print(f"{k} : {v}")

    print("*****************Label Map:*****************")
    for k, v in gridworld.lVar_map.items():
        print(f"{k} : {v}")
    
    # print the number of explicit states
    sys_states, env_states = gridworld.get_number_of_states(verbose=True)
    
    # print DFA Info
    print("*****************Printing DFA Info*****************")
    for k, v in gridworld.dfa_handle.qVar_map.items():
        print(f"{k} : {v}")

    # print DFA Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(gridworld.latches) + len(gridworld.qVars))
    print("Total num of prime latches: ", len(gridworld.prime_latches) + + len(gridworld.prime_qVars))
    print("Total boolean vars: ", len(gridworld.latches) + len(gridworld.qVars) + len(gridworld.prime_latches) + + len(gridworld.prime_qVars) + len(gridworld.rVars))
    print("********************************************************")
    print(f"Total num of explicit states in DFA Game: {gridworld.dfa_handle.num_of_states * (env_states + sys_states):,}")
    print("********************************************************")

    tic = time.time()
    gridworld.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # test preimage computation
    # gridworld.test_preimage()
    # gridworld.test_preimage_2()
    # sys.exit(-1)

    # solve
    tic = time.time()
    strategy, opt_sval = gridworld.solve(verbose=False)
    # hybrid_strategy, hybrid_opt_sval = gridworld.hybrid_solve(verbose=False)
    # bdd_strategy, bdd_opt_sval = gridworld.pure_bdd_solve(verbose=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # if bdd_strategy.compare(strategy, 2):
    #     print("The strategies from both methods are the same!")
    # if bdd_opt_sval.compare(opt_sval, 2):
    #     print("The optimal state values from both methods are the same!")
    # strategy = bdd_strategy
    
    # debugging - print state and optimal value
    # gridworld.convert_cube_to_state_ADD(opt_sval, state_flag=True, action=False, verbose=True)
    
    if strategy is not None:
        gridworld.roll_out_strategy(strategy=strategy, verbose=True)



def dfa_regret_main():

    rows = 3
    columns = 3
    goal = [(2, 2)]

    # init = [(0, 0), (2, 0)]
    # init = [(0, 0), (2, 0), (0, 2), (2, 2), (0, 0)]
    init = [(0, 0), (0, 2), (2, 0)]
    # grid = {}
    # grid = {'wall': [(0, 1)], 'goal': goal}
    grid = {'wall': [(0, 1), (2, 1)], 'goal': goal}
    # door = [(1, 1)]  # list of doors and their locations
    # grid = {'wall': [(0, 1), (2, 1)], 'goal': goal, 'door': door}
    # restricted_env_locs = [(1, 2)]
    players = {'sys': 2, 'env': 1}
    restricted_env_locs = []
    formula = 'F(goal & X(!goal))'
    # formula = 'F(goal)'
    budget = 5

    cooperative_game = False
    ltlf_flag = True


    gridworld = GridWorldDynamicRegretGame(rows=rows, columns=columns,
                                           init=init,
                                           grid=grid, goal=goal,
                                           camera=False,
                                           budget=budget, 
                                           formula=formula,
                                           players=players,
                                           cooperative_game=cooperative_game,
                                           restricted_env_locs=restricted_env_locs,
                                           ltlf_flag=ltlf_flag)


    print('****************Sys Action Map:****************')
    for k, v in gridworld.sys_action_map.items():
        print(f"{k} : {v}")

    print('****************Env Action Map:****************')
    for k, v in gridworld.env_action_map.items():
        print(f"{k} : {v}")
    
    if 'door' in grid.keys():
        print("*****************Door Map:*****************")
        for didx in range(len(grid['door'])):
            print(f'Door{didx} Vars') 
            for k, v in gridworld.dVar_map[didx].items():
                print(f"{k} : {v}")

    print("*****************Label Map:*****************")
    for k, v in gridworld.lVar_map.items():
        print(f"{k} : {v}")
    
    # print the number of explicit states
    # sys_states, env_states = gridworld.get_number_of_states(verbose=True)
    
    # print DFA Info
    print("*****************Printing DFA Info*****************")
    for k, v in gridworld.dfa_handle.qVar_map.items():
        print(f"{k} : {v}")
    

    # print unrolled DFA Game Info
    print("*****************Printing GoU Info*****************")
    print("Total num of latches: ", len(gridworld.latches) + len(gridworld.qVars))
    print("Total num of prime latches: ", len(gridworld.prime_latches) + len(gridworld.prime_qVars))
    print("Total boolean vars: ", len(gridworld.latches) + len(gridworld.prime_latches) + len(gridworld.qVars) + len(gridworld.prime_qVars) + len(gridworld.rVars))

    tic = time.time()
    gridworld.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")
    print("*****************BR Info*****************")
    for k, v in gridworld.brVar_map.items():
        print(f"{k} : {v}")

    print("Total boolean vars in GoBR: ", len(gridworld.gobr_game_latches) + len(gridworld.gobr_game_prime_latches) + len(gridworld.rVars))

    # gridworld.test_pre_image()
    # return

    tic = time.time()
    strategy, rVals = gridworld.regret_solver(verbose=False)
    # hybrid_strategy, hybrid_rVals = dfa_game.hybrid_regret_solver(verbose=False)
    # bdd_strategy, bdd_rVals = dfa_game.pure_bdd_regret_solver(verbose=False)
    toc = time.time()
    print(f"OLD: Time to synthesize Regret-Minimizing strategy: {toc - tic} seconds")

    if strategy is not None:
        gridworld.gobr_roll_out_strategy(strategy=strategy, verbose=True)



if __name__ == "__main__":
    # game_main()

    dfa_game_main()

    # dfa_regret_main()
