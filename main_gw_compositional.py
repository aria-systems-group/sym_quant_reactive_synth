import time
import sys

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors import GridWorldDynamicDoorsGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors_dfa_game import GridWorldDynamicDoorsDFAGame
from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame


def game_main():
    # create grid dictionary
    rows = 2
    columns = 2
    goal = [(1, 1)]
    # init = []
    # grid = {'wall': [(0, 1), (1, 1)], 'goal': goal}
    door = [(1, 1)]  # list of doors and their locations
    grid = {'wall': [(0, 1), (2, 1)], 'goal': goal, 'door': door}
    # restricted_env_locs = [(1, 2)]
    players = {'sys': 2, 'env': 1}
    restricted_env_locs = []

    cooperative_game = False

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
    strategy, opt_sval = gridworld.solve(verbose=False, cooperative_game=False)
    # hybrid_strategy, hybrid_opt_sval = gridworld.hybrid_solve(verbose=False, cooperative_game=False)
    # bdd_strategy, bdd_opt_sval = gridworld.pure_bdd_solve(verbose=False, cooperative_game=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # if strategy.compare(bdd_strategy, 2):
    #     print("The strategies from both methods are the same!")
    # if opt_sval.compare(bdd_opt_sval, 2):
    #     print("The optimal state values from both methods are the same!")
    # strategy = bdd_strategy

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

    # testing things out with doors - relatively simple scenario
    # rows = 3
    # columns = 3
    # goal = [(2, 2)]
    # init = [(0, 0), (2, 0)]
    # # grid = {'wall': [(0, 1), (1, 1)], 'goal': goal}
    # door = [(1, 1)]  # list of doors and their locations
    # grid = {'wall': [(0, 1), (2, 1)], 'goal': goal, 'door': door}
    # formula = 'F(p & F(goal))'  # p is the proposition for photographing the other agent, c is the proposition for colliding
    # restricted_env_locs = [(1, 0)]


    # testing things out with doors - complex scenario
    rows = 5
    columns = 5
    goal = [(4, 4)]
    init = [(0, 0), (4, 3)]
    # grid = {'wall': [(0, 1), (1, 1)], 'goal': goal}
    door = [(1, 2)]  # list of doors and their locations
    restricted_env_locs = [*goal]
    # grid = {'wall': [(1, 1), (3, 1), (2, 1), (3, 2), (3, 3), (2, 3), (1, 3)], 'goal': goal, 'door': door, 's': [(2, 2)]}
    # grid = {'wall': [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3)], 'goal': goal, 's': [(2, 2)]}
    # grid = {'wall': [(2, 1), (3, 1), (3, 2), (3, 3), (2, 3)]}
    grid = {'wall': [(3, 1), (3, 2), (3, 3)], 'goal': goal, 's': [(2, 2)]}
    # restricted_env_locs = [(1, 0)]
    # formula = 'F(s & F(goal)) & G(!c)'  # p is the proposition for photographing the other agent, c is the proposition for colliding
    # formula = 'F(p & F(s & F(goal)))' # p is the proposition for photographing the other agent, c is the proposition for colliding
    formula = 'F(p & F(s & F(goal))) & G(!c)'
    # formula = 'F(s)'

    # testing things out with doors - relatively simple scenario
    rows = 5
    columns = 5
    goal = [(4, 4)]
    goal1 = [(4, 0)]
    # init = [(2, 0), (0, 4), (1, 1)]
    init = [(3, 0), (1, 1), (0, 3)]
    grid = {'wall': [(2, 0), (2, 1), (2, 2), (2, 3), (2, 4)], 'goal0': goal, 'goal1': goal1}
    # grid = {'goal0': goal, 'goal1': goal1}
    players = {'sys': 2, 'env': 1}
    door = [(1, 1)]  # list of doors and their locations
    # grid = {'wall': [(0, 1), (2, 1)], 'goal': goal, 'door': door}
    # formula = 'F(p & F(goal0)) & F(goal1) & G(!c)'  # p is the proposition for photographing the other agent, c is the proposition for colliding
    formula = 'F(p & F(goal0)) & F(goal1)'
    # restricted_env_locs = [(2, 1)]
    restricted_env_locs = [*goal, *goal1]

    cooperative_game = False

    # create a gridworld of size n x m
    if 'door' in grid.keys():
        gridworld =  GridWorldDynamicDoorsDFAGame(rows=rows, columns=columns,
                                                  init=init,
                                                  grid=grid, goal=goal,
                                                  camera=True,
                                                  players=players,
                                                  cooperative_game=cooperative_game,
                                                  restricted_env_locs=restricted_env_locs,
                                                  formula=formula, ltlf_flag=True)
    else:
        gridworld = GridWorldDynamicDFAGame(rows=rows, columns=columns,
                                            init=init,
                                            grid=grid, goal=goal,
                                            camera=True,
                                            players=players,
                                            cooperative_game=cooperative_game,
                                            restricted_env_locs=restricted_env_locs,
                                            formula=formula, ltlf_flag=True)

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
    print("*****************Printing Game Info*****************")
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
    strategy, opt_sval = gridworld.solve(verbose=False, cooperative_game=False)
    # hybrid_strategy, hybrid_opt_sval = gridworld.hybrid_solve(verbose=False, cooperative_game=False)
    # bdd_strategy, bdd_opt_sval = gridworld.pure_bdd_solve(verbose=False, cooperative_game=False)
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


if __name__ == "__main__":
    # game_main()

    dfa_game_main()