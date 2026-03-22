import time
import sys

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame


def game_main():
    # create grid dictionary
    rows = 2
    columns = 2
    goal = [(1, 1)]
    # init = []
    # grid = {'wall': [(0, 1), (1, 1)], 'goal': goal}
    grid = {'wall': [(0, 1)], 'goal': goal}

    # create a gridworld of size n x m
    gridworld = GridWorldDynamicGame(rows=rows, columns=columns, init=[(0, 0), (1, 0)], goal=goal, grid=grid)

    print('****************Sys Action Map:****************')
    for k, v in gridworld.sys_action_map.items():
        print(f"{k} : {v}")

    print('****************Env Action Map:****************')
    for k, v in gridworld.env_action_map.items():
        print(f"{k} : {v}")
    
    # print the number of explicit states
    sys_states, env_states = gridworld.get_number_of_states(verbose=True)

     # print DFA Game Info
    print("*****************Printing DFA Game Info*****************")
    print("Total num of latches: ", len(gridworld.latches))
    print("Total boolean vars: ", len(gridworld.latches) + len(gridworld.rVars))

    tic = time.time()
    gridworld.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # test preimage computation
    # gridworld.test_preimage()
    # sys.exit(-1)

    # solve
    tic = time.time()
    # strategy, opt_sval = gridworld.solve(verbose=False, cooperative_game=False)
    # hybrid_strategy, hybrid_opt_sval = gridworld.hybrid_solve(verbose=False, cooperative_game=False)
    bdd_strategy, bdd_opt_sval = gridworld.pure_bdd_solve(verbose=False, cooperative_game=False)
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
    rows = 10
    columns = 10
    goal = [(2, 2)]
    goal1 = [(2, 0)]
    grid = {'wall': [(2, 1)], 'goal': goal, 'goal1': goal1}
    # grid = {'wall': [(0, 1)], 'goal': goal}

    # create a gridworld of size n x m
    gridworld = GridWorldDynamicDFAGame(rows=rows, columns=columns,
                                        init=[(0, 0), (2, 0)],
                                        grid=grid, goal=goal,
                                        camera=True,
                                        formula='F(p) & G(!c)', ltlf_flag=True)

    print('****************Sys Action Map:****************')
    for k, v in gridworld.sys_action_map.items():
        print(f"{k} : {v}")

    print('****************Env Action Map:****************')
    for k, v in gridworld.env_action_map.items():
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
    print("Total boolean vars: ", len(gridworld.latches) + len(gridworld.qVars) + len(gridworld.rVars))
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
    # strategy, opt_sval = gridworld.solve(verbose=False, cooperative_game=False)
    # hybrid_strategy, hybrid_opt_sval = gridworld.hybrid_solve(verbose=False, cooperative_game=False)
    bdd_strategy, bdd_opt_sval = gridworld.pure_bdd_solve(verbose=False, cooperative_game=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # if bdd_strategy.compare(strategy, 2):
    #     print("The strategies from both methods are the same!")
    # if bdd_opt_sval.compare(opt_sval, 2):
    #     print("The optimal state values from both methods are the same!")
    strategy = bdd_strategy
    
    # debugging - print state and optimal value
    # gridworld.convert_cube_to_state_ADD(opt_sval, state_flag=True, action=False, verbose=True)
    
    if strategy is not None:
        gridworld.roll_out_strategy(strategy=strategy, verbose=True)


if __name__ == "__main__":
    # game_main()

    dfa_game_main()