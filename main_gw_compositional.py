import time
import sys

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame


def game_main():
    # create grid dictionary
    rows = 100
    columns = 100
    goal = [(0, 2)]
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
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # debugging - print state and optimal value
    # gridworld.convert_cube_to_state_ADD(opt_sval, state_flag=True, action=False, verbose=True)
    
    # if strategy is not None:
    #     gridworld.roll_out_strategy(strategy=strategy, verbose=True)


def dfa_game_main():
    # small wrapper that convert goal to formula
    rows = 10
    columns = 10
    goal = [(9, 9)]
    # for idx, g in enumerate(goal):
    # {c + 1:0{len(str(rows))}b}
    # formula = f"F(p{goal[0][0]}{goal[0][1]})"

    # create a gridworld of size n x m
    gridworld = GridWorldDynamicDFAGame(rows=rows, columns=columns,
                                        init=[(0, 0), (1, 1)],
                                        goal=goal, formula='F(p0g)',
                                        ltlf_flag=True)

    print('****************Action Map:****************')
    for k, v in gridworld.action_map.items():
        print(f"{k} : {v}")

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
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    # debugging - print state and optimal value
    # gridworld.convert_cube_to_state_ADD(opt_sval, state_flag=True, action=False, verbose=True)
    
    # if strategy is not None:
    #     gridworld.roll_out_strategy(strategy=strategy, verbose=True)


if __name__ == "__main__":
    game_main()

    # dfa_game_main()