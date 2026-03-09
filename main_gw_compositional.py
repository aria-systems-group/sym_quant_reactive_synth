import sys

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamic


if __name__ == "__main__":
    # create a gridworld of size n x m
    gridworld = GridWorldDynamic(rows=10, columns=10, init=[(0, 0), (1, 1)], goal=[(9, 9)])

    print('****************Action Map:****************')
    for k, v in gridworld.action_map.items():
        print(f"{k} : {v}")

    gridworld.create_transition_relation()

    # test preimage computation
    # gridworld.test_preimage()
    # sys.exit(-1)

    # solve 
    strategy, opt_sval = gridworld.solve(verbose=False, cooperative_game=False)

    # debugging - print state and optimal value
    # gridworld.convert_cube_to_state_ADD(opt_sval, state_flag=True, action=False, verbose=True)
    
    if strategy is not None:
        gridworld.roll_out_strategy(strategy=strategy, verbose=True)