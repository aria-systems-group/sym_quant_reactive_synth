from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamic


if __name__ == "__main__":
    # create a gridworld of size 2 x 2
    gridworld = GridWorldDynamic(rows=2, columns=2, init=[(0, 0), (1, 1)], goal=[(1, 1)])

    gridworld.create_transition_relation()