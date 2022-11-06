from cudd import Cudd

from src.symbolic_graphs.graph_search_scripts import SimpleGridWorld, FrankaWorld
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld

from utls import *
from config import *


if __name__ == "__main__":
    cudd_manager = Cudd()

    if GRIDWORLD:
        # grid world files
        domain_file_path = PROJECT_ROOT + "/pddl_files/grid_world/domain.pddl"
        if OBSTACLE:
            problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}_obstacle1.pddl"
        else:
            problem_file_path = PROJECT_ROOT + f"/pddl_files/grid_world/problem{GRID_WORLD_SIZE}_{GRID_WORLD_SIZE}.pddl"
    
        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'
        
        # grid world dictionary
        wgt_dict = {
            "moveleft"  : 1,
            "moveright" : 2,
            "moveup"    : 3,
            "movedown"  : 4
            }


        gridworld_handle = SimpleGridWorld(domain_file=domain_file_path,
                                           problem_file=problem_file_path,
                                           formulas=formulas,
                                           manager=cudd_manager,
                                           algorithm=algo,
                                           weight_dict=wgt_dict,
                                           ltlf_flag=USE_LTLF,
                                           dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                           verbose=False,
                                           plot_ts=False,
                                           plot_obs=False,
                                           plot=False)
        
        # build the TS and DFA(s)
        gridworld_handle.build_abstraction()
        policy: dict = gridworld_handle.solve(verbose=False)
        gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE)

    elif FRANKAWORLD:
        #Franka World files 
        domain_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/problem.pddl"

        if DIJKSTRAS:
            algo = 'dijkstras'
        elif ASTAR:
            algo = 'astar'
        else:
            algo = 'bfs'
        
        # grid world dictionary
        wgt_dict = {
            "transit" : 1,
            "grasp"   : 2,
            "transfer": 3,
            "release" : 4,
            
            }

        # frankaworld stuff
        frankaworld_handle = FrankaWorld(domain_file=domain_file_path,
                                         problem_file=problem_file_path,
                                         formulas=formulas,
                                         manager=cudd_manager,
                                         weight_dict=wgt_dict,
                                         ltlf_flag=USE_LTLF,
                                         dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                         algorithm=algo,
                                         verbose=False,
                                         plot_ts=False,
                                         plot_obs=False,
                                         plot=False)

        # build the abstraction
        frankaworld_handle.build_abstraction()
        policy: dict = frankaworld_handle.solve(verbose=False)
        frankaworld_handle.simulate(action_dict=policy, print_strategy=True)
    
    elif STRATEGY_SYNTHESIS:
        #Franka World files 
        if TWO_PLAYER_GAME:
            domain_file_path = PROJECT_ROOT + "/pddl_files/dynamic_franka_world/domain.pddl"
            problem_file_path = PROJECT_ROOT + "/pddl_files/dynamic_franka_world/problem.pddl"
        
        else:
            domain_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/domain.pddl"
            problem_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/problem.pddl"


        wgt_dict = {}

        # partitioned frankaworld stuff
        frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                        problem_file=problem_file_path,
                                                        formulas=formulas,
                                                        manager=cudd_manager,
                                                        weight_dict=wgt_dict,
                                                        ltlf_flag=USE_LTLF,
                                                        dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                        algorithm='qual',
                                                        verbose=False,
                                                        plot_ts=False,
                                                        plot_obs=False,
                                                        plot=False)
        # build the abstraction
        frankapartition_handle.build_abstraction()

    else:
        warnings.warn("Please set atleast one flag to True - FRANKAWORLD or GRIDWORLD!")