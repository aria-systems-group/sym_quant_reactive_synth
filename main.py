import sys
import warnings

from cudd import Cudd

from src.symbolic_graphs.graph_search_scripts import SimpleGridWorld, FrankaWorld
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld, FrankaRegretSynthesis, FrankaSymbolicRegretSynthesis

from utls import *
from config import *


if __name__ == "__main__":
    for iteration in range(1):
        print("**********************************************************************************************************")
        print(f"************************************** Iteration: {iteration}********************************************")
        print("**********************************************************************************************************")
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
            print("No. of Boolean Variables in the memory:", cudd_manager.size())
            policy: dict = gridworld_handle.solve(verbose=False)
            gridworld_handle.simulate(action_dict=policy, gridworld_size=GRID_WORLD_SIZE)

        elif FRANKAWORLD:
            # Franka World files 
            domain_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/domain.pddl"
            problem_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/problem.pddl"

            if DIJKSTRAS:
                algo = 'dijkstras'
            elif ASTAR:
                algo = 'astar'
            else:
                algo = 'bfs'
            
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
                                            sup_locs=SUP_LOC,
                                            top_locs=TOP_LOC,
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
            # sys.exit(-1)
            print("No. of Boolean Variables in the memory:", cudd_manager.size())
            policy: dict = frankaworld_handle.solve(verbose=False)
            frankaworld_handle.simulate(action_dict=policy, print_strategy=True)
        
        elif STRATEGY_SYNTHESIS:
            # Franka World files 
            if TWO_PLAYER_GAME:
                domain_file_path = PROJECT_ROOT + "/pddl_files/dynamic_franka_world/domain.pddl"
                problem_file_path = PROJECT_ROOT + "/pddl_files/dynamic_franka_world/problem.pddl"
            
            elif TWO_PLAYER_GAME_BND:
                domain_file_path = PROJECT_ROOT + "/pddl_files/bounded_dynamic_franka_world/domain.pddl"
                problem_file_path = PROJECT_ROOT + "/pddl_files/bounded_dynamic_franka_world/problem.pddl"

                assert HUMAN_INT_BND >= 0, "Please make sure you enter a non-negative number of human interventions."

            else:
                domain_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/domain.pddl"
                problem_file_path = PROJECT_ROOT + "/pddl_files/simple_franka_world/problem.pddl"


            wgt_dict = {
                "transit" : 1,
                "grasp"   : 1,
                "transfer": 1,
                "release" : 1,
                "human": 0
                }


            # partitioned frankaworld stuff
            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=problem_file_path,
                                                            formulas=formulas,
                                                            manager=cudd_manager,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            weight_dict=wgt_dict,
                                                            ltlf_flag=USE_LTLF,
                                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                            algorithm=GAME_ALGORITHM,
                                                            verbose=False,
                                                            plot_ts=False,
                                                            plot_obs=False,
                                                            plot=False)
            
            if 'quant' in GAME_ALGORITHM:
                assert TWO_PLAYER_GAME_BND is False, "We do not have symbolic bounded quantitative synthesis implemented yet. Please set TWO_PLAYER_GAME flag to True"

            elif GAME_ALGORITHM == 'qual':
                assert TWO_PLAYER_GAME is not TWO_PLAYER_GAME_BND, "Please set only one flag to True - BND_TWO_PLAYER_GAME or TWO_PLAYER_GAME!"
            else:
                warnings.warn("Make sure you select atleast one Algorithm - 'qual' or 'quant-adv' or 'quant-coop'")
                sys.exit(-1)

            # build the abstraction
            frankapartition_handle.build_abstraction(dynamic_env=TWO_PLAYER_GAME,
                                                     bnd_dynamic_env=TWO_PLAYER_GAME_BND,
                                                     max_human_int=HUMAN_INT_BND)
            # sys.exit(-1)                                      
            print(f"****************** # Total Boolean Variables: {cudd_manager.size()} ******************")
            frankapartition_handle.solve(verbose=True, monolithic_tr=MONOLITHIC_TR)

        elif REGRET_SYNTHESIS:
            # domain_file_path = PROJECT_ROOT + "/pddl_files/franka_regret_world/two_blocks/domain.pddl"
            # problem_file_path = PROJECT_ROOT + "/pddl_files/franka_regret_world/two_blocks/problem.pddl"

            ##### ARCH CONSTRUCTION DOMAIN
            # domain_file_path = PROJECT_ROOT + "/pddl_files/franka_regret_world/arch/domain.pddl"
            # problem_file_path = PROJECT_ROOT + "/pddl_files/franka_regret_world/arch/problem.pddl"

            ##### Simple Test domain - formula on line 75 in config.py script
            # domain_file_path = PROJECT_ROOT + "/pddl_files/franka_regret_world/test/domain.pddl"
            # problem_file_path = PROJECT_ROOT + "/pddl_files/franka_regret_world/test/problem.pddl"

            ##### IROS 23 benchmark - varying boxes domain
            domain_file_path = PROJECT_ROOT + "/pddl_files/iros23_pddl_files/domain.pddl"
            problem_file_path = PROJECT_ROOT + "/pddl_files/iros23_pddl_files/varying_boxes/p00.pddl"

            wgt_dict = {
                "transit" : 1,
                "grasp"   : 1,
                "transfer": 1,
                "release" : 1,
                "human": 0
                }
            
            if REGRET_HYBRID:
                regret_synthesis_handle = FrankaRegretSynthesis(domain_file=domain_file_path,
                                                                problem_file=problem_file_path,
                                                                formulas=formulas,
                                                                manager=cudd_manager,
                                                                sup_locs=SUP_LOC,
                                                                top_locs=TOP_LOC,
                                                                weight_dict=wgt_dict,
                                                                ltlf_flag=USE_LTLF,
                                                                dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                                weighting_factor=3,
                                                                reg_factor=1.25,
                                                                algorithm=None,
                                                                verbose=False,
                                                                print_layer=False,
                                                                plot_ts=False,
                                                                plot_obs=False,
                                                                plot=False)
            else:
                regret_synthesis_handle = FrankaSymbolicRegretSynthesis(domain_file=domain_file_path,
                                                                        problem_file=problem_file_path,
                                                                        formulas=formulas,
                                                                        manager=cudd_manager,
                                                                        sup_locs=SUP_LOC,
                                                                        top_locs=TOP_LOC,
                                                                        weight_dict=wgt_dict,
                                                                        ltlf_flag=USE_LTLF,
                                                                        dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                                        weighting_factor=3,
                                                                        reg_factor=1.25,
                                                                        algorithm=None,
                                                                        verbose=False,
                                                                        print_layer=False,
                                                                        plot_ts=False,
                                                                        plot_obs=False,
                                                                        plot=False)
                        
            regret_synthesis_handle.build_abstraction()
            regret_synthesis_handle.solve(verbose=False, just_adv_game=False, run_monitor=False, monolithic_tr=MONOLITHIC_TR)

            print(f"****************** # Total Boolean Variables: { cudd_manager.size()} ******************")

        else:
            warnings.warn("Please set atleast one flag to True - GRIDWORLD, FRANKAWORLD, STRATEGY_SYNTHESIS, or REGRET_SYNTHESIS!")
            sys.exit(-1)
        
        # convert bytes to MegaBytes and print the Memory usage
        print(f"Memory in use (MB): {cudd_manager.readMemoryInUse()/(10**6)}")