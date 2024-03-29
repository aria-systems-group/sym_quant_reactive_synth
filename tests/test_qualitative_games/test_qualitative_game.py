import os
import unittest

from typing import List
from cudd import Cudd, BDD

from src.algorithms.strategy_synthesis import ReachabilityGame
from src.symbolic_graphs.strategy_synthesis_scripts import FrankaPartitionedWorld



# config flags 

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

TWO_PLAYER_GAME: bool = True # Set this flag to true when you want to contruct a two-player game env.
TWO_PLAYER_GAME_BND: bool = False  # Set this flag to true when you want to construct som bounded no. off human interventions.
GAME_ALGORITHM = 'qual' # choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game

HUMAN_INT_BND: int = 0  # DOES not matter

USE_LTLF: bool = True # Construct DFA from LTLf

DYNAMIC_VAR_ORDERING: bool = False

SUP_LOC = []
TOP_LOC = []


class TestReachabilityGame(unittest.TestCase):
    def test_abstraction(self):
        """
         Check all the tests related abstraction construction for Winning Strategy
        """
        # TEST for various formulas - THe formulas are the same as the Qunatitative games. 
        formulas = ['F(p00 & p11)', 
                    'F(p01 & XF(p17))',
                    'F(p01 & p16 & F(p17 & F(p16)))',
                    'F(p01 | p11)']

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"
        
        # correct values
        cor_total_vars: List[int] = [16, 17, 17, 16]

        for task_id, task in enumerate(formulas):
            cudd_manager = Cudd()


            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=problem_file_path,
                                                            formulas=[task],
                                                            manager=cudd_manager,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            weight_dict=None,
                                                            ltlf_flag=USE_LTLF,
                                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                            algorithm=GAME_ALGORITHM,
                                                            verbose=False,
                                                            plot_ts=False,
                                                            plot_obs=False,
                                                            plot=False)

            ### ENSURE that the support locations are empty in the actual config file - else this might throw errors!
            # build the abstraction
            frankapartition_handle.build_abstraction(dynamic_env=TWO_PLAYER_GAME,
                                                    bnd_dynamic_env=TWO_PLAYER_GAME_BND,
                                                    max_human_int=HUMAN_INT_BND)

            total_vars: int = cudd_manager.size()
            
            self.assertEqual(total_vars,
                             cor_total_vars[task_id],
                             msg=f"Mismatch in the Total # of boolean vars required to construct the Symbolic Weighted Abstraction for formula {task}")
            
            self.assertEqual(frankapartition_handle.ts_handle.ecount,
                             340,
                             msg=f"Mismatch in the # of edges in the Symbolic Weighted Abstraction for formula {task}.")
    

    def test_synthesis(self):
        """
         Check all the tests related winning strategy synthesis assuming human to be adversarial. 
        """

        # TEST for various formulas 
        formulas = ['F(p00 & p11)',    # Adv strategy should not exist
                    'F(p01 & XF(p17))',  # Adv exists as the robot can force the human
                    #(a slight variant from the Quant-adv game where the boxes are being manipulated in their initial region of operations )
                    'F(p01 & p16 & F(p17 & F(p16)))',  # Adv. strategy exists.
                    'F(p01 | p11)']  # Adv. strategy will prefer p01 as p11 is not possible

        domain_file_path = PROJECT_ROOT + "/pddl_files/domain.pddl"
        problem_file_path = PROJECT_ROOT + "/pddl_files/problem.pddl"


        # No. of iteration req. to reach the fixed point
        cor_fp: List[int] = [9, 9, 16, 4]

        for task_id, task in enumerate(formulas):
            cudd_manager = Cudd()
            frankapartition_handle = FrankaPartitionedWorld(domain_file=domain_file_path,
                                                            problem_file=problem_file_path,
                                                            formulas=[task],
                                                            manager=cudd_manager,
                                                            sup_locs=SUP_LOC,
                                                            top_locs=TOP_LOC,
                                                            weight_dict=None,
                                                            ltlf_flag=USE_LTLF,
                                                            dyn_var_ord=DYNAMIC_VAR_ORDERING,
                                                            algorithm=GAME_ALGORITHM,
                                                            verbose=False,
                                                            plot_ts=False,
                                                            plot_obs=False,
                                                            plot=False)
            # build the abstraction
            frankapartition_handle.build_abstraction(dynamic_env=TWO_PLAYER_GAME,
                                                     bnd_dynamic_env=TWO_PLAYER_GAME_BND,
                                                     max_human_int=HUMAN_INT_BND)
            
            reachability_handle = ReachabilityGame(ts_handle=frankapartition_handle.ts_handle,
                                                   dfa_handle=frankapartition_handle.dfa_handle,
                                                   ts_curr_vars=frankapartition_handle.ts_x_list,
                                                   dfa_curr_vars=frankapartition_handle.dfa_x_list,
                                                   ts_obs_vars=frankapartition_handle.ts_obs_list,
                                                   sys_act_vars=frankapartition_handle.ts_robot_vars,
                                                   env_act_vars=frankapartition_handle.ts_human_vars,
                                                   cudd_manager=frankapartition_handle.manager)

            win_str: BDD = reachability_handle.solve(verbose=False)

            # ensure winning strategy exisits
            self.assertNotEqual(win_str, cudd_manager.bddZero(), "Could not synthesize a winning strategy!")

            if win_str:
                # ensure that you reach the fixed point correctly
                self.assertEqual(max(reachability_handle.stra_list.keys()), cor_fp[task_id], "Error computing the fixed point.")

                # this has to be done to ensure that the strategy synthesized does indeed reach the accepting state
                reachability_handle.roll_out_strategy(transducer=win_str, verbose=False)
        
            del reachability_handle.stra_list
            del reachability_handle.winning_states


if __name__ == "__main__":
    unittest.main()
