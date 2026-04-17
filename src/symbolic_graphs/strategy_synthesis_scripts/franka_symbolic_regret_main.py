import re
import sys
import time
import math
import warnings


from bidict import bidict
from collections import defaultdict
from typing import Union, List, Tuple, Dict

from cudd import Cudd, BDD, ADD

from src.algorithms.strategy_synthesis import SymbolicGraphOfUtlCooperativeGame, GraphofBRAdvGame

from src.symbolic_graphs.symbolic_regret_graphs import SymbolicGraphOfUtility
from src.symbolic_graphs.hybrid_regret_graphs import HybridGraphOfBR

from src.symbolic_graphs.strategy_synthesis_scripts import FrankaRegretSynthesis


class FrankaSymbolicRegretSynthesis(FrankaRegretSynthesis):

    def __init__(self,
                 domain_file: str,
                 problem_file: str,
                 formulas: Union[List, str],
                 manager: Cudd,
                 algorithm: str,
                 sup_locs: List[str],
                 top_locs: List[str],
                 weight_dict: dict = {},
                 ltlf_flag: bool = True,
                 dyn_var_ord: bool = False,
                 verbose: bool = False,
                 plot_ts: bool = False,
                 plot_obs: bool = False,
                 plot_dfa: bool = False,
                 plot: bool = False,
                 print_layer: bool = False,
                 create_lbls: bool = True,
                 weighting_factor: int = 1,
                 reg_factor: float = 1):
        super().__init__(domain_file=domain_file,
                         problem_file=problem_file,
                         formulas=formulas,
                         manager=manager,
                         algorithm=algorithm,
                         sup_locs=sup_locs,
                         top_locs=top_locs,
                         weight_dict=weight_dict,
                         ltlf_flag=ltlf_flag,
                         dyn_var_ord=dyn_var_ord,
                         verbose=verbose,
                         plot_ts=plot_ts,
                         plot_obs=plot_obs,
                         plot_dfa=plot_dfa,
                         print_layer=print_layer,
                         plot=plot,
                         create_lbls=create_lbls,
                         weighting_factor=weighting_factor,
                         reg_factor=reg_factor)
    

    def build_add_graph_of_utility(self, verbose: bool = False, just_adv_game: bool = False, monolithic_tr: bool = False):
        """
         Override parent method to constrcut Graph of Utility in symbolic fashion.
        """
        print("******************Computing Min-Max (aVal) on the original graph******************")
        min_max_handle = self.get_energy_budget(verbose=verbose, just_adv_game=just_adv_game, monolithic_tr=monolithic_tr)
        if just_adv_game:
            return

        # get the max action cost
        max_action_cost: int = min_max_handle._get_max_tr_action_cost()

        print("******************Constructing Graph of utility******************")
        graph_of_utls_handle = SymbolicGraphOfUtility(curr_vars=self.ts_x_list,
                                                      lbl_vars=self.ts_obs_list,
                                                      state_utls_vars=self.prod_utls_vars,
                                                      robot_action_vars=self.ts_robot_vars,
                                                      human_action_vars=self.ts_human_vars,
                                                      task=self.ts_handle.task,
                                                      domain=self.ts_handle.domain,
                                                      ts_state_map=self.ts_handle.pred_int_map,
                                                      ts_states=self.ts_handle.ts_states,
                                                      manager=self.manager,
                                                      weight_dict=self.ts_handle.weight_dict,
                                                      seg_actions=self.ts_handle.actions,
                                                      ts_state_lbls=self.ts_handle.state_lbls,
                                                      dfa_state_vars=self.dfa_x_list,
                                                      sup_locs=self.sup_locs,
                                                      top_locs=self.top_locs,
                                                      dfa_handle=self.dfa_handle,
                                                      ts_handle=self.ts_handle,
                                                      int_weight_dict=self.int_weight_dict,
                                                      budget=self.reg_energy_budget,
                                                      max_ts_action_cost=max_action_cost)
        

        start: float = time.time()
        graph_of_utls_handle.create_sym_tr_actions(mod_act_dict=self.mod_act_dict,
                                                   verbose=False)
        stop: float = time.time()
        print("Time took for constructing the Sym TR for Graph of Utility: ", stop - start)
        self.comp_dict['GoU_TR_time'] = stop - start

        # compute reach states from the init state
        start: float = time.time()
        graph_of_utls_handle.compute_graph_of_utility_reachable_states(mod_act_dict=self.mod_act_dict, 
                                                                       print_layers=self.print_layers,
                                                                       boxes=self.task_boxes,
                                                                       verbose=False)
        stop: float = time.time()
        print("Time took for constructing Reachable states on Graph of Utility: ", stop - start)
        self.comp_dict['GoU_Reachable_Time'] = stop - start
        self.graph_of_utls_handle = graph_of_utls_handle
    

    def solve(self, verbose: bool = False, just_adv_game: bool = False, run_monitor: bool = False, monolithic_tr: bool = False):
        """
         Overrides base method to first construct the required graph and then run Value Iteration. 

         Set just_adv_game flag to True if you just want to play an adversarial game.

         Set run_monitor to True if you want the human to choose strategy for both player.
          Note, for Robot player, we are restricted to regret minimizing strategies only. For Human player. we can select any strategy. 
        """
        print("**********************************************************************************************************")
        print("******************************************** TR: {approach} ***********************************************".format(approach='Monolithic' if monolithic_tr else 'Partitioned'))
        print("**********************************************************************************************************")
        # constuct graph of utility
        self.build_add_graph_of_utility(verbose=verbose, just_adv_game=just_adv_game, monolithic_tr=monolithic_tr)
        if just_adv_game:
            return

        print("******************Computing cVals on Graph of utility******************")

        # compute the min-min value from each state
        gou_min_min_handle = SymbolicGraphOfUtlCooperativeGame(gou_handle=self.graph_of_utls_handle,
                                                               ts_handle=self.ts_handle,
                                                               dfa_handle=self.dfa_handle,
                                                               ts_curr_vars=self.ts_x_list,
                                                               dfa_curr_vars=self.dfa_x_list,
                                                               sys_act_vars=self.ts_robot_vars,
                                                               env_act_vars=self.ts_human_vars,
                                                               ts_obs_vars=self.ts_obs_list,
                                                               ts_utls_vars=self.prod_utls_vars,
                                                               cudd_manager=self.manager,
                                                               monolithic_tr=monolithic_tr)
        
        # compute the cooperative value from each prod state in the graph of utility
        start: float = time.time()
        cvals: ADD = gou_min_min_handle.solve(verbose=False, print_layers=self.print_layers)
        stop: float = time.time()
        print("Time took for computing cVals is: ", stop - start)
        self.comp_dict['GoU_synth_time'] = stop - start

        # sanity checking
        print("******************Computing BA Vals on Graph of utility******************")
        start: float = time.time()
        # compute the best alternative from each edge for cumulative payoff
        self.graph_of_utls_handle.get_best_alternatives(mod_act_dict=self.mod_act_dict,
                                                        cooperative_vals=cvals,
                                                        print_layers=self.print_layers,
                                                        verbose=False)
        stop: float = time.time()
        print("Time took for computing the set of best alternatives: ", stop - start)
        self.comp_dict['BA_Comp_Time'] = stop - start
        # construct additional boolean vars for set of best alternative values
        self.prod_ba_vars: List[ADD] = self._create_symbolic_lbl_vars(state_lbls=self.graph_of_utls_handle.ba_set,
                                                                      state_var_name='r',
                                                                      add_flag=True)
        self.abs_dict['brVars'] = len(self.prod_ba_vars)

        print("******************Constructing Graph of Best Response******************")
        # construct of Best response G^{br}
        graph_of_br_handle = HybridGraphOfBR(curr_vars=self.ts_x_list,
                                             lbl_vars=self.ts_obs_list,
                                             robot_action_vars=self.ts_robot_vars,
                                             human_action_vars=self.ts_human_vars,
                                             task=self.ts_handle.task,
                                             domain=self.ts_handle.domain,
                                             ts_state_map=self.ts_handle.pred_int_map,
                                             ts_states=self.ts_handle.ts_states,
                                             manager=self.manager,
                                             weight_dict=self.ts_handle.weight_dict,
                                             seg_actions=self.ts_handle.actions,
                                             ts_state_lbls=self.ts_handle.state_lbls,
                                             dfa_state_vars=self.dfa_x_list,
                                             sup_locs=self.sup_locs,
                                             top_locs=self.top_locs,
                                             ts_handle=self.ts_handle,
                                             dfa_handle=self.dfa_handle,
                                             symbolic_gou_handle=self.graph_of_utls_handle,
                                             prod_ba_vars=self.prod_ba_vars)

        start: float = time.time()
        graph_of_br_handle.construct_graph_of_best_response(mod_act_dict=self.mod_act_dict,
                                                            print_layers=self.print_layers,
                                                            print_leaf_nodes=False,
                                                            verbose=False,
                                                            debug=True)
        stop: float = time.time()
        print("Time took for costructing the Graph of Best Response: ", stop - start)
        self.comp_dict['GoBR_TR_Creation_Time'] = stop - start


        # compute regret-minmizing strategies
        gbr_min_max_handle =  GraphofBRAdvGame(prod_gbr_handle=graph_of_br_handle,
                                               prod_gou_handle=self.graph_of_utls_handle,
                                               ts_handle=self.ts_handle,
                                               dfa_handle=self.dfa_handle,
                                               ts_curr_vars=self.ts_x_list,
                                               dfa_curr_vars=self.dfa_x_list,
                                               ts_obs_vars=self.ts_obs_list,
                                               prod_utls_vars=self.prod_utls_vars,
                                               prod_ba_vars=self.prod_ba_vars,
                                               sys_act_vars=self.ts_robot_vars,
                                               env_act_vars=self.ts_human_vars,
                                               cudd_manager=self.manager,
                                               monolithic_tr=monolithic_tr)
        
        print("******************Computing Regret Minimizing strategies on Graph of Best Response******************")
        start: float = time.time()
        reg_str: ADD = gbr_min_max_handle.solve(verbose=False,  print_layers=self.print_layers)
        stop: float = time.time()
        print("Time took for computing min-max strs on the Graph of best Response: ", stop - start)
        self.comp_dict['Synth_time'] = stop - start

        if reg_str:
            gbr_min_max_handle.roll_out_strategy(strategy=reg_str, verbose=True, ask_usr_input=run_monitor)
            print("Done Rolling out.")

