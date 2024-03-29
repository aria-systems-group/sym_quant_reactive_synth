import re
import sys
import time
import math
import copy
import warnings

from bidict import bidict
from itertools import product, combinations
from collections import defaultdict
from typing import Tuple, List, Dict, Union

from cudd import Cudd, BDD, ADD

from src.explicit_graphs import CausalGraph

from src.symbolic_graphs import SymbolicDFAFranka, SymbolicAddDFAFranka
from src.symbolic_graphs import SymbolicFrankaTransitionSystem, SymbolicWeightedFrankaTransitionSystem

from src.algorithms.blind_search import SymbolicSearch, MultipleFormulaBFS
from src.algorithms.weighted_search import SymbolicDijkstraSearch, MultipleFormulaDijkstra
from src.algorithms.weighted_search import SymbolicBDDAStar, MultipleFormulaBDDAstar

from src.simulate_strategy import roll_out_franka_strategy, roll_out_franka_strategy_nLTL

from .base_main import BaseSymMain

from utls import *


class FrankaWorld(BaseSymMain):

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
                 create_lbls: bool = True):
        super().__init__(domain_file, problem_file, formulas, manager, plot_dfa, ltlf_flag, dyn_var_ord)
        self.algorithm: str = algorithm
        self.weight_dict: Dict[str, int] = weight_dict

        self.verbose: bool = verbose
        self.plot: bool = plot
        self.plot_ts: bool = plot_ts
        self.plot_obs: bool = plot_obs
        self.dyn_var_ordering: bool = dyn_var_ord

        # maps individual predicates to a unique int
        self.pred_int_map: bidict = bidict({})
        self.create_lbls: bool = create_lbls

        # support and top locations referred during the Arch Abstarction construction
        self.sup_locs = sup_locs
        self.top_locs = top_locs

        self.ts_handle: Union[SymbolicFrankaTransitionSystem, SymbolicWeightedFrankaTransitionSystem] = None
        self.dfa_handle_list: Union[SymbolicDFAFranka, SymbolicAddDFAFranka] = None

        self.ts_x_list: List[ADD] = None
        self.ts_y_list: List[ADD] = None
        self.ts_obs_list: List[ADD] = None

        self.dfa_x_list: List[ADD] = None
        self.dfa_y_list: List[ADD] = None
    

    def build_abstraction(self, draw_causal_graph: bool = False):
        """
        A main function that construct a symbolic Franka World TS and its corresponsing DFA
        """
        print("*****************Creating Boolean variables for Frankaworld!*****************")
        if self.algorithm in ['dijkstras','astar']:
            # All vars (TS, DFA and Predicate) are of type ADDs
            sym_tr, ts_curr_state, ts_next_state, ts_lbl_states = self.build_weighted_add_abstraction(draw_causal_graph=draw_causal_graph)
            
            # The tuple contains the DFA handle, DFA curr and next vars in this specific order
            dfa_tr, dfa_curr_state, dfa_next_state = self.build_add_symbolic_dfa(sym_tr_handle=sym_tr)
        
        elif self.algorithm == 'bfs':
            sym_tr, ts_curr_state, ts_next_state, ts_lbl_states = self.build_bdd_abstraction(draw_causal_graph=draw_causal_graph)

            dfa_tr, dfa_curr_state, dfa_next_state = self.build_bdd_symbolic_dfa(sym_tr_handle=sym_tr)
        
        else:
            warnings.warn("Please enter a valid graph search algorthim. Currently Available - bfs (BDD), dijkstras (BDD/ADD), astar (BDD/ADD)")


        self.ts_handle = sym_tr
        self.dfa_handle_list = dfa_tr

        self.ts_x_list = ts_curr_state
        self.ts_y_list = ts_next_state
        self.ts_obs_list = ts_lbl_states

        self.dfa_x_list = dfa_curr_state
        self.dfa_y_list = dfa_next_state


        if self.dyn_var_ordering:
            self.set_variable_reordering(make_tree_node=True,
                                         ts_sym_var_len=len(ts_curr_state),
                                         ts_obs_var_len=len(ts_lbl_states))
    

    def build_bdd_symbolic_dfa(self, sym_tr_handle: SymbolicFrankaTransitionSystem) -> Tuple[List[SymbolicDFAFranka], List[BDD], List[BDD]]:
        """
         This function calls Symbolic Franka DFA to decode the edge formulas into state lbls as per the symbolic state lbl dictionary
          and construct the symbolic TR accoridngly.
        """

        # create a list of DFAs
        DFA_handles = []
        DFA_curr_vars = []
        DFA_nxt_vars = []

        for _idx, fmla in enumerate(self.formulas):
            # create different boolean variables for different DFAs - [ai_0 for ith DFA]
            dfa_curr_state, dfa_next_state, _dfa = self.create_symbolic_dfa_graph(formula= fmla,
                                                                                  dfa_num=_idx)

            # create TR corresponding to each DFA - dfa name is only used dumping graph 
            dfa_tr = SymbolicDFAFranka(curr_states=dfa_curr_state,
                                       next_states=dfa_next_state,
                                       predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                       pred_int_map=sym_tr_handle.pred_int_map,
                                       manager=self.manager,
                                       dfa=_dfa,
                                       ltlf_flag=self.ltlf_flag,
                                       dfa_name=f'dfa_{_idx}')
            if self.ltlf_flag:
                dfa_tr.create_symbolic_ltlf_transition_system(verbose=False, plot=self.plot_dfa)
            else:
                raise NotImplementedError()

            # We extend DFA vars list as we dont need them stored in separate lists
            DFA_handles.append(dfa_tr)
            DFA_curr_vars.extend(dfa_curr_state)
            DFA_nxt_vars.extend(dfa_next_state)
        
        return DFA_handles, DFA_curr_vars, DFA_nxt_vars
    

    def build_add_symbolic_dfa(self, sym_tr_handle: SymbolicWeightedFrankaTransitionSystem) -> Tuple[List[SymbolicAddDFAFranka], List[ADD], List[ADD]]:
        """
        A helper function to build a symbolic DFA given a formula from ADD Variables.
        """      
        # create a list of DFAs
        DFA_handles = []
        DFA_curr_vars = []
        DFA_nxt_vars = []

        for _idx, fmla in enumerate(self.formulas):
            # create different ADD variables for different DFAs
            add_dfa_curr_state, add_dfa_next_state, _dfa = self.create_symbolic_dfa_graph(formula=fmla,
                                                                                          dfa_num=_idx,
                                                                                          add_flag=True)

            # create TR corresponding to each DFA - dfa name is only used dumping graph 
            dfa_tr = SymbolicAddDFAFranka(curr_states=add_dfa_curr_state,
                                          next_states=add_dfa_next_state,
                                          predicate_add_sym_map_lbl=sym_tr_handle.predicate_add_sym_map_lbl,
                                          predicate_sym_map_lbl=sym_tr_handle.predicate_sym_map_lbl,
                                          pred_int_map=sym_tr_handle.pred_int_map,
                                          manager=self.manager,
                                          dfa=_dfa,
                                          ltlf_flag=self.ltlf_flag,
                                          dfa_name=f'dfa_{_idx}')
            
            if self.ltlf_flag:
                dfa_tr.create_symbolic_ltlf_transition_system(verbose=self.verbose, plot=self.plot_dfa)
            else:
                raise NotImplementedError()

            # We extend DFA vars list as we dont need them stored in separate lists
            DFA_handles.append(dfa_tr)
            DFA_curr_vars.extend(add_dfa_curr_state)
            DFA_nxt_vars.extend(add_dfa_next_state)
        
        return DFA_handles, DFA_curr_vars, DFA_nxt_vars
    

    def set_variable_reordering(self, make_tree_node: bool = False, **kwargs):
        """
        This function is called when DYNAMIC_VAR_ORDERING is True.

        Different ways to speed up the process
        1. AutodynaEnable() - Enable Dyanmic variable reordering
        2. ReorderingStatus() - Return the current reordering status and default method
        3. EnablingOrderingMonitoring() - Enable monitoring of a variable order 
        4. maxReorderings() - Read and set maximum number of variable reorderings 
        5. EnablereorderingReport() - Enable reporting of variable reordering

        MakeTreeNode() - Allows us to specify constraints over groups of variables. For example, we can constrain x, x'
        to always be contiguous. Thus, the relative ordering within the group is left unchanged. 

        MTR takes in two args -
        low: 
        size: 2 (grouping curr state vars and their corresponding primes together)

        """
        self.manager.autodynEnable()

        if make_tree_node:
            # Current, we follow the convention where we first build the TS variables, then the observations,
            # and finally the dfa variables. Within the TS and DFA, we pait vars and their primes together.
            # The observation variables are all grouped together as one.
            var_reorder_counter = 0  
            for i in range(self.manager.size()):
            # for i in range(kwargs['ts_sym_var_len']):
                if i<= kwargs['ts_sym_var_len']:
                    self.manager.makeTreeNode(2*i, 2)

        if self.verbose:
            self.manager.enableOrderingMonitoring()
        else:
            self.manager.enableReorderingReporting()
    

    def solve(self, verbose: bool = False) -> dict:
        """
          A function that calls the appropriate solver based on the algorithm specified and if single LTL of multiple formulas have been passed.
        """

        if len(self.formulas) > 1:
            start: float = time.time()
            if self.algorithm == 'dijkstras':
                graph_search = MultipleFormulaDijkstra(ts_handle=self.ts_handle,
                                                       dfa_handles=self.dfa_handle_list,
                                                       ts_curr_vars=self.ts_x_list,
                                                       ts_next_vars=self.ts_y_list,
                                                       dfa_curr_vars=self.dfa_x_list,
                                                       dfa_next_vars=self.dfa_y_list,
                                                       ts_obs_vars=self.ts_obs_list,
                                                       cudd_manager=self.manager)

                # call dijkstras for solving minimum cost path over nLTLs
                action_dict: dict = graph_search.composed_symbolic_dijkstra_nLTL(verbose=verbose)
            
            elif self.algorithm == 'astar':
                graph_search =  MultipleFormulaBDDAstar(ts_handle=self.ts_handle,
                                                        dfa_handles=self.dfa_handle_list,
                                                        ts_curr_vars=self.ts_x_list,
                                                        ts_next_vars=self.ts_y_list,
                                                        dfa_curr_vars=self.dfa_x_list,
                                                        dfa_next_vars=self.dfa_y_list,
                                                        ts_obs_vars=self.ts_obs_list,
                                                        cudd_manager=self.manager)
                # For A* we ignore heuristic computation time                                  
                start: float = time.time()
                action_dict = graph_search.composed_symbolic_Astar_search_nLTL(verbose=verbose)

            elif self.algorithm == 'bfs':
                graph_search = MultipleFormulaBFS(ts_handle=self.ts_handle,
                                                  dfa_handles=self.dfa_handle_list,
                                                  ts_curr_vars=self.ts_x_list,
                                                  ts_next_vars=self.ts_y_list,
                                                  dfa_curr_vars=self.dfa_x_list,
                                                  dfa_next_vars=self.dfa_y_list,
                                                  ts_obs_vars=self.ts_obs_list,
                                                  cudd_manager=self.manager)

                # call BFS for multiple formulas 
                action_dict: dict = graph_search.symbolic_bfs_nLTL(verbose=verbose)

            stop: float = time.time()
            print("Time took for plannig: ", stop - start)
        
        else:
            start: float = time.time()
            if self.algorithm == 'dijkstras':
                # shortest path graph search with Dijkstras
                graph_search =  SymbolicDijkstraSearch(ts_handle=self.ts_handle,
                                                       dfa_handle=self.dfa_handle_list[0],
                                                       ts_curr_vars=self.ts_x_list,
                                                       ts_next_vars=self.ts_y_list,
                                                       dfa_curr_vars=self.dfa_x_list,
                                                       dfa_next_vars=self.dfa_y_list,
                                                       ts_obs_vars=self.ts_obs_list,
                                                       cudd_manager=self.manager)

                action_dict = graph_search.composed_symbolic_dijkstra_wLTL(verbose=verbose)

            elif self.algorithm == 'astar':
                # shortest path graph search with Symbolic A*
                graph_search =  SymbolicBDDAStar(ts_handle=self.ts_handle,
                                                 dfa_handle=self.dfa_handle_list[0],
                                                 ts_curr_vars=self.ts_x_list,
                                                 ts_next_vars=self.ts_y_list,
                                                 dfa_curr_vars=self.dfa_x_list,
                                                 dfa_next_vars=self.dfa_y_list,
                                                 ts_obs_vars=self.ts_obs_list,
                                                 cudd_manager=self.manager)
                # For A* we ignore heuristic computation time                                  
                start: float = time.time()
                action_dict = graph_search.composed_symbolic_Astar_search(verbose=verbose)


            elif self.algorithm == 'bfs':
                graph_search = SymbolicSearch(ts_handle=self.ts_handle,
                                              dfa_handle=self.dfa_handle_list[0], 
                                              ts_curr_vars=self.ts_x_list,
                                              ts_next_vars=self.ts_y_list,
                                              dfa_curr_vars=self.dfa_x_list,
                                              dfa_next_vars=self.dfa_y_list,
                                              ts_obs_vars=self.ts_obs_list,
                                              cudd_manager=self.manager)

                action_dict = graph_search.composed_symbolic_bfs_wLTL(verbose=verbose, obs_flag=False)

            stop: float = time.time()
            print("Time took for plannig: ", stop - start)

        return action_dict
    

    def simulate(self, action_dict: dict, print_strategy: bool = False):
        """
        A function to simulate the synthesize policy for the gridworld agent.
        """
        ts_handle = self.ts_handle
        
        ts_curr_vars = self.ts_x_list
        ts_next_vars = self.ts_y_list
        
        dfa_curr_vars = self.dfa_x_list
        dfa_next_vars = self.dfa_y_list

        if len(self.formulas) > 1:
            dfa_handles = self.dfa_handle_list

            if self.algorithm in ['dijkstras','astar']:
                init_state_ts_sym = ts_handle.sym_add_init_states
                state_obs_dd = ts_handle.sym_add_state_labels
            
            else:
                init_state_ts_sym = ts_handle.sym_init_states
                state_obs_dd = ts_handle.sym_state_labels

            franka_strategy = roll_out_franka_strategy_nLTL(ts_handle=ts_handle,
                                                                dfa_handles=dfa_handles,
                                                                action_map=action_dict,
                                                                init_state_ts_sym=init_state_ts_sym,
                                                                state_obs_dd=state_obs_dd,
                                                                ts_curr_vars=ts_curr_vars,
                                                                ts_next_vars=ts_next_vars,
                                                                dfa_curr_vars=dfa_curr_vars,
                                                                dfa_next_vars=dfa_next_vars)

            if print_strategy:
                print("{:<30}".format('Action'))
                for _ts_state, _action in franka_strategy: 
                    print("{:<30}".format(_action,))

        else:
            dfa_handle = self.dfa_handle_list[0]

            if self.algorithm in ['dijkstras','astar']:
                init_state_ts = ts_handle.sym_add_init_states
                state_obs_dd = ts_handle.sym_add_state_labels
            else:
                init_state_ts = ts_handle.sym_init_states
                state_obs_dd = ts_handle.sym_state_labels

            franka_strategy = roll_out_franka_strategy(ts_handle=ts_handle,
                                                        dfa_handle=dfa_handle,
                                                        action_map=action_dict,
                                                        init_state_ts=init_state_ts,
                                                        state_obs_dd=state_obs_dd,
                                                        ts_curr_vars=ts_curr_vars,
                                                        ts_next_vars=ts_next_vars,
                                                        dfa_curr_vars=dfa_curr_vars,
                                                        dfa_next_vars=dfa_next_vars)

            if print_strategy:
                print("{:<30}".format('Action'))
                for _ts_state, _action in franka_strategy: 
                    print("{:<30}".format(_action,))
                    
    

    def _create_symbolic_lbl_vars(self, state_lbls: list, state_var_name: str, add_flag: bool = False) -> List[Union[BDD, ADD]]:
        """
         A function that create only one set of vars for the objects passed. This function does not create prime varibables. 
        """
        state_lbl_vars: list = []
        _num_of_sym_vars = self.manager.size()
        num: int = math.ceil(math.log2(len(state_lbls)))

        # happens when number of domain_facts passed as argument is 1
        if num == 0:
            num += 1
        
        for num_var in range(num):
            _var_index = num_var + _num_of_sym_vars
            if add_flag:
                state_lbl_vars.append(self.manager.addVar(_var_index, f'{state_var_name}{num_var}'))
            else:
                state_lbl_vars.append(self.manager.bddVar(_var_index, f'{state_var_name}{num_var}'))
        
        return state_lbl_vars


    def compute_valid_franka_state_tuples(self, robot_preds: Dict[str, list], on_preds: Dict[str, list], verbose: bool = False) -> list:
        """
         A function that take the cartesian product of all possbile robot states with all possible world configurations

         robot_preds: all ready, holding predicates along with valid (holding, to-loc) and (ready, to-obj) predicates
         on_preds: all possible grounded (n boxes with their location and gripper free) predicates and n-1 grounded predicates 
        
        The cartesian product gives all possible states of the Franka abstraction.
        """
        # there two typs of prodct, -robot conf where gripper free will have all boxes grounded
        # -robot ocnf where gripper is not free will have n-1 boxes grounded

        _valid_combos_free = list(product(robot_preds['gfree'], on_preds['nb']))
        _valid_combos_occ = list(product(robot_preds['gocc'], on_preds['b']))
        _valid_combos = _valid_combos_free + _valid_combos_occ

        if verbose:
            print(f"********************************* # Valid States in Frank abstraction: {len(_valid_combos)} *********************************")

        _state_tuples = []
        for _exp_state in _valid_combos:
            _state_tpl = []
            for pred in _exp_state:
                if isinstance(pred, tuple):
                    tmp_tuple = [self.pred_int_map[_s] for _s in pred]
                else:
                    tmp_tuple = [self.pred_int_map[pred]]
                _state_tpl.extend(tmp_tuple) 

            
            _state_tuples.append(tuple(sorted(_state_tpl)))

        return _state_tuples


    def _get_all_box_combos(self, boxes_dict: dict, predicate_dict: dict) -> Dict[str, list]:
        """
        The franka world has the world configuration (on b# l#) embedded into it's state defination. 
        Also, we could have all n but 1 boxes grounded with that single box (not grounded) being currently manipulated.
        
        Thus, a valid set of state labels be
            1) all boxes grounded - (on b0 l0)(on b1 l1)...
            2) all but 1 grounded - (on b0 l0)(~(on b1 l1))(on b2 l2)...
        
        Hence, we need to create enough Boolean variables to accomodate all these possible configurations.
        """
        parent_combo_list = {'nb': [],   # preds where all boxes are grounded ad gripper free
                             'b': []     # preds where n-1 boxes are grounded
                             }

        # create all grounded configurations
        all_preds = [val for _, val in boxes_dict.items()]
        all_preds += [predicate_dict['gripper']]
        all_combos = list(product(*all_preds, repeat=1))

        # when all the boxes are grouded then the gripper predicate is set to free
        parent_combo_list['nb'].extend(all_combos)


        # create n-1 combos
        num_of_boxes = len(boxes_dict)

        if num_of_boxes - 1 == 1:
            return parent_combo_list
        
        # create all possible n-1 combinations of all boxes thhat can be grounded
        combos = combinations([*boxes_dict.keys()], num_of_boxes - 1)

        # iterate through every possible combo
        for combo in combos:
            # iterate through the tuple of boxes and create their combos
            box_loc_list = [boxes_dict[box] for box in combo]
            parent_combo_list['b'].extend(list(product(*box_loc_list, repeat=1)))

        return parent_combo_list
    

    def post_process_world_conf(self, possible_lbl: dict, locs: List[str]) -> dict:
        """
         This function take as input a dict whose values is the list all possible world confg.
          We need to remove states where two or more boxes that share the same location 
        """
        new_possible_lbl = copy.deepcopy(possible_lbl)
        # dont_add = False
        for key, value in possible_lbl.items():
            _valid_lbls = []
            for lbl in value:
                dont_add = False
                for loc in locs:
                    if len(re.split(loc, str(lbl))) >= 3:
                        dont_add = True
                        break
                if not dont_add:
                    _valid_lbls.append(lbl)
            new_possible_lbl[key] = _valid_lbls
        
        return new_possible_lbl
    

    def _create_all_holding_to_loc_combos(self, predicate_dict: dict)-> List[tuple]:
        """
         A helper function that creates all the valid combinations of holding and to-loc predicates. 
         A valid combination is one where holding's box and location arguements are same as
         to-loc's box and location arguement. 
        """
        _valid_combos = []
        for b in predicate_dict['holding'].keys():
            for l in predicate_dict['holding'][b].keys():
                _valid_combos.extend(list(product(predicate_dict['holding'][b][l], predicate_dict['to_loc'][b][l])))
        
        return _valid_combos


    def _create_all_ready_to_obj_combos(self, predicate_dict: dict) -> List[tuple]:
        """
         A helper function that creates all the valid combinations of ready and to-obj predicates. 
         A valid combination is one where ready location arguement is same as to-obj location arguement. 
        """
        _valid_combos = []
        for key in predicate_dict['ready'].keys():
            if key != 'else':
                _valid_combos.extend(list(product(predicate_dict['ready'][key], predicate_dict['to_obj'][key])))

        return _valid_combos
    

    def compute_valid_predicates(self, predicates: List[str], boxes: List[str], locations: List[str]) -> Tuple[List, List, List]:
        """
        A helper function that segretaes the predicates as required by the symbolic transition relation. We separate them based on
         1) all gripper predicates - we do not need to create prime version for these
         2) all on predicates - we do need to create prime version for these
         3) all holding predicates 
         4) rest of the predicates - holding, ready, to-obj, and to-loc predicates. We do create prime version for these. 
        """

        predicate_dict = {
            'ready': defaultdict(lambda: []),
            'to_obj': defaultdict(lambda: []),
            'to_loc': defaultdict(lambda: defaultdict(lambda: [])),
            'holding': defaultdict(lambda: defaultdict(lambda: [])),
            'ready_all': [],
            'holding_all': [],
            'to_obj_all': [],
            'to_loc_all': [],
            'on': [],
            'gripper': []
        }

        # dictionary where we segreate on predicates based on boxes - all b0, b1 ,... into seperate list 
        boxes_dict = {box: [] for box in boxes} 

        # define patterns to find box ids and locations
        _loc_pattern = "[l|L][\d]+"
        _box_pattern = "[b|B][\d]+"

        for pred in predicates:
            if 'on' in pred:
                predicate_dict['on'].append(pred)
                for b in boxes:
                    if b in pred:
                        boxes_dict[b].append(pred)
                        break
            
            elif 'gripper' in pred:
                predicate_dict['gripper'].append(pred)

            else:
                # ready predicate is not parameterized by box
                if not 'ready' in pred:
                    _box_state: str = re.search(_box_pattern, pred).group()
                    _loc_state: str = re.search(_loc_pattern, pred).group()
                else:
                    # ready predicate can have else as a valid location 
                    if 'else' in pred:
                        _loc_state = 'else'
                    else:
                        _loc_state: str = re.search(_loc_pattern, pred).group()

                if 'holding' in pred:
                    predicate_dict['holding_all'].append(pred)
                    predicate_dict['holding'][_box_state][_loc_state].append(pred)
                elif 'ready' in pred:
                    predicate_dict['ready_all'].append(pred)
                    predicate_dict['ready'][_loc_state].append(pred)
                elif 'to-obj' in pred:
                    predicate_dict['to_obj_all'].append(pred)
                    predicate_dict['to_obj'][_loc_state].append(pred)
                elif 'to-loc' in  pred:
                    predicate_dict['to_loc_all'].append(pred)
                    predicate_dict['to_loc'][_box_state][_loc_state].append(pred)
        
        # create predicate int map
        _ind_pred_list = predicate_dict['ready_all'] + \
             predicate_dict['holding_all'] + predicate_dict['to_obj_all'] + predicate_dict['to_loc_all']
        _pred_map = {pred: num for num, pred in enumerate(_ind_pred_list)}
        _pred_map = bidict(_pred_map)

        # get all valid robot conf predicates
        _valid_robot_preds = {'gfree': [], 
                              'gocc': []}

        # we store valid robot conf into types, one where robot conf exisit when gripper is free and the other robot conf. where gripper is not free
        _valid_robot_preds['gfree'].extend(predicate_dict['ready_all'])
        _valid_robot_preds['gocc'].extend(predicate_dict['holding_all'])

        _valid_robot_preds['gfree'].extend(self._create_all_ready_to_obj_combos(predicate_dict))
        _valid_robot_preds['gocc'].extend(self._create_all_holding_to_loc_combos(predicate_dict))

        # create on predicate map
        len_robot_conf = len(_ind_pred_list)
        _pred_map.update({pred: len_robot_conf + num for num, pred in enumerate(predicate_dict['on'] + predicate_dict['gripper'])})

        # we create all n and n-1 combos
        # n combos when all boxes and gripper is not free are grounded 
        # and n-1 when one of the boxes is being manipulated and gripper is not free
        _valid_box_preds = self._get_all_box_combos(boxes_dict=boxes_dict, predicate_dict=predicate_dict)
        
        # when you have two objects, then individual on predicates are also valid combos 
        if len(_valid_box_preds['b']) == 0:
           _valid_box_preds['b'].extend(predicate_dict['on'])
        
        self.pred_int_map = _pred_map

        # update boxes dictionary with gripper 
        boxes_dict.update({'gripper': ['(gripper free)']})

        _valid_box_preds = self.post_process_world_conf(_valid_box_preds, locations)
        
        return _valid_robot_preds, _valid_box_preds, boxes_dict


    def create_symbolic_causal_graph(self, draw_causal_graph: bool = False, add_flag: bool = False) -> Tuple:
        """
        A function to create an instance of causal graph which call pyperplan. We access the task related properties pyperplan
        and create symbolic TR related to action.   

        _causal_graph_instance.task.facts: Grounded facts about the world 
        _causal_graph_instance.task.initial_sttates: initial condition(s) of the world
        _causal_graph_instance.task.goals:  Desired Final condition(s) of the world
        _causal_graph_instance.task.operators: Actions that the agent (Franka) can take from all the grounded facts

        """
        _causal_graph_instance = CausalGraph(problem_file=self.problem_file,
                                             domain_file=self.domain_file,
                                             draw=draw_causal_graph)

        _causal_graph_instance.build_causal_graph(add_cooccuring_edges=False, relabel=False)

        task_facts: List[str] = _causal_graph_instance.task.facts
        boxes: List[str] = _causal_graph_instance.task_objects
        locations: List[str] = _causal_graph_instance.task_locations

        # compute all valid preds of the robot conf and box conf.
        robot_preds, on_preds, box_preds = self.compute_valid_predicates(predicates=task_facts, boxes=boxes, locations=locations)
        
        # compute all the possible states
        ts_state_tuples = self.compute_valid_franka_state_tuples(robot_preds=robot_preds, on_preds=on_preds, verbose=True)

        curr_vars, next_vars = self.create_symbolic_vars(num_of_facts=len(ts_state_tuples),
                                                         add_flag=add_flag)
        
        # box_preds has predicated segregated as per boxes
        ts_lbl_vars = []
        for _id, b in enumerate(box_preds.keys()): 
            if add_flag:
                ts_lbl_vars.extend(self._create_symbolic_lbl_vars(state_lbls=box_preds[b],
                                                                    state_var_name=f'b{_id}_',
                                                                    add_flag=add_flag))
            # for Franka world with no human and edge weights, we store the bVars for each box in a list and append it to a parent list.
            # This is done to accomodate for SymbolicFrankaTransitionSystem._create_sym_state_label_map()'s implementation  
            else:
                 ts_lbl_vars.append(self._create_symbolic_lbl_vars(state_lbls=box_preds[b],
                                                                    state_var_name=f'b{_id}_',
                                                                    add_flag=add_flag))

        return _causal_graph_instance.task, _causal_graph_instance.problem.domain, curr_vars, next_vars, ts_state_tuples, ts_lbl_vars, boxes, box_preds
        

    def build_bdd_abstraction(self, draw_causal_graph: bool = False) -> Tuple[SymbolicFrankaTransitionSystem, List[BDD], List[BDD], List[BDD]]:
        """
         Main Function to Build Transition System that only represent valid edges without any weights
        """
        task, domain, ts_curr_vars, ts_next_vars, ts_state_tuples, ts_lbl_vars, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph)

        sym_tr = SymbolicFrankaTransitionSystem(curr_states=ts_curr_vars,
                                                next_states=ts_next_vars,
                                                lbl_states=ts_lbl_vars,
                                                task=task,
                                                domain=domain,
                                                ts_states=ts_state_tuples,
                                                ts_state_map=self.pred_int_map,
                                                manager=self.manager)
        start: float = time.time()
        sym_tr.create_transition_system_franka(boxes=boxes,
                                               state_lbls=possible_lbls,
                                               add_exist_constr=True,
                                               verbose=self.verbose,
                                               plot=self.plot_ts)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)


        return sym_tr, ts_curr_vars, ts_next_vars, ts_lbl_vars
    

    def _create_weight_dict(self, task) -> Dict[str, int]:
        """
         A function that loop over all the paramterized actions, like transit b0 l2, transit b1 l2, grasp b0, grasp b1 etc., and
          assigns their corresponding from the weight dictionary specified as input.
        """
        new_weight_dict = {}
        for op in task.operators:
            # extract the action name
            _generic_action = op.name.split()[0]
            _generic_action = _generic_action[1:]   # remove the intial '(' braket
            
            weight: int = self.weight_dict[_generic_action]
            new_weight_dict[op.name] = weight

        return new_weight_dict


    def build_weighted_add_abstraction(self, draw_causal_graph: bool = False) -> Tuple[SymbolicWeightedFrankaTransitionSystem, List[ADD], List[ADD], List[ADD]]:
        """
         Main Function to Build Transition System that represents valid edges with their corresponding weights
        """
        task, domain, add_ts_curr_vars, add_ts_next_vars, ts_state_tuples, add_ts_lbl_vars, boxes, possible_lbls = self.create_symbolic_causal_graph(draw_causal_graph=draw_causal_graph,
                                                                                                                                                     add_flag=True)

        # get the actual parameterized actions and add their corresponding weights
        new_weight_dict = self._create_weight_dict(task=task)

        # sort them according to their weights and then convert them in to addConst; reverse will sort the weights in descending order
        weight_dict = {k: v for k, v in sorted(new_weight_dict.items(), key=lambda item: item[1], reverse=True)}
        for action, w in weight_dict.items():
            weight_dict[action] = self.manager.addConst(int(w))
        
        sym_tr = SymbolicWeightedFrankaTransitionSystem(curr_states=add_ts_curr_vars,
                                                        next_states=add_ts_next_vars,
                                                        lbl_states=add_ts_lbl_vars,
                                                        weight_dict=weight_dict,
                                                        ts_states=ts_state_tuples,
                                                        ts_state_map=self.pred_int_map,
                                                        task=task,
                                                        domain=domain,
                                                        manager=self.manager)
        
        start: float = time.time()
        sym_tr.create_weighted_transition_system_franka(boxes=boxes,
                                                        state_lbls=possible_lbls,
                                                        add_exist_constr=True,
                                                        verbose=self.verbose,
                                                        plot=self.plot_ts)
        
        stop: float = time.time()
        print("Time took for constructing the abstraction: ", stop - start)

        return sym_tr, add_ts_curr_vars, add_ts_next_vars, add_ts_lbl_vars