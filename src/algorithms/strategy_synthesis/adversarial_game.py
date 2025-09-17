import re 
import sys
import math 
import copy
import time
import yaml
import random

from functools import reduce
from itertools import product
from subprocess import Popen, PIPE
from collections import defaultdict
from typing import List, DefaultDict, Union, Tuple, Dict
from bidict import bidict

from cudd import Cudd, BDD, ADD

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import ADDPartitionedDFA
from src.symbolic_graphs import DynWeightedPartitionedFrankaAbs
from src.symbolic_graphs.hybrid_regret_graphs import HybridGraphOfUtility, HybridGraphOfBR

from utls import *


class AdversarialGame(BaseSymbolicSearch):
    """
     A class that implements optimal strategy synthesis for the Robot player assuming Human player to be adversarial with quantitative constraints. 
    """

    def __init__(self,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 sys_act_vars: List[ADD],
                 env_act_vars: List[ADD],
                 cudd_manager: Cudd,
                 monolithic_tr: bool = False):
        super().__init__(ts_obs_vars, cudd_manager)

        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        self.ts_transition_fun_list: List[List[ADD]] = ts_handle.sym_tr_actions
        self.ts_bdd_transition_fun_list: List[List[BDD]] = []

        self.dfa_transition_fun_list: List[ADD] = dfa_handle.tr_state_adds

        # need these two during preimage computation 
        self.dfa_bdd_x_list = [i.bddPattern() for i in dfa_curr_vars]
        self.dfa_bdd_transition_fun_list: List[BDD] = [i.bddPattern() for i in self.dfa_transition_fun_list]

        self.ts_action_idx_map: bidict = ts_handle.tr_action_idx_map

        self.ts_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.ts_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_add_sym_map_curr.inv

        self.ts_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_sym_to_robot_act_map: bidict = ts_handle.predicate_sym_map_robot.inv

        self.obs_add: ADD = ts_handle.sym_state_labels
        
        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle
        
        # create corresponding cubes to avoid repetition
        self.ts_xcube: ADD = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube: ADD = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube: ADD = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])

        self.sys_cube: ADD = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube: ADD = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist: list = self.ts_x_list + self.dfa_x_list
        self.prod_xcube: ADD = reduce(lambda x, y: x & y, self.prod_xlist)

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

        # ADD that keeps track of the optimal values of state at each iteration
        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())
        
        # get the bdd version of the transition function as vectorComposition only works with
        for act in self.ts_transition_fun_list:
            act_ls = []
            for avar in act:
                act_ls.append(avar.bddPattern())
        
            self.ts_bdd_transition_fun_list.append(act_ls)
        
        # mimimum energy required from the init states
        self.init_state_value: Union[int, float] = math.inf
        
        # flag to construct monolithic TR and use differen pre-image computation code - only extracting weight changes.
        self.monolithic_tr: bool = monolithic_tr
        
        if monolithic_tr:
            self.construct_monolithic_tr()
    
    
    def construct_monolithic_tr(self):
        """
          A helper function to constructs the monolithic TR for TS and for the utility. As the utility Tr captures evolution of utls var
           under each ts action, actiosn with different edge weight should be stored in different bdd. For e.g., 

           u ---a_s---> u' and u---a_s---> u'' are possible a_s is boolean formula assocated with modified robot action
            which are unique for TS TR but not for utls TR
        """
        num_of_ts_bvars: int = len(self.ts_bdd_transition_fun_list[0])

        # get set of edge weight. convert list and use the index of the weight as the map for the monolithic TR for utls vars
        edge_weights: List[int] = list(set(self.ts_handle.int_weight_dict.values()))

        self.mono_ts_bdd_transition_fun_list: List[BDD] = [[self.manager.bddZero() for _ in range(num_of_ts_bvars)] for _ in range(len(edge_weights))]
        
        # Monolithic TS TR 
        for ts_id, tr_action in enumerate(self.ts_bdd_transition_fun_list):
            # get the edge weight its corresponding BDD idx to be stored in
            ts_act_name: str = self.ts_action_idx_map.inv[ts_id]
            action_cost: ADD =  self.ts_handle.weight_dict[ts_act_name]
            act_val: int = list(action_cost.generate_cubes())[0][1]

            mono_tr_idx = edge_weights.index(act_val)
            assert act_val != math.inf and act_val != 0, "Error constrcuting Monolithic TR for Utls Vars"
            
            for var_tr_id, var_tr_dd in enumerate(tr_action):
                self.mono_ts_bdd_transition_fun_list[mono_tr_idx][var_tr_id] |= var_tr_dd
        
        # override original bdd transition and update edge weighy 
        self.ts_bdd_transition_fun_list = self.mono_ts_bdd_transition_fun_list
        self.mono_ts_action_idx_wgt_map = {idx: wgt for idx, wgt in enumerate(edge_weights)}
    

    def _create_lbl_cubes(self) -> List[ADD]:
        """
        A helper function that create cubses of each lbl and store them in a list in the same order as the original order.
         These cubes are used when we convert a BDD to lbl state where we need to extract each lbl.
        """
        sym_lbl_xcube_list = [] 
        for vars_list in self.ts_obs_list:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        return sym_lbl_xcube_list
    

    def _get_max_tr_action_cost(self) -> int:
        """
        A helper function that retireves the highest cost amongst all the transiton function costs
        """
        _max = 0
        for tr_action in self.ts_handle.weight_dict.values():
            if not tr_action.isZero():
                action_cost = tr_action.findMax()
                action_cost_int = int(re.findall(r'\d+', action_cost.__repr__())[0])
                if action_cost_int > _max:
                    _max = action_cost_int
        
        return _max

    # overriding base class
    def get_prod_states_from_dd(self, dd_func: ADD, **kwargs) -> None:
        """
         This method overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        tmp_dd_func: BDD = dd_func.bddPattern()
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=tmp_dd_func, prod_curr_list=kwargs['prod_curr_list']) 
        for prod_cube in prod_cube_string:
            # convert the cube back to 0-1 ADD
            tmp_prod_cube: ADD = prod_cube.toADD()
            _ts_dd = tmp_prod_cube.existAbstract(self.dfa_xcube)
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(_ts_dd, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=tmp_prod_cube,
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"([{_ts_name}], {_dfa_name})")
    

    def get_state_value_from_dd(self, dd_func: ADD, prod_list: List[ADD], **kwargs) -> None:
        """
         A helper function that print the value associated with each state.
          
          @param: dd_func is wining_state ADD with minimum value at any given iteration
        """

        addVars = []
        for cube, sval in dd_func.generate_cubes():
            if sval == math.inf:
                continue
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.addVar(_idx) not in prod_list:
                    continue
                
                elif self.manager.addVar(_idx) in prod_list:
                    if var == 2:
                        _amb_var.append([self.manager.addVar(_idx), ~self.manager.addVar(_idx)])
                    elif var == 0:
                        var_list.append(~self.manager.addVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.addVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)

            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    addVars.append((reduce(lambda a, b: a & b, var_list), sval))
                    var_list = list(set(var_list) - set(_ele))
            else:
                addVars.append((reduce(lambda a, b: a & b, var_list), sval))
        
        
        for cube, val in addVars: 
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(cube.existAbstract(self.dfa_xcube), kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=cube,
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"({_ts_name}, {_dfa_name}) {val}")

    

    def get_prod_state_act_from_dd(self, dd_func: ADD, **kwargs) -> None:
        """
         This method overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        tmp_dd_func: BDD = dd_func.bddPattern()
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=tmp_dd_func, prod_curr_list=kwargs['prod_curr_list']) 
        for prod_cube in prod_cube_string:
            # convert the cube back to 0-1 ADD
            tmp_prod_cube: ADD = prod_cube.toADD()
            # get the state ADD
            _ts_dd: ADD = tmp_prod_cube.existAbstract(self.dfa_xcube & self.sys_env_cube)
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(_ts_dd, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=tmp_prod_cube,
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            # get the Robot action ADD
            ract: ADD = prod_cube.existAbstract(self.prod_xcube & self.ts_obs_cube & self.env_cube)
            ract_name = self.ts_sym_to_robot_act_map[ract]

            # rct_cubes = self.convert_prod_cube_to_func(dd_func=ract, prod_curr_list=self.sys_act_vars)

            hact: ADD = prod_cube.existAbstract(self.prod_xcube & self.ts_obs_cube & self.sys_cube)
            hact_cubes = self.convert_prod_cube_to_func(dd_func=hact, prod_curr_list=self.env_act_vars)
            for dd in hact_cubes:
                hact_name =  self.ts_sym_to_human_act_map[dd]
            
                print(f"({_ts_name}, {_dfa_name})  ----{ract_name} & {hact_name}")
    

    def get_pre_states_bdd_compose(self, ts_action: List[BDD], From: BDD, prod_curr_list=None, **kwargs) -> BDD:
        """
         This method implement the predecessor computation using BDD's compose operation
        """
        # DFA's predecessor computation - sequential compose version
        for var, bdd_func in zip(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list):
            index = self.manager.bddVariables().index(var)
            From: BDD = From.compose(bdd_func, index)
        
        dfa_win_states: BDD = From
        game_dfa_win_states: BDD = dfa_win_states
        # Game's predecessor computation using BDD's compose
        # for var, bdd_func in zip(prod_curr_list, ts_action):
        #     index = self.manager.bddVariables().index(var)
        #     # game_dfa_win_states: BDD = game_dfa_win_states.compose(bdd_func, index)
        

        # hard coding and testing them out
        # for var, bdd_func in zip(prod_curr_list, ts_action):
            # index = self.manager.bddVariables().index(var)
            # game_dfa_win_states: BDD = game_dfa_win_states.compose(bdd_func, index)
        game_dfa_win_states: BDD = game_dfa_win_states.vectorCompose(prod_curr_list[0:3], ts_action[0:3])
        game_dfa_win_states: BDD = game_dfa_win_states.vectorCompose(prod_curr_list[3:], ts_action[3:])
        
        return dfa_win_states, game_dfa_win_states


    def get_pre_states_add_compose(self, ts_action: List[BDD], From: BDD, prod_curr_list=None, **kwargs) -> ADD:
        """
         This method implement the predecessor computation using ADD's compose operation
        """
        FromADD: ADD = From.toADD()
        # DFA's predecessor computation - sequential compose version
        for var, bdd_func in zip(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list):
            index = self.manager.bddVariables().index(var)
            FromADD: ADD = FromADD.compose(bdd_func.toADD(), index)
        
        dfa_win_states: ADD = FromADD
        game_dfa_win_states: ADD = dfa_win_states
        # Game's predecessor computation using BDD's compose
        for var, bdd_func in zip(prod_curr_list, ts_action):
            # TODO: check if the index is the same when the variable is BDD and ADD. 
            index = self.manager.bddVariables().index(var)
            game_dfa_win_states: ADD = game_dfa_win_states.compose(bdd_func.toADD, index)
        
        return dfa_win_states, game_dfa_win_states


    def print_stuff(self, states: BDD, **kwargs) -> None:
        # testing_add = states
        if isinstance(states, BDD):
            states = states.toADD()
        t1 = states.existAbstract(self.env_cube)
        self.get_prod_states_from_dd(dd_func=t1.existAbstract(self.sys_cube), sym_lbl_cubes=kwargs['sym_lbl_cubes'], prod_curr_list=kwargs['prod_dfa_bdd_curr_list'])
    

    def get_pre_states(self, ts_action: List[BDD], From: BDD, prod_curr_list=None, **kwargs) -> BDD:
        """
         Compute the predecessors using the compositional approach. From is a collection of 0-1 ADD.
          As vectorCompose functionality only works for bdd, we have to first convert From to 0-1 BDD, 
        """
        # first evolve over DFA and then evolve over the TS
        mod_win_state: BDD = From.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)  
        pre_prod_state: BDD = mod_win_state.vectorCompose(prod_curr_list, ts_action) 
            
        return mod_win_state, pre_prod_state
    
    
    def evolve_as_per_human(self, curr_state_tuple: tuple, curr_dfa_state: ADD, ract_name: str, valid_human_act: str) -> ADD:
        """
         A function that compute the next state tuple given the current state tuple. 
        """
        next_exp_states = self.ts_handle.adj_map[curr_state_tuple][ract_name][valid_human_act]
        nxt_state_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)
        
        # look up its corresponding formula
        nxt_ts_state: ADD = self.ts_handle.get_sym_state_from_tuple(nxt_state_tuple)

        # update the DFA state as per the human move.
        nxt_ts_lbl = nxt_ts_state.existAbstract(self.ts_xcube)

        # create DFA edge and check if it satisfies any of the dges or not
        for dfa_state in self.dfa_sym_to_curr_state_map.keys():
            bdd_dfa_state: BDD = dfa_state.bddPattern()
            dfa_pre: BDD = bdd_dfa_state.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
            edge_exists: bool = not (dfa_pre & (curr_dfa_state.bddPattern() & nxt_ts_lbl.bddPattern())).isZero()

            if edge_exists:
                nxt_dfa_state = dfa_state
                break
            
        return nxt_ts_lbl & nxt_dfa_state, nxt_state_tuple


    def human_intervention(self,
                           ract_name:str,
                           rnext_tuple: tuple,
                           curr_state_tuple: tuple,
                           curr_dfa_state: ADD,
                           valid_human_acts: list,
                           verbose: bool = False) -> tuple:
        """
         Evolve on the game as per human intervention
        """
        # get the next action
        hnext_tuple = curr_state_tuple

        for hact in valid_human_acts:
            nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple, curr_dfa_state=curr_dfa_state, ract_name=ract_name, valid_human_act=hact)

            # forcing human to not make a move that satisfies the specification
            if (self.target_DFA & nxt_prod_state).isZero():
                if verbose:
                    print(f"Human Moved: New Conf. {self.ts_handle.get_state_from_tuple(nxt_ts_tuple)}")
                return nxt_ts_tuple

        return rnext_tuple
    

    def solve(self, verbose: bool = False, print_layers: bool = False) -> ADD:
        """
         Method that computes the optimal strategy for the system with minimum payoff under adversarial environment assumptions. 

         print_layer: set this flag to True to see Value Iteration progress
         verbose: set this flag to True to print values of states at each iteration.
        """

        ts_states: ADD = self.obs_add
        accp_states: ADD = ts_states & self.target_DFA

        # convert it to 0 - Infinity ADD
        accp_states = accp_states.ite(self.manager.addZero(), self.manager.plusInfinity())
        
        # strategy - optimal (state & robot-action) pair stored in the ADD
        strategy: ADD  = self.manager.plusInfinity()

        # initializes accepting states to be zero
        self.winning_states[0] |= self.winning_states[0].min(accp_states)
        strategy = strategy.min(accp_states)

        if verbose:
            print_layers = True

        layer: int = 0
        c_max: int = self._get_max_tr_action_cost()

        prod_curr_list = []
        prod_curr_list.extend([lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        prod_curr_list.extend(self.ts_x_list)
        
        prod_bdd_curr_list = [_avar.bddPattern() for _avar in prod_curr_list]

        prod_dfa_bdd_curr_list = prod_bdd_curr_list + self.dfa_bdd_x_list

        sym_lbl_cubes = self._create_lbl_cubes()

        while True: 
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                init_state_cube = list(((self.init_TS & self.init_DFA) & self.winning_states[layer]).generate_cubes())[0]
                init_val: int = init_state_cube[1]
                self.init_state_value = init_val
                if init_val != math.inf:
                    print(f"A Winning Strategy Exists!!. The Min Energy is {init_val}")
                    return strategy
                else:
                    print("No Winning Strategy Exists!!!")
                    # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
                    del self.winning_states
                    return
            
            if print_layers:
                print(f"**************************Layer: {layer}**************************")

            _win_state_bucket: Dict[BDD] = defaultdict(lambda: self.manager.bddZero())

            # convert the winning states into buckets of BDD
            _max_interval_val = layer * c_max
            for sval in range(_max_interval_val + 1):
                # get the states with state value equal to sval and store them in their respective bukcets
                win_sval = self.winning_states[layer].bddInterval(sval, sval)
                
                if not win_sval.isZero():
                    _win_state_bucket[sval] |= win_sval
            
            _pre_buckets: Dict[ADD] = defaultdict(lambda: self.manager.addZero())

            # compute the predecessor and store them by action cost + successor cost
            if self.monolithic_tr:
                for tr_idx, tr_action in enumerate(self.ts_bdd_transition_fun_list):
                    # we get from the new weight dictionary
                    act_val = self.mono_ts_action_idx_wgt_map[tr_idx]
                    for sval, succ_states in _win_state_bucket.items():
                        vCompose_pre_dfa_states, vCompose_pre_states = self.get_pre_states(ts_action=tr_action, From=succ_states, prod_curr_list=prod_bdd_curr_list)
                        # prod_dfa_bdd_curr_list=prod_dfa_bdd_curr_list, sym_lbl_cubes=sym_lbl_cubes)

                        # call BDD's compose operation 
                        compose_pre_dfa_states_bdd, compose_pre_states_bdd = self.get_pre_states_bdd_compose(ts_action=tr_action, From=succ_states, prod_curr_list=prod_bdd_curr_list)
                        print("************************** BDD Compose States **************************")
                        self.print_stuff(states=compose_pre_dfa_states_bdd, prod_dfa_bdd_curr_list=prod_dfa_bdd_curr_list, sym_lbl_cubes=sym_lbl_cubes)

                        print("************************** BDD Vector Compose States **************************")
                        self.print_stuff(states=vCompose_pre_dfa_states, prod_dfa_bdd_curr_list=prod_dfa_bdd_curr_list, sym_lbl_cubes=sym_lbl_cubes)
                        # # call ADD's compose operation
                        # compose_pre_dfa_states_add, compose_pre_states_add = self.get_pre_states_add_compose(ts_action=tr_action, From=succ_states, prod_curr_list=prod_bdd_curr_list)

                        # # sanity checking - BDD version
                        assert vCompose_pre_dfa_states == compose_pre_dfa_states_bdd, "[Error] Result of Vector Compose is different from BDD Compose for DFA predecessor computation." 
                        # assert vCompose_pre_dfa_state == compose_pre_dfa_states_add.bddInterval(1, 1), "[Error] Result of Vector Compose is different from ADD Compose for DFA predecessor computation."
                        
                        # assert vCompose_pre_states == compose_pre_states_bdd, "[Error] Result of Vector Compose is different from Compose for BDD version predecessor computation"
                        # assert vCompose_pre_states == compose_pre_states_add.bddInterval(1, 1), "[Error] Result of Vector Compose is different from Compose for ADD version DFA predecessor computation."

                        pre_states = vCompose_pre_states

                        if not pre_states.isZero():
                            _pre_buckets[act_val + sval] |= pre_states.toADD()

            else:
                for tr_idx, tr_action in enumerate(self.ts_bdd_transition_fun_list):
                    curr_act_name: str = self.ts_action_idx_map.inv[tr_idx]
                    action_cost: ADD =  self.ts_handle.weight_dict[curr_act_name]
                    act_val: int = list(action_cost.generate_cubes())[0][1]
                    for sval, succ_states in _win_state_bucket.items():
                        pre_states: BDD = self.get_pre_states(ts_action=tr_action, From=succ_states, prod_curr_list=prod_bdd_curr_list)

                        if not pre_states.isZero():
                            _pre_buckets[act_val + sval] |= pre_states.toADD()
            
            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, _pre_buckets.values())

            # now take univ abstraction to remove edges to states with infinity value
            upre_states: ADD = pre_states.univAbstract(self.env_cube)

            # print non-zero states in this iteration
            if verbose:
                print(f"Non-Zero states at Iteration {layer + 1}")
                self.get_prod_states_from_dd(dd_func=upre_states.existAbstract(self.sys_cube), sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_dfa_bdd_curr_list)
            
            # We need to take the unions of all the (s, a_s, a_e). But, we tmp_strategy to have background vale of inf, useful later when we take max() operation.
            # Thus, I am using this approach. 
            tmp_strategy: ADD = upre_states.ite(self.manager.addOne(), self.manager.plusInfinity())

            # accomodate for worst-case human behavior
            for sval, apre_s in _pre_buckets.items():
                tmp_strategy = tmp_strategy.max(apre_s.ite(self.manager.addConst(int(sval)), self.manager.addZero()))

            new_tmp_strategy: ADD = tmp_strategy

            # go over all the human actions and preserve the maximum one
            for human_tr_dd in self.ts_sym_to_human_act_map.keys():
                new_tmp_strategy = new_tmp_strategy.max(tmp_strategy.restrict(human_tr_dd)) 

            # compute the minimum of state action pairs
            strategy = strategy.min(new_tmp_strategy)

            self.winning_states[layer + 1] |= self.winning_states[layer]

            for tr_dd in self.ts_sym_to_robot_act_map.keys():
                # remove the dependency for that action and preserve the minimum value for every state
                self.winning_states[layer + 1] = self.winning_states[layer  + 1].min(strategy.restrict(tr_dd))

            if verbose:
                print(f"Minimum State value at Iteration {layer +1}")
                self.get_state_value_from_dd(dd_func=self.winning_states[layer + 1], sym_lbl_cubes=sym_lbl_cubes, prod_list=[*prod_curr_list, *self.dfa_x_list])
            
            # update counter 
            layer += 1


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to roll out the synthesized Min-Max strategy
        """

        curr_dfa_state = self.init_DFA
        curr_prod_state = self.init_TS & curr_dfa_state
        counter = 0
        max_layer: int = max(self.winning_states.keys())
        ract_name: str = ''

        sym_lbl_cubes = self._create_lbl_cubes()

        if verbose:
            init_ts = self.init_TS
            init_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=init_ts, sym_lbl_xcube_list=sym_lbl_cubes)
            init_ts = self.ts_handle.get_state_from_tuple(state_tuple=init_ts_tuple)
            print(f"Init State: {init_ts[1:]}")     

        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # current state tuple 
            curr_ts_state: ADD = curr_prod_state.existAbstract(self.dfa_xcube & self.sys_env_cube).bddPattern().toADD()   # to get 0-1 ADD
            curr_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)
            
            # get the state with its minimum state value
            opt_sval_cube =  list((curr_prod_state & self.winning_states[max_layer]).generate_cubes())[0]
            curr_prod_state_sval: int = opt_sval_cube[1]

            # this gives us a sval-infinity ADD
            curr_state_act_cubes: ADD =  strategy.restrict(curr_prod_state)
            # get the 0-1 version
            act_cube: ADD = curr_state_act_cubes.bddInterval(curr_prod_state_sval, curr_prod_state_sval).toADD()

            list_act_cube = self.convert_add_cube_to_func(act_cube, curr_state_list=self.sys_act_vars)

            # if multiple winning actions exisit from same state
            if len(list_act_cube) > 1:
                ract_name = None
                while ract_name is None:
                    ract_dd: List[int] = random.choice(list_act_cube)
                    ract_name = self.ts_sym_to_robot_act_map.get(ract_dd, None)

            else:
                ract_name = self.ts_sym_to_robot_act_map[act_cube]

            if verbose:
                print(f"Step {counter}: Conf: {self.ts_handle.get_state_from_tuple(curr_ts_tuple)} Act: {ract_name}")
            
            # look up the next tuple 
            next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][ract_name]['r']
            next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)

            # Human Intervention
            # flip a coin and choose to intervene or not intervene
            coin = random.randint(0, 1)
            # coin = 1
            if coin:
                # check if there any human action from the current state
                valid_acts = set(self.ts_handle.adj_map.get(curr_ts_tuple, {}).get(ract_name, {}).keys())
                human_acts = valid_acts.difference('r')
                
                if len(human_acts) > 0:
                    next_tuple = self.human_intervention(ract_name=ract_name,
                                                         curr_state_tuple=curr_ts_tuple,
                                                         rnext_tuple=next_tuple,
                                                         curr_dfa_state=curr_dfa_state,
                                                         valid_human_acts=human_acts,
                                                         verbose=verbose)

            # look up its corresponding formula
            curr_ts_state: ADD = self.ts_handle.get_sym_state_from_tuple(next_tuple)

            # convert the 0-1 ADD to BDD for DFA edge checking
            curr_ts_lbl: BDD = curr_ts_state.existAbstract(self.ts_xcube).bddPattern()

            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state in self.dfa_sym_to_curr_state_map.keys():
                bdd_dfa_state: BDD = dfa_state.bddPattern()
                dfa_pre: BDD = bdd_dfa_state.vectorCompose(self.dfa_bdd_x_list, self.dfa_bdd_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state.bddPattern() & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state: BDD = dfa_state
                    break
            
            curr_prod_state: ADD = curr_ts_state & curr_dfa_state

            counter += 1
        
        # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
        del self.winning_states


class GraphofBRAdvGame(BaseSymbolicSearch):
    """
     This class plays a Min-Max game over the graph Best Response. 
    """

    def __init__(self,
                 prod_gbr_handle: HybridGraphOfBR,
                 prod_gou_handle: HybridGraphOfUtility,
                 ts_handle: DynWeightedPartitionedFrankaAbs,
                 dfa_handle: ADDPartitionedDFA,
                 ts_curr_vars: List[ADD],
                 dfa_curr_vars: List[ADD],
                 ts_obs_vars: List[ADD],
                 prod_utls_vars: List[ADD],
                 prod_ba_vars: List[ADD],
                 sys_act_vars: List[ADD],
                 env_act_vars: List[ADD],
                 cudd_manager: Cudd,
                 monolithic_tr: bool = False):
        super().__init__(ts_obs_vars, cudd_manager)

        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.init_prod = self.init_TS & self.init_DFA & prod_gou_handle.predicate_sym_map_utls[0] & prod_gbr_handle.predicate_sym_map_ba[math.inf]

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        self.prod_utls_list = prod_utls_vars
        self.prod_ba_list = prod_ba_vars

        self.prod_trans_func_list: List[List[ADD]] = prod_gbr_handle.sym_tr_actions
        self.prod_bdd_trans_func_list: List[List[BDD]] = []


        self.ts_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        # self.ts_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_add_sym_map_curr.inv

        self.ts_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_sym_to_robot_act_map: bidict = ts_handle.predicate_sym_map_robot.inv
        
        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle
        self.prod_gou_handle = prod_gou_handle
        self.prod_gbr_handle = prod_gbr_handle
        
        # create corresponding cubes to avoid repetition
        self.ts_xcube: ADD = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube: ADD = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube: ADD = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        self.prod_utls_cube: ADD = reduce(lambda x, y: x & y, self.prod_utls_list)
        self.prod_ba_cube: ADD = reduce(lambda x, y: x & y, self.prod_ba_list)
        self.sys_cube: ADD = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube: ADD = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist: list =  self.dfa_x_list + [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list] + self.ts_x_list + self.prod_utls_list + self.prod_ba_list
        self.prod_xcube: ADD = reduce(lambda x, y: x & y, self.prod_xlist)

        # get the bdd variant
        self.prod_bdd_curr_list = [_avar.bddPattern() for _avar in self.prod_xlist]

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

        # ADD that keeps track of the optimal values of state at each iteration
        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())
        
        # get the bdd version of the transition function as vectorComposition only works with
        for act in self.prod_trans_func_list:
            act_ls = []
            for avar in act:
                act_ls.append(avar.bddPattern())
        
            self.prod_bdd_trans_func_list.append(act_ls)
        
        # mimimum energy required from the init states
        self.init_state_value: Union[int, float] = math.inf

        if monolithic_tr:
            num_of_bvars: int = len(self.prod_bdd_trans_func_list[0])
            self.mono_prod_bdd_trans_func_list: List[BDD] = [self.manager.bddZero() for _ in range(num_of_bvars)]
            
            for act in self.prod_bdd_trans_func_list:
                for var_id, var_dd in enumerate(act):
                    self.mono_prod_bdd_trans_func_list[var_id] |= var_dd
            
            # override original bdd transition 
            self.prod_bdd_trans_func_list = self.mono_prod_bdd_trans_func_list

    

    def _create_lbl_cubes(self) -> List[ADD]:
        """
        A helper function that creates cubes of each lbl and store them in a list in the same order as the original order.
         These cubes are used when we convert a BDD to lbl state where we need to extract each lbl.
        """
        sym_lbl_xcube_list = [] 
        for vars_list in self.ts_obs_list:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        return sym_lbl_xcube_list
    

    def get_state_value_from_dd(self, dd_func: ADD, state_val: int, **kwargs) -> tuple:
        """
         A helper function to print the value associated with each prod state.
        """

        addVars = []
        for cube, sval in dd_func.generate_cubes():
            if sval == math.inf:
                continue
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                if var == 2 and self.manager.addVar(_idx) not in self.prod_xlist:
                    continue
                
                elif self.manager.addVar(_idx) in self.prod_xlist:
                    if var == 2:
                        _amb_var.append([self.manager.addVar(_idx), ~self.manager.addVar(_idx)])
                    elif var == 0:
                        var_list.append(~self.manager.addVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.addVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                        sys.exit(-1)

            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    addVars.append((reduce(lambda a, b: a & b, var_list), sval))
                    var_list = list(set(var_list) - set(_ele))
            else:
                addVars.append((reduce(lambda a, b: a & b, var_list), sval))
        
        for cube, _ in addVars: 
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(cube.existAbstract(self.dfa_xcube & self.prod_utls_cube & self.prod_ba_cube), kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"
            _prod_utl: int = self.prod_gou_handle.predicate_sym_map_utls.inv[cube.existAbstract(self.ts_xcube & self.ts_obs_cube & self.dfa_xcube & self.prod_ba_cube)]
            _prod_ba: int = self.prod_gbr_handle.predicate_sym_map_ba.inv[cube.existAbstract(self.ts_xcube & self.ts_obs_cube & self.dfa_xcube & self.prod_utls_cube)]

            _dfa_name = self._look_up_dfa_name(prod_dd=cube.existAbstract(self.prod_utls_cube & self.prod_ba_cube),
                                               dfa_dict=self.dfa_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"[({_ts_name}, {_dfa_name}), {_prod_utl}, {_prod_ba}]: {state_val}")
        
        return _ts_tuple
    

    def get_sym_prod_state_from_tuple(self, prod_state_tuple: tuple) -> ADD:
        """
         A helper function that return the symbolic representation of the prod state tuple
        """

        assert len(prod_state_tuple) == 4, \
         "Encountered an invalid prod state tuple. The encoding should be of length 4: <ts-tuple>.<dfa state>.<utl val>.<ba val>. Fix this!!!"

        ts_tuple: tuple = prod_state_tuple[0]
        dfa_tuple: int = prod_state_tuple[1]
        utls_tuple: int = prod_state_tuple[2]
        br_tuple: int = prod_state_tuple[3]

        # look up its corresponding formula
        ts_state: ADD = self.ts_handle.get_sym_state_from_tuple(ts_tuple)
        dfa_sym_state: ADD = self.dfa_handle.dfa_predicate_add_sym_map_curr[dfa_tuple]
        utls_sym: ADD = self.prod_gou_handle.predicate_sym_map_utls[utls_tuple]
        ba_sym: ADD = self.prod_gbr_handle.predicate_sym_map_ba[br_tuple]
        
        prod_state_sym = ts_state & dfa_sym_state & utls_sym & ba_sym

        assert not (prod_state_sym).isZero(), "Error constructing symbolic prod repr from prod state tuple. Fix This!!!"

        return prod_state_sym
    

    def get_reg_val_prod_state(self, max_layer: int, prod_state: ADD) -> int:
        """
         A helper function that computes the Reg valuea associated with a state
        """
        if prod_state & self.winning_states[max_layer] == self.manager.addZero():
            # states can optimal value zero regret
            curr_prod_state_sval = 0
        else:
            opt_sval_cube =  list((prod_state & self.winning_states[max_layer]).generate_cubes())[0]
            curr_prod_state_sval: int = opt_sval_cube[1]
        
        return curr_prod_state_sval


    def get_strategy_and_val(self, strategy: ADD, max_layer: int, curr_prod_state: ADD, print_all_act_vals: bool = False, curr_prod_tuple: tuple = ()) -> Tuple[ADD, int]:
        """
         A helper function, that given the current state as 0-1 ADD computes the action to take and the reg value assocated with the current state.
        """

        if print_all_act_vals:
            valid_acts = set(self.prod_gbr_handle.prod_adj_map[curr_prod_tuple].keys())
            # loop through all valid actions and print the reg value associated with each action
            ract_cube_list = []
            ract_val_list = []
            for ridx, ract in enumerate(valid_acts):
                # get the sym repre
                _act_dd = self.ts_sym_to_robot_act_map.inv[ract]
                _act_cube: ADD = strategy.restrict(curr_prod_state & _act_dd)
                if _act_cube == self.manager.addZero():
                    print(f"[{ridx}] {ract}: reg val: {0}")
                    ract_val_list.append(0)
                else:
                    _act_cube_string = list(_act_cube.generate_cubes())
                    assert len(_act_cube_string) == 1, "Error computing act cubes during rollout. Fix This!!!"
                    print(f"[{ridx}] {ract}: reg val: {_act_cube_string[0][1]}")
                    ract_val_list.append(_act_cube_string[0][1])
                
                ract_cube_list.append(_act_dd)
                
            
            # if we want to manully play the game
            nxt_act_idx = int(input("Enter Next action id: "))
            # ract_name: str = ract_cube_list[nxt_act_idx]
            return ract_cube_list[nxt_act_idx], ract_val_list[nxt_act_idx]


        # get the state with its minimum state value
        if curr_prod_state & self.winning_states[max_layer] == self.manager.addZero():
            # states can optimal value zero regreyt
            curr_prod_state_sval = 0
            act_cube: ADD = strategy.restrict(curr_prod_state).bddInterval(0, 0).toADD()
        else:
            opt_sval_cube =  list((curr_prod_state & self.winning_states[max_layer]).generate_cubes())[0]
            curr_prod_state_sval: int = opt_sval_cube[1]

            # this gives us a sval-infinity ADD
            curr_state_act_cubes: ADD =  strategy.restrict(curr_prod_state)

            # get the 0-1 version
            act_cube: ADD = curr_state_act_cubes.bddInterval(curr_prod_state_sval, curr_prod_state_sval).toADD()
        

        return act_cube, curr_prod_state_sval
    

    def get_random_hact(self, curr_prod_tuple: tuple, curr_prod_sym: ADD, ract_name: str, sym_lbl_cubes: List[ADD], curr_prod_state_sval: int, verbose: bool = False) -> ADD:
        """
         A helper function that will randomly choose human interventions by tossing a coin.
        """
        assert len(curr_prod_tuple) == 4, \
         "Encountered an invalid prod state tuple. The encoding should be of length 4: <ts-tuple>.<dfa state>.<utl val>.<ba val>. Fix this!!!"

        curr_ts_tuple = curr_prod_tuple[0]
        curr_dfa_tuple = curr_prod_tuple[1]
        curr_prod_utl = curr_prod_tuple[2]
        curr_prod_ba = curr_prod_tuple[3]

        valid_acts = set(self.prod_gbr_handle.prod_adj_map.get((curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba), {}).get(ract_name, {}).keys())
        human_acts = valid_acts.difference('r')

        if len(human_acts) > 0:
            next_prod_sym, hact_name = self.human_intervention(ract_name=ract_name,
                                                                curr_prod_tuple=(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba),
                                                                valid_human_acts=human_acts)
            print("Human Moved: ")
            self.get_state_value_from_dd(dd_func=next_prod_sym, sym_lbl_cubes=sym_lbl_cubes, state_val='')
            print(f"Act: {hact_name}")
        
            return next_prod_sym
        
        return curr_prod_sym

    
    def get_hact_from_user(self, curr_prod_tuple: tuple, curr_prod_sym: ADD, ract_name: str,  sym_lbl_cubes: List[ADD], strategy: ADD, max_layer: int):
        """
         A helper function that asks the user for next human action.
        """
        # print all the possible actions
        assert len(curr_prod_tuple) == 4, \
         "Encountered an invalid prod state tuple. The encoding should be of length 4: <ts-tuple>.<dfa state>.<utl val>.<ba val>. Fix this!!!"

        curr_ts_tuple = curr_prod_tuple[0]
        curr_dfa_tuple = curr_prod_tuple[1]
        curr_prod_utl = curr_prod_tuple[2]
        curr_prod_ba = curr_prod_tuple[3]

        valid_acts = set(self.prod_gbr_handle.prod_adj_map.get((curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba), {}).get(ract_name, {}).keys())
        human_acts = valid_acts.difference('r')

        if len(human_acts) > 0:
            sym_prod_list = []
            hact_list = []
            print("===================================================================================")
            # for hidx, hact in enumerate(human_acts):
            for hidx, hact in enumerate(valid_acts):
                nxt_prod_tuple: tuple = self.prod_gbr_handle.prod_adj_map[curr_prod_tuple][ract_name][hact]
                nxt_prod_sym: ADD = self.get_sym_prod_state_from_tuple(nxt_prod_tuple)
                # _ , reg_val = self.get_strategy_and_val(strategy=strategy, max_layer=max_layer, curr_prod_state=nxt_prod_sym)
                print(f"[{hidx}] {hact}")
                # self.get_state_value_from_dd(dd_func=nxt_prod_sym, sym_lbl_cubes=sym_lbl_cubes, state_val=reg_val)
                hact_list.append(hact)
                sym_prod_list.append(nxt_prod_sym)
            print("===================================================================================")
            # ask for user input 
            nxt_state_idx = int(input("Enter Next state id: "))

            while nxt_state_idx > len(sym_prod_list):
                print("Please enter a valid index")
                nxt_state_idx = int(input("Enter Next state id: "))

            return sym_prod_list[nxt_state_idx]
        
        return curr_prod_sym


    def human_intervention(self,
                           ract_name: str,
                           curr_prod_tuple: tuple,
                           valid_human_acts: list) -> Tuple[ADD, str]:
        """
         Evolve on the game as per human intervention.

         Return the 0-1 ADD repr of the next state, and the human action string
        """

        for hact in valid_human_acts:
            nxt_prod_tuple: tuple = self.prod_gbr_handle.prod_adj_map[curr_prod_tuple][ract_name][hact]
            nxt_prod_sym: ADD = self.get_sym_prod_state_from_tuple(nxt_prod_tuple)

            return nxt_prod_sym, hact
    

    def get_pre_states(self, From: BDD) -> BDD:
        """
         Compute the pre-image on the product graph
        """
        pre_prod_state: BDD = self.manager.bddZero()
        
        # Monolithic TR
        if isinstance(self.prod_bdd_trans_func_list[0], BDD):
            pre_prod_state |= From.vectorCompose(self.prod_bdd_curr_list, self.prod_bdd_trans_func_list)
        
        # Partitioned TR 
        else:
            for ts_transition in self.prod_bdd_trans_func_list:
                pre_prod_state |= From.vectorCompose(self.prod_bdd_curr_list, [*ts_transition])
            
        return pre_prod_state
    

    def solve(self, verbose: bool = False, print_layers: bool = False) -> ADD:
        """
         Method to compute optimal strategies for the robot assuming human to Adversarial for given graph of best response
        """
        accp_states = self.prod_gbr_handle.leaf_nodes

        # strategy - optimal (state & robot-action) pair stored in the ADD
        strategy: ADD  = self.manager.plusInfinity()

        # initializes accepting states to be zero
        self.winning_states[0] |= self.winning_states[0].min(accp_states)
        strategy = strategy.min(accp_states)

        if verbose:
            print_layers = True

        layer: int = 0

        sym_lbl_cubes = self._create_lbl_cubes()

        while True:
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                if self.init_prod & self.winning_states[layer] == self.manager.addZero():
                    self.init_state_value = 0
                    print(f"A Winning Strategy Exists!!. The Min Regret value is {self.init_state_value}")
                    return strategy
                
                init_state_cube = list((self.init_prod & self.winning_states[layer]).generate_cubes())[0]
                init_val: int = init_state_cube[1]
                self.init_state_value = init_val
                if init_val != math.inf:
                    print(f"A Winning Strategy Exists!!. The Min Regret value is {init_val}")
                    return strategy
                else:
                    print("No Winning Strategy Exists!!!")
                    # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
                    del self.winning_states
                    return
            
            if print_layers:
                print(f"**************************Layer: {layer}**************************")

            _win_state_bucket: Dict[BDD] = defaultdict(lambda: self.manager.bddZero())

            # convert the winning states into buckets of BDD
            for sval in self.prod_gbr_handle.leaf_vals:
                # get the states with state value equal to sval and store them in their respective bukcets
                win_sval = self.winning_states[layer].bddInterval(sval, sval)

                if not win_sval.isZero():
                    _win_state_bucket[sval] |= win_sval
            
            
            _pre_buckets: Dict[ADD] = defaultdict(lambda: self.manager.addZero())

            # compute the predecessor and store them by successor cost
            for sval, succ_states in _win_state_bucket.items():
                pre_states: BDD = self.get_pre_states(From=succ_states)

                if not pre_states.isZero():
                    _pre_buckets[sval] |= pre_states.toADD()
            
            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, _pre_buckets.values())

            # now take univ abstraction to remove edges to states with infinity value
            upre_states: ADD = pre_states.univAbstract(self.env_cube)

            tmp_strategy: ADD = upre_states.ite(self.manager.addZero(), self.manager.plusInfinity())

            for sval, apre_s in _pre_buckets.items():
                # we skip the zero states
                if sval != 0:
                    tmp_strategy = tmp_strategy.max(apre_s.ite(self.manager.addConst(int(sval)), self.manager.addZero()))

            new_tmp_strategy: ADD = tmp_strategy

            # go over all the human actions and preserve the minimum one
            for human_tr_dd in self.ts_sym_to_human_act_map.keys():
                new_tmp_strategy = new_tmp_strategy.max(tmp_strategy.restrict(human_tr_dd)) 

            # compute the minimum of state action pairs
            strategy = strategy.min(new_tmp_strategy)

            self.winning_states[layer + 1] |= self.winning_states[layer]

            for tr_dd in self.ts_sym_to_robot_act_map.keys():
                # remove the dependency for that action and preserve the minimum value for every state
                self.winning_states[layer + 1] = self.winning_states[layer  + 1].min(strategy.restrict(tr_dd))
            
            if verbose:
                print(f"Minimum State value at Iteration {layer + 1}")
                # we know the set of values that can possible prpagate value. So we only print states with those values
                for lval in self.prod_gbr_handle.leaf_vals:
                    dd_func = self.winning_states[layer + 1].bddInterval(lval, lval).toADD()
                    self.get_state_value_from_dd(dd_func=dd_func, sym_lbl_cubes=sym_lbl_cubes, state_val=lval)
                    print("===================================================================================")
                print("********************************************************************************************")

            # update counter 
            layer += 1
    

    def roll_out_strategy(self, strategy: ADD, ask_usr_input: bool = False, verbose: bool = False):
        """
         A helper function to roll out the optimal regret minimizing strategy.
        """
        counter = 0
        max_layer: int = max(self.winning_states.keys())
        ract_name: str = ''

        sym_lbl_cubes = self._create_lbl_cubes()

        curr_prod_state = self.init_prod

        if verbose:
            init_ts = self.init_TS
            init_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=init_ts, sym_lbl_xcube_list=sym_lbl_cubes)
            init_ts = self.ts_handle.get_state_from_tuple(state_tuple=init_ts_tuple)
            print(f"Init State: {init_ts[1:]}")
        
        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # current state tuple 
            curr_ts_state: ADD = curr_prod_state.existAbstract(self.dfa_xcube & self.sys_env_cube & self.prod_utls_cube & self.prod_ba_cube).bddPattern().toADD()   # to get 0-1 ADD
            curr_dfa_state: ADD = curr_prod_state.existAbstract(self.ts_xcube & self.ts_obs_cube & self.sys_env_cube & self.prod_utls_cube & self.prod_ba_cube ).bddPattern().toADD()
            curr_dfa_tuple: int = self.dfa_sym_to_curr_state_map[curr_dfa_state]
            curr_ts_tuple: tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)
            curr_prod_utl: int = self.prod_gou_handle.predicate_sym_map_utls.inv[curr_prod_state.existAbstract(self.ts_xcube & self.ts_obs_cube & self.dfa_xcube & self.prod_ba_cube)]
            curr_prod_ba: int = self.prod_gbr_handle.predicate_sym_map_ba.inv[curr_prod_state.existAbstract(self.ts_xcube & self.ts_obs_cube & self.dfa_xcube & self.prod_utls_cube)]
            
            # get the action. . .
            act_cube, curr_prod_state_sval = self.get_strategy_and_val(strategy=strategy,
                                                                       max_layer=max_layer,
                                                                       curr_prod_state=curr_prod_state,
                                                                       curr_prod_tuple=(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba),
                                                                       print_all_act_vals=ask_usr_input)

            list_act_cube = self.convert_add_cube_to_func(act_cube, curr_state_list=self.sys_act_vars)

            # if multiple winning actions exists from same state
            if len(list_act_cube) > 1:
                ract_list = []
                for ridx, ract_dd in enumerate(list_act_cube):
                    ract_name = self.ts_sym_to_robot_act_map[ract_dd]
                    next_prod_tuple = self.prod_gbr_handle.prod_adj_map[(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba)][ract_name]['r']
                    next_prod_sym = self.get_sym_prod_state_from_tuple(next_prod_tuple)
                    # next_reg_val = self.get_reg_val_prod_state(max_layer=max_layer, prod_state=next_prod_sym) 
                    print(f"{ridx}: {ract_name}")
                    # self.get_state_value_from_dd(dd_func=next_prod_sym, sym_lbl_cubes=sym_lbl_cubes, state_val=next_reg_val)
                    ract_list.append(ract_name)
            
                if ask_usr_input:
                    nxt_act_idx = int(input("Enter Next action id: "))
                    ract_name: str = ract_list[nxt_act_idx]
                else:
                    ract_name: str = random.choice(ract_list)
                
            else:
                ract_name: str = self.ts_sym_to_robot_act_map[act_cube]

            if verbose:
                print(f"Step {counter}")
                ts_tuple = self.get_state_value_from_dd(dd_func=curr_prod_state, sym_lbl_cubes=sym_lbl_cubes, state_val=curr_prod_state_sval)
                print(f"Act: {ract_name}")

            # look up the next tuple 
            next_prod_tuple = self.prod_gbr_handle.prod_adj_map[(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba)][ract_name]['r']
            next_prod_sym = self.get_sym_prod_state_from_tuple(next_prod_tuple)
            
            if not ask_usr_input:
                coin = random.randint(0, 1)
                # coin = 1

                if coin:
                    next_prod_sym = self.get_random_hact(curr_prod_tuple=(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba),
                                                         curr_prod_sym=next_prod_sym,
                                                         ract_name=ract_name,
                                                         sym_lbl_cubes=sym_lbl_cubes,
                                                         curr_prod_state_sval=curr_prod_state_sval,
                                                         verbose=verbose)
            else:
                next_prod_sym = self.get_hact_from_user(curr_prod_tuple=(curr_ts_tuple, curr_dfa_tuple, curr_prod_utl, curr_prod_ba),
                                                        curr_prod_sym=next_prod_sym,
                                                        sym_lbl_cubes=sym_lbl_cubes,
                                                        ract_name=ract_name,
                                                        strategy=strategy,
                                                        max_layer=max_layer)

            curr_prod_state = next_prod_sym
            
            counter += 1

        # need to delete this dict that holds cudd object to avoid segfaults after exiting python code
        del self.winning_states
