import re 
import sys
import copy
import random

from functools import reduce
from collections import defaultdict
from typing import List, DefaultDict, Union, Tuple
from bidict import bidict

from cudd import Cudd, BDD

from src.algorithms.base import BaseSymbolicSearch
from src.symbolic_graphs import PartitionedDFA
from src.symbolic_graphs import DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem

from utls import *


class ReachabilityGame(BaseSymbolicSearch):
    """
     A class that implements a Reachability algorithm to compute a set of winning region and the
     corresponding winning strategy for both the players in a Game G = (S, E). A winning strategy induces a play for the
     system player that satisfies the winning condition, i.e., to reach the accepting states on a symbolic given game.
    
    
     For a Reachability game the goal in terms of LTL could be written as : F(Accn States), i.e., Eventually
     reach the accepting region. A strategy for the sys ensures that it can force a visit to the accn states
     while the strategy for the env player is to remain in the trap region.
    """

    def __init__(self,
                 ts_handle: Union[DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem],
                 dfa_handle: PartitionedDFA,
                 ts_curr_vars: List[BDD],
                 dfa_curr_vars: List[BDD],
                 ts_obs_vars: List[BDD],
                 sys_act_vars: List[BDD],
                 env_act_vars: List[BDD],
                 cudd_manager: Cudd):
        super().__init__(ts_obs_vars, cudd_manager)
        self.init_TS = ts_handle.sym_init_states
        self.target_DFA = dfa_handle.sym_goal_state
        self.init_DFA = dfa_handle.sym_init_state

        self.ts_x_list = ts_curr_vars
        self.dfa_x_list = dfa_curr_vars

        self.sys_act_vars = sys_act_vars
        self.env_act_vars = env_act_vars

        # self.ts_transition_fun_list: List[BDD] = ts_handle.tr_state_bdds
        self.ts_transition_fun_list: List[List[BDD]] = ts_handle.sym_tr_actions
        self.dfa_transition_fun_list: List[BDD] = dfa_handle.tr_state_bdds

        self.ts_bdd_sym_to_curr_state_map: bidict = ts_handle.predicate_sym_map_curr.inv
        self.ts_bdd_sym_to_S2obs_map: bidict = ts_handle.predicate_sym_map_lbl.inv
        self.dfa_bdd_sym_to_curr_state_map: bidict = dfa_handle.dfa_predicate_sym_map_curr.inv

        self.ts_bdd_sym_to_human_act_map: bidict = ts_handle.predicate_sym_map_human.inv
        self.ts_bdd_sym_to_robot_act_map: bidict = ts_handle.predicate_sym_map_robot.inv

        self.obs_bdd: BDD = ts_handle.sym_state_labels

        self.ts_handle = ts_handle
        self.dfa_handle = dfa_handle

        self.winning_states: DefaultDict[int, BDD] = defaultdict(lambda: self.manager.bddZero())

        # create corresponding cubes to avoid repetition
        self.ts_xcube: BDD = reduce(lambda x, y: x & y, self.ts_x_list)
        self.dfa_xcube: BDD = reduce(lambda x, y: x & y, self.dfa_x_list)
        self.ts_obs_cube: BDD = reduce(lambda x, y: x & y, [lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])

        self.sys_cube: BDD = reduce(lambda x, y: x & y, self.sys_act_vars)
        self.env_cube: BDD = reduce(lambda x, y: x & y, self.env_act_vars)

        # create ts and dfa combines cube
        self.prod_xlist: list = self.ts_x_list + self.dfa_x_list
        self.prod_xcube: BDD = reduce(lambda x, y: x & y, self.prod_xlist)

        # sys and env cube
        self.sys_env_cube = reduce(lambda x, y: x & y, [*self.sys_act_vars, *self.env_act_vars])

        self.stra_list = defaultdict(lambda: self.manager.bddZero())

        # for bounded Game abstraction with addition boolean vars for # remaining human interventions
        if isinstance(self.ts_handle, BndDynamicFrankaTransitionSystem):
            self.max_hint: int = self.ts_handle.max_hint - 1
            self.hint_cube: BDD = reduce(lambda x, y: x & y, self.ts_handle.sym_vars_hint)
            self.hint_list = self.ts_handle.sym_vars_hint
            self.ts_bdd_sym_to_hint_map: bidict = self.ts_handle.predicate_sym_map_hint.inv

        self._initialize_w_t()
    
    def _initialize_w_t(self) -> None:
        """
         W : Set of winning states 
         T : Set of winning states and actions (winning strategies)
        
        This function initializes the set of winning states and set of winning states to the set of accepting states
        """
        ts_states: BDD = self.obs_bdd
        accp_states: BDD = ts_states & self.target_DFA

        self.winning_states[0] |= accp_states
    

    def _create_lbl_cubes(self):
        """
        A helper function that create cubses of each lbl and store them in a list in the same order as the original order.
         These cubes are used when we convert a BDD to lbl state where we need to extract each lbl.
        """
        sym_lbl_xcube_list = [] 
        for vars_list in self.ts_obs_list:
            sym_lbl_xcube_list.append(reduce(lambda x, y: x & y, vars_list))
        
        return sym_lbl_xcube_list


    # overriding base class
    def get_prod_states_from_dd(self, dd_func: BDD, obs_flag: bool = False, **kwargs) -> None:
        """
         This method overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        prod_curr_list = kwargs['prod_curr_list']
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=dd_func, prod_curr_list=prod_curr_list) 
        for prod_cube in prod_cube_string:
            _ts_dd = prod_cube.existAbstract(self.dfa_xcube)
            _ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(_ts_dd, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube,
                                               dfa_dict=self.dfa_bdd_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"([{_ts_name}], {_dfa_name})")
    

    def evolve_as_per_human(self, curr_state_tuple: tuple, curr_dfa_state: BDD, ract_name: str) -> BDD:
        """
         A function that compute the next state tuple given the current state tuple. 
        """
        next_exp_states = random.choice(self.ts_handle.adj_map[curr_state_tuple][ract_name]['h'])
        nxt_state_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)
        
        # look up its corresponding formula
        nxt_ts_state: BDD = self.ts_handle.get_sym_state_from_tuple(nxt_state_tuple)

        # update the DFA state as per the human move.
        nxt_ts_lbl = nxt_ts_state.existAbstract(self.ts_xcube)

        # create DFA edge and check if it satisfies any of the dges or not
        for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
            dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
            edge_exists: bool = not (dfa_pre & (curr_dfa_state & nxt_ts_lbl)).isZero()

            if edge_exists:
                nxt_dfa_state = dfa_state
                break
            
        return nxt_ts_lbl & nxt_dfa_state, nxt_state_tuple


    def human_intervention(self,
                           ract_name:str,
                           curr_state_tuple: tuple,
                           curr_dfa_state: BDD,
                           verbose: bool = False) -> tuple:
        """
         Evolve on the game as per human intervention
        """
        # get the next action
        hnext_tuple = curr_state_tuple

        itr = 0
        nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple, curr_dfa_state=curr_dfa_state, ract_name=ract_name)
        
        # forcing human to not make a move that satisfies the specification
        while not (self.target_DFA & nxt_prod_state).isZero():
            nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple, curr_dfa_state=curr_dfa_state, ract_name=ract_name)
            # hacky way to avoid infinite looping
            itr += 1
            if itr > 5:
                break

        if verbose:
            print(f"Human Moved: New Conf. {self.ts_handle.get_state_from_tuple(nxt_ts_tuple)}")

        return nxt_ts_tuple


    def roll_out_strategy(self, transducer: BDD, verbose: bool = False) -> None:
        """
         A function to roll out the synthesize winning strategy
        """

        curr_dfa_state = self.init_DFA
        curr_prod_state = self.init_TS & curr_dfa_state
        counter = 0
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
            curr_ts_state: BDD = curr_prod_state.existAbstract(self.dfa_xcube & self.sys_env_cube)
            curr_ts_tuple = self.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)
            
            curr_state_act: BDD =  transducer & curr_prod_state
            curr_act: BDD = curr_state_act.existAbstract(self.prod_xcube & self.ts_obs_cube)

            curr_act_cubes = self.convert_prod_cube_to_func(dd_func=curr_act, prod_curr_list=self.sys_act_vars)

            # if multiple winning actions exisit from same state
            if len(curr_act_cubes) > 1:
                ract_name = None
                while ract_name is None:
                    ract_dd: List[int] = random.choice(curr_act_cubes)
                    ract_name = self.ts_bdd_sym_to_robot_act_map.get(ract_dd, None)

            else:
                ract_name = self.ts_bdd_sym_to_robot_act_map[curr_act]

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
                if self.ts_handle.adj_map.get(curr_ts_tuple, {}).get(ract_name, {}).get('h'):
                    next_tuple = self.human_intervention(ract_name=ract_name,
                                                         curr_state_tuple=curr_ts_tuple,
                                                         curr_dfa_state=curr_dfa_state,
                                                         verbose=verbose)
            
            
            
            # look up its corresponding formula
            curr_ts_state: BDD = self.ts_handle.get_sym_state_from_tuple(next_tuple)

            curr_ts_lbl: BDD = curr_ts_state.existAbstract(self.ts_xcube)

            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
                dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state = dfa_state
                    break 

           
            curr_prod_state = curr_ts_state & curr_dfa_state

            counter += 1


    def get_strategy(self, transducer: BDD, verbose: bool = False) -> BDD:
        """
         A function that compute the action to take from each state from the transducer. 
        """
        strategy: BDD = transducer.solveEqn(self.sys_act_vars)
        state_action = strategy[0].vectorCompose(self.sys_act_vars, strategy[1])

        # restrict the strategy to non accpeting state, as we don not care what the Roboto after it reaches an accepting state
        if verbose:
            self.get_state_action(state_action & self.init_DFA)
        
        return state_action

    
    def get_pre_states(self, layer: int, prod_curr_list=None) -> BDD:
        """
         We  have an additional human intervention variables that we need to account 
        """
        pre_prod_state: BDD = self.manager.bddZero()

        # first evolve over DFA and then evolve over the TS
        mod_win_state: BDD = self.winning_states[layer].vectorCompose(self.dfa_x_list, [*self.dfa_transition_fun_list])
        
        for ts_transition in self.ts_transition_fun_list:
            pre_prod_state |= mod_win_state.vectorCompose(prod_curr_list,[*ts_transition])
        
        return pre_prod_state
    

    def solve(self, verbose: bool = False) -> BDD:
        """
         This function compute the set of winning states and winnign strategies. 
        """        
        closed = self.manager.bddZero()  # BDD to keep track of winning states explore till now

        layer: int = 0

        self.stra_list[layer] = self.winning_states[layer]
        if verbose:
            closed |= self.winning_states[layer]
        
        # prod_curr_list = [ele for ele in self.dfa_x_list]
        prod_curr_list = []
        prod_curr_list.extend([lbl for sym_vars_list in self.ts_obs_list for lbl in sym_vars_list])
        if isinstance(self.ts_handle, BndDynamicFrankaTransitionSystem):
            prod_curr_list.extend([*self.ts_x_list, *self.ts_handle.sym_vars_hint])
        else:
            prod_curr_list.extend([*self.ts_x_list])

        sym_lbl_cubes = self._create_lbl_cubes()

        while True:
            if layer > 0 and self.stra_list[layer].compare(self.stra_list[layer - 1], 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                if not ((self.init_TS & self.init_DFA) & self.stra_list[layer]).isZero():
                    print("A Winning Strategy Exists!!")

                    return self.stra_list[layer]
                else:
                    print("No Winning Strategy Exists!!!")
                    return


            print(f"**************************Layer: {layer}**************************")
            
            pre_prod_state = self.get_pre_states(layer=layer, prod_curr_list=prod_curr_list)

            # do universal quantification
            pre_univ = (pre_prod_state).univAbstract(self.env_cube)

            # remove self loops
            self.stra_list[layer + 1] |= self.stra_list[layer] | (~self.winning_states[layer] & pre_univ)
            
            # if init state is reached
            if not ((self.init_TS & self.init_DFA) & self.stra_list[layer + 1]).isZero():
                print(f"************************** Reached the Initial State at layer: {layer + 1} **************************")
                print("A Winning Strategy Exists!!")

                # return winning_str
                if verbose:
                    print(f"Winning states at Iter {layer + 1}")
                    new_states = ~closed & pre_univ.existAbstract(self.sys_env_cube)
                    self.get_prod_states_from_dd(dd_func=new_states, sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_curr_list)

                    closed |= pre_univ.existAbstract(self.sys_env_cube)# & self.ts_obs_cube)
                return self.stra_list[layer + 1]

            # do existentail quantification
            self.winning_states[layer + 1] |=  self.stra_list[layer + 1].existAbstract(self.sys_cube)

            # print new winning states in each iteration
            if verbose:
                print(f"Winning states at Iter {layer + 1}")
                new_states = ~closed & pre_univ.existAbstract(self.sys_env_cube)
                self.get_prod_states_from_dd(dd_func=new_states, sym_lbl_cubes=sym_lbl_cubes, prod_curr_list=prod_curr_list)

                closed |= pre_univ.existAbstract(self.sys_env_cube)

            layer +=1


class BndReachabilityGame(ReachabilityGame):
    """
     Compute the Winning strategy for Game Abstraction with bounded human intervention encounded as counter to each Node. 
    """

    def __init__(self,
                 ts_handle: Union[DynamicFrankaTransitionSystem, BndDynamicFrankaTransitionSystem],
                 dfa_handle: PartitionedDFA,
                 ts_curr_vars: List[BDD],
                 dfa_curr_vars: List[BDD],
                 ts_obs_vars: List[BDD],
                 sys_act_vars: List[BDD],
                 env_act_vars: List[BDD],
                 cudd_manager: Cudd):
        super().__init__(ts_handle,
                         dfa_handle,
                         ts_curr_vars,
                         dfa_curr_vars,
                         ts_obs_vars,
                         sys_act_vars,
                         env_act_vars,
                         cudd_manager)


    def get_prod_states_from_dd(self, dd_func: BDD, obs_flag: bool = False, **kwargs) -> None:
        """
         This base class overrides the base method by return the Actual state name using the
          pred int map dictionary rather than the state tuple. 
        """
        prod_curr_list = kwargs['prod_curr_list']
        prod_cube_string: List[BDD] = self.convert_prod_cube_to_func(dd_func=dd_func, prod_curr_list=prod_curr_list) 
        for prod_cube in prod_cube_string:
            _ts_dd = prod_cube.existAbstract(self.dfa_xcube)
            _ts_tuple, _ts_hint = self.ts_handle.get_state_tuple_from_sym_state(_ts_dd, kwargs['sym_lbl_cubes'])
            _ts_name = self.ts_handle.get_state_from_tuple(_ts_tuple)
            assert _ts_name is not None, "Couldn't convert TS Cube to its corresponding State. FIX THIS!!!"

            _dfa_name = self._look_up_dfa_name(prod_dd=prod_cube.existAbstract(self.hint_cube),
                                               dfa_dict=self.dfa_bdd_sym_to_curr_state_map,
                                               ADD_flag=False,
                                               **kwargs)
            
            print(f"([{_ts_name},{_ts_hint}], {_dfa_name})")


    def evolve_as_per_human(self, curr_state_tuple: tuple, curr_dfa_state: BDD, curr_hint: int) -> Tuple[BDD, tuple]:
        """
         A function that compute the next state tuple given the current state tuple. 
        """
        next_exp_states = random.choice(self.ts_handle.adj_map[curr_state_tuple][curr_hint]['h'])
        nxt_state_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)
        
        # look up its corresponding formula
        nxt_ts_state: BDD = self.ts_handle.get_sym_state_from_tuple(nxt_state_tuple)

        # update the DFA state as per the human move.
        nxt_ts_lbl = nxt_ts_state.existAbstract(self.ts_xcube)

        # create DFA edge and check if it satisfies any of the dges or not
        for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
            dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
            edge_exists: bool = not (dfa_pre & (curr_dfa_state & nxt_ts_lbl)).isZero()

            if edge_exists:
                nxt_dfa_state = dfa_state
                break
            
        return nxt_ts_lbl & nxt_dfa_state, nxt_state_tuple
    

    def human_intervention(self,
                           curr_state_tuple: tuple,
                           curr_dfa_state: BDD,
                           curr_hint: int,
                           verbose: bool = False) -> Tuple[tuple, int]:
        """
         Evolve on the game as per human intervention
        """
        # get the next action
        hnext_tuple = curr_state_tuple

        itr = 0
        nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple,
                                                                curr_hint=curr_hint,
                                                                curr_dfa_state=curr_dfa_state)
        
        # forcing human to not make a move that satisfies the specification
        while not (self.target_DFA & nxt_prod_state).isZero():
            nxt_prod_state, nxt_ts_tuple = self.evolve_as_per_human(curr_state_tuple=hnext_tuple,
                                                                    curr_hint=curr_hint,
                                                                    curr_dfa_state=curr_dfa_state)
            # hacky way to avoid infinite looping
            itr += 1
            if itr > 5:
                break

        if verbose:
            diff = list(set(nxt_ts_tuple) - set(curr_state_tuple))
            assert len(diff) == 1, "Error with Human Intevention. More than one block moved in one Human move. Fix This!!!" 
            print(f"Human Moved: {self.ts_handle.get_state_from_tuple(diff)}; New Conf. {self.ts_handle.get_state_from_tuple(nxt_ts_tuple)}")

        # reduce the human int count by 1
        curr_hint -= 1
        curr_state_tuple = nxt_ts_tuple

        return curr_state_tuple, curr_hint


    def roll_out_strategy(self,
                          transducer: BDD,
                          verbose: bool = False) -> None:
        """
         A function to roll out the synthesized winning strategy
        """

        curr_dfa_state = self.init_DFA
        curr_prod_state = self.init_TS & curr_dfa_state
        curr_hint_dd: BDD = self.init_TS.existAbstract(self.ts_xcube & self.ts_obs_cube)
        curr_hint: int = self.ts_bdd_sym_to_hint_map[curr_hint_dd]
        counter = 0
        ract_name: str = ''

        sym_lbl_cubes = self._create_lbl_cubes()
        
        if verbose:
            init_ts = self.init_TS & curr_hint_dd
            init_ts_tuple, _  = self.ts_handle.get_state_tuple_from_sym_state(sym_state=init_ts, sym_lbl_xcube_list=sym_lbl_cubes)
            init_ts = self.ts_handle.get_state_from_tuple(state_tuple=init_ts_tuple)
            print(f"Init State: {init_ts[1:]}")

        # until you reach a goal state. . .
        while (self.target_DFA & curr_prod_state).isZero():
            # current state tuple
            curr_ts_state: BDD = curr_prod_state.existAbstract(self.dfa_xcube & self.sys_env_cube)
            curr_ts_tuple, _ = self.ts_handle.get_state_tuple_from_sym_state(sym_state=curr_ts_state, sym_lbl_xcube_list=sym_lbl_cubes)

            curr_state_act: BDD =  transducer & curr_prod_state
            curr_act: BDD = curr_state_act.existAbstract(self.prod_xcube & self.ts_obs_cube & self.hint_cube)

            # flip a coin and choose to intervene or not intervene
            coin = random.randint(0, 1)
            # coin = 1
            human_int: bool = False
            # Human Intervention
            if coin and curr_hint > 0:
                if self.ts_handle.adj_map.get(curr_ts_tuple, {}).get(curr_hint, {}).get('h'):
                    human_int = True
                    next_tuple, curr_hint = self.human_intervention(curr_state_tuple=curr_ts_tuple,
                                                                    curr_dfa_state=curr_dfa_state,
                                                                    curr_hint=curr_hint,
                                                                    verbose=verbose)
            if not human_int:
                # if all the action from the current state belong to the 
                curr_act_cubes = self.convert_prod_cube_to_func(dd_func=curr_act, prod_curr_list=self.sys_act_vars)

                # if multiple winning actions exisit from same state
                if len(curr_act_cubes) > 1:
                    ract_name = None
                    while ract_name is None:
                        ract_dd: List[int] = random.choice(curr_act_cubes)
                        ract_name = self.ts_bdd_sym_to_robot_act_map.get(ract_dd, None)

                else:
                    ract_name = self.ts_bdd_sym_to_robot_act_map[curr_act]

                if verbose:
                    print(f"Step {counter}: Hint: {curr_hint}: Conf: {self.ts_handle.get_state_from_tuple(curr_ts_tuple)} Act: {ract_name}")
                
                # look up the next tuple 
                next_exp_states = self.ts_handle.adj_map[curr_ts_tuple][curr_hint][ract_name]
                next_tuple = self.ts_handle.get_tuple_from_state(next_exp_states)

            # look up its corresponding formula
            curr_ts_state: BDD = self.ts_handle.get_sym_state_from_tuple(next_tuple)

            curr_ts_lbl: BDD = curr_ts_state.existAbstract(self.ts_xcube)

            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state in self.dfa_bdd_sym_to_curr_state_map.keys():
                dfa_pre = dfa_state.vectorCompose(self.dfa_x_list, self.dfa_transition_fun_list)
                edge_exists: bool = not (dfa_pre & (curr_dfa_state & curr_ts_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state = dfa_state
                    break 

           
            curr_prod_state = curr_ts_state & curr_dfa_state & self.ts_bdd_sym_to_hint_map.inv[curr_hint]

            counter += 1