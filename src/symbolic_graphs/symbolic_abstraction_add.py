import re
import sys
import copy
import graphviz as gv

from typing import Tuple, List, Dict
from collections import defaultdict
from functools import reduce
from cudd import Cudd, BDD, ADD
from itertools import product

from src.explicit_graphs import FiniteTransitionSystem

from bidict import bidict

from config import *
from utls import *


class SymbolicWeightedTransitionSystem(object):
    """
    A class to construct a symbolic Weighted transition system for each operator - in our case Actions. 

    curr_state: Symbolic Boolean vairables corresponding to current states
    next_state: Symbolic Boolean vairables corresponding to next states
    lbl_state: Symbolic Boolean vairables corresponding to labels 
    task: Pyperplan object that contains information regarding the task 
    domain : Pyperplan object that contains information regarding the domain
    observations: All the possible observations for a give problem type. For grid world this is same a objects
    manager: CUDD Manager
    weight_dict: A mapping from pddl action to their corresponding weights in the dictionary. 

    We had to create a dedicated class for this as all the variables are add variables in the this class.
    This includes states, transition function, and labels 
    """

    def __init__(self, curr_states: List[ADD], next_states: List[ADD], lbl_states: List[ADD], task, domain, weight_dict, manager: Cudd):
        self.sym_add_vars_curr: List[ADD] = curr_states
        self.sym_add_vars_next: List[ADD] = next_states
        self.sym_add_vars_lbl: List[ADD] = lbl_states
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.facts:dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.manager: Cudd = manager
        self.weight_dict: Dict[str, List[ADD]] = weight_dict

        self.tr_action_idx_map: dict = {}
        self.sym_add_init_states = manager.addZero()
        self.sym_add_goal_states = manager.addZero()
        self.sym_add_state_labels = manager.addZero()
        self.sym_tr_actions: list = []
        self.predicate_add_sym_map_curr: bidict = {}
        self.predicate_add_sym_map_nxt: bidict = {}
        self.predicate_add_sym_map_lbl: bidict = {}

        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl: bidict = {}

        self._create_sym_var_map()
        self._initialize_adds_for_actions()
        self._initialize_sym_init_goal_states()
    

    def _initialize_adds_for_actions(self):
        """
        A function to intialize bdds for all the actions
        """
        #  initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = list(self.weight_dict.keys())
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [self.manager.addZero() for _ in range(len(self.weight_dict))]
    

    def _initialize_sym_init_goal_states(self):
        """
        Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        _init_list = [self.predicate_add_sym_map_curr.get(s) for s in list(self.init)]
        _goal_list = [self.predicate_add_sym_map_curr.get(s) for s in list(self.goal)]

        self.sym_add_init_states = reduce(lambda a, b: a | b, _init_list) 
        self.sym_add_goal_states = reduce(lambda a, b: a | b, _goal_list)
    

    def _create_sym_var_map(self):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """

        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_curr)))

        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(self.facts)})
        _node_int_map_next = copy.deepcopy(_node_int_map_curr)

        _node_bdd_int_map_curr = copy.deepcopy(_node_int_map_curr)
        _node_bdd_int_map_next = copy.deepcopy(_node_int_map_curr)

        assert len(boolean_str) >= len(_node_int_map_next), "FIX THIS: Looks like there are more Facts that boolean variables!"

        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            _next_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _curr_val_list.append(self.sym_add_vars_curr[_idx])
                    _next_val_list.append(self.sym_add_vars_next[_idx])
                else:
                    _curr_val_list.append(~self.sym_add_vars_curr[_idx])
                    _next_val_list.append(~self.sym_add_vars_next[_idx])
            
            # we will create two dictionary, one mapping add to curr and nxt state
            # the other dictionary will make the corresponding bdd to curr and nxt state
            _bool_add_func_curr = reduce(lambda a, b: a & b, _curr_val_list)
            _bool_add_func_nxt = reduce(lambda a, b: a & b, _next_val_list)

            _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()
            _bool_bdd_func_nxt = _bool_add_func_nxt.bddPattern()

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_add_func_curr
            _node_int_map_next[_key] = _bool_add_func_nxt

            _node_bdd_int_map_curr[_key] = _bool_bdd_func_curr
            _node_bdd_int_map_next[_key] = _bool_bdd_func_nxt 
        
        self.predicate_add_sym_map_curr = _node_int_map_curr
        self.predicate_add_sym_map_nxt = _node_int_map_next

        self.predicate_sym_map_curr = _node_bdd_int_map_curr
        self.predicate_sym_map_nxt = _node_bdd_int_map_next  


    def _create_sym_state_label_map(self, domain_lbls):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """
        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_lbl)))

        _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(domain_lbls)})
        _node_bdd_int_map_lbl = copy.deepcopy(_node_int_map_lbl)

        assert len(boolean_str) >= len(_node_int_map_lbl), "FIX THIS: Looks like there are more lbls that boolean variables!"

        # loop over all the boolean string and convert them to their respective bdd vars
        for _key, _value in _node_int_map_lbl.items():
            _lbl_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _lbl_val_list.append(self.sym_add_vars_lbl[_idx])
                else:
                    _lbl_val_list.append(~self.sym_add_vars_lbl[_idx])
            
            _bool_add_func_curr = reduce(lambda a, b: a & b, _lbl_val_list)
            _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()

            # update bidict accordingly
            _node_int_map_lbl[_key] = _bool_add_func_curr
            _node_bdd_int_map_lbl[_key] = _bool_bdd_func_curr
        
        self.predicate_add_sym_map_lbl = _node_int_map_lbl
        self.predicate_sym_map_lbl = _node_bdd_int_map_lbl
    

    def create_weighted_transition_system(self, verbose: bool = False, plot: bool = False):
        """
        A function to create the TR function for each Action we have defined in the domain along with its weight
        """
        if not all(isinstance(tr_const, ADD) for tr_const in self.weight_dict.values()):
            print("Please Make sure your edge weights are of type ADD. FIX THIS!!!")
            sys.exit()

        for _action in self.task.operators:
            self.build_actions_tr_func_weighted(curr_edge_action=_action)

        if verbose:
            for _action, _idx in self.tr_action_idx_map.items():
                print(f"Charateristic Function for action {_action} \n")
                print(self.sym_tr_actions[_idx], " \n")
                if plot:
                    file_path = PROJECT_ROOT + f'/plots/{_action}_ADD_trans_func.dot'
                    file_name = PROJECT_ROOT + f'/plots/{_action}_ADD__trans_func.pdf'
                    self.manager.dumpDot([self.sym_tr_actions[_idx]], file_path=file_path)
                    gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)
    

    def build_actions_tr_func_weighted(self, curr_edge_action: str):
        """
        A function to build a symbolic transition function corresponding to each action along with its weight
        """

        _actions = list(self.weight_dict.keys())
        # since pre, post condition are all forzenset, we iterate over it
        pre_conds = tuple(curr_edge_action.preconditions)
        add_effects = tuple(curr_edge_action.add_effects)
        del_effects = tuple(curr_edge_action.del_effects)

        # get the bool formula for the above predicates in terms of add variables
        # Note add in the predictice_add... is not the same as addition but it is Algebraic Decision Diagram
        pre_list = [self.predicate_add_sym_map_curr.get(pre_cond) for pre_cond in pre_conds]
        add_list = [self.predicate_add_sym_map_nxt.get(add_effect) for add_effect in add_effects]
        del_list = [self.predicate_add_sym_map_nxt.get(del_effect) for del_effect in del_effects]

        if len(pre_list) != 0:
            pre_sym = reduce(lambda a, b: a & b, pre_list)
        else:
            pre_sym = self.manager.bddOne()
        if len(add_list) != 0:
            add_sym = reduce(lambda a, b: a & b, add_list)
        else:
            add_sym = self.manager.bddOne()
        if len(del_list) != 0:
            del_sym = reduce(lambda a, b: a & b, del_list)
        else:
            del_sym = self.manager.bddZero()
        
        _curr_action_name = curr_edge_action.name

        # instead looking for the action, extract it (action name)
        _action = _curr_action_name.split()[0]
        _action = _action[1:]   # remove the intial '(' braket

        # assert that its a valid name
        assert _action in _actions, "FIX THIS: Failed extracting a valid action."

        # pre_sym - precondition will be false when starting from the initial states; mean you can take the action under all conditions
        if pre_sym.isZero():
            pre_sym = pre_sym.negate()
        
        if add_sym.isZero():
            add_sym = add_sym.negate()

        _idx = self.tr_action_idx_map.get(_action)  

        self.sym_tr_actions[_idx] |= pre_sym & add_sym & ~del_sym & self.weight_dict[_action]
        
        return _action
    

    def create_state_obs_add(self, domain_lbls, verbose: bool = False, plot: bool = False):
        """
        A function to create the Charactersitic function for each state observation in the domain. 
        """
        self._create_sym_state_label_map(domain_lbls=domain_lbls)
        # this onlt works for grid world for now. A state looks like - `at skbn l#` and we are trying extract l# and add that as bdd
        for state in self.facts:
            split_str = re.split(" ", state)
            _loc = split_str[-1][:-1]   # remove the trailing bracket

            # look up the correspondig boolean formula associated with 
            _state_add = self.predicate_add_sym_map_curr.get(state)
            _obs_add = self.predicate_add_sym_map_lbl.get(_loc)

            assert _obs_add or _state_add is not None, "Looks like we extracted an invalid observation. FIX THIS!!!"

            self.sym_add_state_labels |= _state_add & _obs_add
        
        if verbose:
            print("*************Printing State Observation BDD*************")
            print(self.sym_add_state_labels)
            if plot: 
                file_path = PROJECT_ROOT + f'/plots/S2Obs_ADD_trans_func.dot'
                file_name = PROJECT_ROOT + f'/plots/S2Obs_ADD_trans_func.pdf'
                self.manager.dumpDot([self.sym_add_state_labels], file_path=file_path)
                gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)



class SymbolicWeightedFrankaTransitionSystem():
    """
     ADD version of the BDD Franka abstraction construction class.
    """
    
    def __init__(self,
                 curr_states: List[ADD],
                 next_states: List[ADD],
                 lbl_states: List[ADD],
                 task, domain,
                 weight_dict,
                 ts_state_map: dict,
                 ts_states: list,
                 manager: Cudd):
        
        self.sym_add_vars_curr: List[ADD] = curr_states
        self.sym_add_vars_next: List[ADD] = next_states
        self.sym_add_vars_lbl: List[ADD] = lbl_states
        
        self.init: frozenset = task.initial_state
        self.goal: frozenset = task.goals
        self.ts_states: dict = ts_states
        self.pred_int_map: dict = ts_state_map
        
        self.facts: dict = task.facts
        self.task: dict = task
        self.domain: dict = domain
        self.manager: Cudd = manager
        self.weight_dict: Dict[str, List[ADD]] = weight_dict

        self.tr_action_idx_map: dict = {}
        self.sym_add_init_states = manager.addZero()
        self.sym_add_goal_states = manager.addZero()
        self.sym_add_state_labels = manager.addZero()
        self.sym_tr_actions: list = []
        self.predicate_add_sym_map_curr: bidict = {}
        self.predicate_add_sym_map_nxt: bidict = {}
        self.predicate_add_sym_map_lbl: bidict = {}

        self.predicate_sym_map_curr: bidict = {}
        self.predicate_sym_map_nxt: bidict = {}
        self.predicate_sym_map_lbl: bidict = {}

        self._create_sym_var_map()
        self._initialize_adds_for_actions()
        self._initialize_sym_init_goal_states()
    

    def _initialize_adds_for_actions(self):
        """
        A function to intialize bdds for all the actions
        """
        #  initiate BDDs for all the action 
        action_idx_map = bidict()
        _actions = list(self.weight_dict.keys())
        for _idx, _action in enumerate(_actions):
            action_idx_map[_action] = _idx
        
        self.tr_action_idx_map = action_idx_map
        self.sym_tr_actions = [self.manager.addZero() for _ in range(len(self.weight_dict))]
    

    def _initialize_sym_init_goal_states(self):
        """
        Initialize the inital states of the Transition system with their corresponding symbolic init state vairants.
        """
        init_tuple = self.get_tuple_from_state(self.init)
        goal_tuple = self.get_tuple_from_state(self.goal)

        self.sym_add_init_states = self.predicate_add_sym_map_curr.get(init_tuple)
        self.sym_add_goal_states = None

        assert self.sym_add_init_states is not None, "Error extracting the Sym init state. FIX THIS!!!"
    

    def _create_sym_var_map(self):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it
        """

        # create all combinations of 1-true and 0-false
        boolean_str = list(product([1, 0], repeat=len(self.sym_add_vars_curr)))

        _node_int_map_curr = bidict({state: boolean_str[index] for index, state in enumerate(self.ts_states)})
        _node_int_map_next = copy.deepcopy(_node_int_map_curr)

        _node_bdd_int_map_curr = copy.deepcopy(_node_int_map_curr)
        _node_bdd_int_map_next = copy.deepcopy(_node_int_map_curr)

        assert len(boolean_str) >= len(_node_int_map_next), "FIX THIS: Looks like there are more Facts that boolean variables!"

        # loop over all the boolean strings and convert them respective bdd vars
        for _key, _value in _node_int_map_curr.items():
            _curr_val_list = []
            _next_val_list = []
            for _idx, _ele in enumerate(_value):
                if _ele == 1:
                    _curr_val_list.append(self.sym_add_vars_curr[_idx])
                    _next_val_list.append(self.sym_add_vars_next[_idx])
                else:
                    _curr_val_list.append(~self.sym_add_vars_curr[_idx])
                    _next_val_list.append(~self.sym_add_vars_next[_idx])
            
            # we will create two dictionary, one mapping add to curr and nxt state
            # the other dictionary will make the corresponding bdd to curr and nxt state
            _bool_add_func_curr = reduce(lambda a, b: a & b, _curr_val_list)
            _bool_add_func_nxt = reduce(lambda a, b: a & b, _next_val_list)

            _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()
            _bool_bdd_func_nxt = _bool_add_func_nxt.bddPattern()

            # update bidict accordingly
            _node_int_map_curr[_key] = _bool_add_func_curr
            _node_int_map_next[_key] = _bool_add_func_nxt

            _node_bdd_int_map_curr[_key] = _bool_bdd_func_curr
            _node_bdd_int_map_next[_key] = _bool_bdd_func_nxt 
        
        self.predicate_add_sym_map_curr = bidict(_node_int_map_curr)
        self.predicate_add_sym_map_nxt = bidict(_node_int_map_next)

        self.predicate_sym_map_curr = bidict(_node_bdd_int_map_curr)
        self.predicate_sym_map_nxt = bidict(_node_bdd_int_map_next)
    

    def get_tuple_from_state(self, preds: list) -> Tuple[int]:
        """
         Given, a predicate tuple, this function return the corresponding state tuple
        """
        _int_tuple = [self.pred_int_map[pred] for pred in preds]

        return tuple(sorted(_int_tuple))
    

    def get_state_from_tuple(self, state_tuple: tuple) -> List[str]:
        """
         Given, a predicate tuple, this function return the corresponding state tuple
        """
        if isinstance(state_tuple, tuple):
            _states = [self.pred_int_map.inv[state] for state in state_tuple]
        else:
            _states = self.pred_int_map.inv[state_tuple[0]]

        return _states
    

    def _check_exist_constraint(self, boxes: List[str], curr_state_lbl: tuple, action_name: str) -> bool:
        """
        A helper function that take as input the state label (on b0 l0)(on b1 l1) and the action name,
         extracts the destination location from action name and its corresponding world conf tuple.
         We then take the intersection of the  curr_state_lbl & corresponding world conf tuple.
         
        Return False if intersection is non-empty else True
        """
        finite_ts = FiniteTransitionSystem(None)

        if 'transfer' in action_name: 
            box_id, locs = finite_ts._get_multiple_box_location(multiple_box_location_str=action_name)
            dloc = locs[1]
        elif 'release' in action_name:
            box_id, locs = finite_ts._get_box_location(box_location_state_str=action_name)
            dloc = locs
        
        # if box1 is being manipulated, get the list of rest of boxes (that have to be grounded) at this instance
        tmp_copy = copy.deepcopy(boxes)
        tmp_copy.remove(f'b{box_id}')

        # create predicates that say on b0 l1 (l1 is the destination in the action)
        on_preds = [self.pred_int_map[f'(on {bid} {dloc})'] for bid in tmp_copy]
        
        if set(on_preds).intersection(set(curr_state_lbl)):
            return False

        return True
    

    def get_conds_from_state(self, state_tuple: tuple, only_world_conf: bool = False, only_robot_conf: bool = False) -> Tuple[int]:
        """
         A function that loops through the state tuple, and returns the tuple corresponding the robot conf or world ocnf.

         @param: only_world_conf - Set this to true if you only want to return `on` predicates (box locations)
         @param: only_robot_conf - Set this to true if you want to return predicates related to robot conf. (ready, to-obj, holding, to-loc) 
        """
        preds = self.get_state_from_tuple(state_tuple=state_tuple)

        _int_tuple = []
        for pred in preds:
            if only_world_conf:
                if ('on' in pred) or ('gripper' in pred):
                    _int_tuple.append(self.pred_int_map[pred])

            elif only_robot_conf:
                if not(('on' in pred) or ('gripper' in pred)):
                    _int_tuple.append(self.pred_int_map[pred])
        
        return tuple(sorted(_int_tuple))
    

    def _create_sym_state_label_map(self, domain_lbls):
        """
        Loop through all the facts that are reachable and assign a boolean funtion to it.
         
         This method is called whten Gridworld state labels are created
        """
        
        # loop over each box and create its corresponding boolean formula 
        for b_id, preds in domain_lbls.items():
            # get its corresponding boolean vars
            _tmp_vars_list = []
            if b_id == 'gripper':
                _tmp_vars_list.append(self.sym_add_vars_lbl[-1])
            else:
                for bvar in self.sym_add_vars_lbl:
                    if f'{b_id}_' in str(bvar.bddPattern()):
                        _tmp_vars_list.append(bvar)

            # create all combinations of 1-true and 0-false
            boolean_str = list(product([1, 0], repeat=len(_tmp_vars_list)))

            _node_int_map_lbl = bidict({state: boolean_str[index] for index, state in enumerate(preds)})
            _node_bdd_int_map_lbl = copy.deepcopy(_node_int_map_lbl)

            assert len(boolean_str) >= len(_node_int_map_lbl), "FIX THIS: Looks like there are more lbls that boolean variables!"

            # loop over all the boolean string and convert them to their respective bdd vars
            for _key, _value in _node_int_map_lbl.items():
                _lbl_val_list = []
                for _idx, _ele in enumerate(_value):
                    if _ele == 1:
                        _lbl_val_list.append(_tmp_vars_list[_idx])
                    else:
                        _lbl_val_list.append(~_tmp_vars_list[_idx])
                
                _bool_add_func_curr = reduce(lambda a, b: a & b, _lbl_val_list)
                _bool_bdd_func_curr = _bool_add_func_curr.bddPattern()

                # update bidict accordingly
                _node_int_map_lbl[_key] = _bool_add_func_curr
                _node_bdd_int_map_lbl[_key] = _bool_bdd_func_curr
            
            self.predicate_add_sym_map_lbl.update(_node_int_map_lbl)
            self.predicate_sym_map_lbl.update(_node_bdd_int_map_lbl)
        
        self.predicate_sym_map_lbl = bidict(self.predicate_sym_map_lbl)
        self.predicate_add_sym_map_lbl = bidict(self.predicate_add_sym_map_lbl)
    

    def get_sym_state_lbl_from_tuple(self, state_lbl_tuple: tuple) -> ADD:
        """
         A function that converts the corresponding state lbl tuple to its explicit predicate form,
          looks up its corresponding boolean formula, and return the conjunction of all the boolean formula
        """
        # get the explicit preds
        exp_lbls = self.get_state_from_tuple(state_tuple=state_lbl_tuple)

        _sym_lbls_list = [self.predicate_add_sym_map_lbl[lbl] for lbl in exp_lbls]

        # if gripper is not free then explicitly add not(gripper free) to the state lbl
        if '(gripper free)' not in exp_lbls:
            _sym_lbls_list.append(~self.predicate_add_sym_map_lbl['(gripper free)'])


        sym_lbl = reduce(lambda x, y: x & y, _sym_lbls_list)

        assert not sym_lbl.isZero(), "Error constrcuting the symbolic lbl associated with each state. FIX THIS!!!"

        return sym_lbl
    

    def get_states_from_dd(self, dd_func: ADD) -> List[ADD]:
        """
         A function that wraps around _convert_state_lbl_cube_to_func() by first converting the dd_func into a BDD,
          extracting all the cubes and then converting them back to 0-1 ADD. 
        """

        tmp_dd_func: BDD = dd_func.bddPattern()
        tmp_ts_x_list: List[BDD] = [_avar.bddPattern() for _avar in self.sym_add_vars_curr]
        cube_strings: List[BDD] = self._convert_state_lbl_cube_to_func(dd_func=tmp_dd_func, prod_curr_list=tmp_ts_x_list)

        # convert them back ADDs and return them
        cubes: List[ADD] = [cube.toADD() for cube in cube_strings] 

        return cubes

    
    def _convert_state_lbl_cube_to_func(self, dd_func: BDD, prod_curr_list = None) ->  List[BDD]:
        """
         A helper function to extract a cubes from the DD and print them in human-readable form. 
        """
        ddVars = []
        for cube in dd_func.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube):
                # skip the primed variables
                if var == 2 and self.manager.bddVar(_idx) not in prod_curr_list:   # not x list is better than y _list because we also have dfa vairables 
                    continue   # skipping over prime states 
                else:
                    if var == 2:
                        _amb_var.append([self.manager.bddVar(_idx), ~self.manager.bddVar(_idx)])   # count how many vars are missing to fully define the bdd
                    elif var == 0:
                        var_list.append(~self.manager.bddVar(_idx))
                    elif var == 1:
                        var_list.append(self.manager.bddVar(_idx))
                    else:
                        print("CUDD ERRROR, A variable is assigned an unaccounted integer assignment. FIX THIS!!")
                        sys.exit(-1)
                
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))  # *is to pass list as iterable
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    ddVars.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                ddVars.append(reduce(lambda a, b: a & b, var_list))
        
        return ddVars
    

    def add_edge_to_action_tr(self, action_name: str, curr_state_tuple: tuple, next_state_tuple: tuple) -> None:
        """
         A helper function that add the edge from curr state to the next state in their respective action Transition Relations (TR)
        """
        curr_state_sym: ADD = self.predicate_add_sym_map_curr[curr_state_tuple]
        nxt_state_sym: ADD = self.predicate_add_sym_map_nxt[next_state_tuple]

        _idx = self.tr_action_idx_map.get(action_name)

        self.sym_tr_actions[_idx] |= curr_state_sym & nxt_state_sym & self.weight_dict[action_name]

    

    def create_weighted_transition_system_franka(self,
                                                 boxes: List[str],
                                                 state_lbls: List,
                                                 add_exist_constr: bool = True,
                                                 verbose:bool = False,
                                                 plot: bool = False):
        """
         This function create the symbolic trnaition relation for the Franka World.

         The construction of TR for the franka world is a bit different than the gridworld. In gridworld the
          complete information of the world, i.e., the agent current location is embedded and thus
          create_trasntion_system()'s implementation sufficient when we label the states.

         Unlike gridworld, in franka world, we have to start from the iniital states with the initial conditions
          explicitly mentioned in problem file. From here, we create the state (pre) and its corresponding labels,
          we unroll the graph by taking all valid actions, compute the image, and their corresponding labels and
          keep iterating until we reach a fix point. 
        
        @param add_exist_constr: Adds the existential constraint that
            1) no other box should exist at the destination while performing Transfer action - transfer b# l# l#
            2) no other box should exist at the drop location while performing Relase action - release b# l# 
        """

        if verbose:
            print(f"Creating TR for Actions:", *self.tr_action_idx_map.keys())

        self._create_sym_state_label_map(domain_lbls=state_lbls)

        open_list = defaultdict(lambda: self.manager.addZero())

        closed: ADD = self.manager.addZero()

        init_state_sym = self.sym_add_init_states

        # get the state lbls and create state and state lbl mappinng
        state_lbl: tuple = self.get_conds_from_state(state_tuple=self.predicate_add_sym_map_curr.inv[init_state_sym], only_world_conf=True)
        init_lbl_sym: ADD = self.get_sym_state_lbl_from_tuple(state_lbl)

        self.sym_add_state_labels |= init_state_sym & init_lbl_sym

        layer = 0

        # no need to check if other boxes are placed at the destination loc during transfer and release as there is only one object
        if len(boxes) == 1:
            add_exist_constr = False

        open_list[layer] |= init_state_sym

        while not open_list[layer].isZero():
            # remove all states that have been explored
            open_list[layer] = open_list[layer] & ~closed

            # If unexpanded states exist ...
            if not open_list[layer].isZero():
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer]

                if verbose:
                    print(f"******************************* Layer: {layer}*******************************")

                # get all the states
                # sym_state = self._convert_state_lbl_cube_to_func(dd_func= open_list[layer], prod_curr_list=self.sym_vars_curr)
                sym_state: List[ADD] = self.get_states_from_dd(dd_func= open_list[layer])
                for state in sym_state:
                    curr_state_tuple = self.predicate_add_sym_map_curr.inv[state]
                    
                    _valid_pre_list = []
                    # compute the image of the TS states
                    for action in self.task.operators:
                        # set action feasbility flag to True - used during transfer and release action to check the des loc is empty
                        action_feas: bool = True
                        pre_tuple = self.get_tuple_from_state(action.preconditions)
                        _necc_robot_conf = self.get_conds_from_state(pre_tuple, only_robot_conf=True)

                        _intersect: bool = set(pre_tuple).issubset(curr_state_tuple)

                        if _intersect:
                            # get valid pres from current state tuple
                            pre_robot_conf = self.get_conds_from_state(curr_state_tuple, only_robot_conf=True)
                            pre_robot_conf = tuple(set(pre_robot_conf).intersection(_necc_robot_conf))
                            pre_world_conf = self.get_conds_from_state(curr_state_tuple, only_world_conf=True)

                            _valid_pre = sorted(pre_robot_conf + pre_world_conf)
                            
                            if tuple(_valid_pre) != curr_state_tuple:
                                _valid_pre_sym: ADD = self.predicate_add_sym_map_curr[tuple(_valid_pre)]
                                # check if this state has already being explored or not
                                if not (_valid_pre_sym & closed).isZero():
                                    continue
                                _valid_pre_list.append(_valid_pre_sym)

                            # add existential constraints to transfer and relase action
                            if add_exist_constr and (('transfer' in action.name) or ('release' in action.name)):
                                action_feas = self._check_exist_constraint(boxes=boxes,
                                                                           curr_state_lbl=_valid_pre,
                                                                           action_name=action.name)
                            
                            if not action_feas:
                                continue

                            # get add and del tuples 
                            add_tuple = self.get_tuple_from_state(action.add_effects)
                            del_tuple = self.get_tuple_from_state(action.del_effects)

                            # construct the tuple for next state
                            next_tuple = list(set(_valid_pre) - set(del_tuple))
                            next_tuple = tuple(sorted(list(set(next_tuple + list(add_tuple)))))

                            # look up its corresponding formula
                            next_sym_state = self.predicate_add_sym_map_nxt[next_tuple]

                            if verbose:
                                cstate = self.get_state_from_tuple(state_tuple=tuple(_valid_pre))
                                nstate = self.get_state_from_tuple(state_tuple=next_tuple)
                                print(f"Adding edge: {cstate} -------{action.name}------> {nstate}")
                            
                            # add The edge to its corresponding action
                            self.add_edge_to_action_tr(action_name=action.name,
                                                       curr_state_tuple=tuple(_valid_pre),
                                                       next_state_tuple=next_tuple)


                            # swap variables 
                            next_sym_state = next_sym_state.swapVariables(self.sym_add_vars_curr, self.sym_add_vars_next)

                            # get their corresponding lbls 
                            next_tuple_lbl = self.get_conds_from_state(state_tuple=next_tuple, only_world_conf=True)
                            next_lbl_sym: ADD = self.get_sym_state_lbl_from_tuple(next_tuple_lbl)
                            self.sym_add_state_labels |= next_sym_state & next_lbl_sym

                            # store the image in the next bucket
                            open_list[layer + 1] |= next_sym_state

                    for _val_pre_sym in _valid_pre_list:
                        # add them the observation bdd
                        _valid_pre_lbl = self.get_conds_from_state(state_tuple=self.predicate_add_sym_map_curr.inv[_val_pre_sym],
                                                                   only_world_conf=True)
                        _valid_pre_lbl_sym = self.get_sym_state_lbl_from_tuple(_valid_pre_lbl)
                        self.sym_add_state_labels |= _val_pre_sym & _valid_pre_lbl_sym

                        closed |= _val_pre_sym
                
                layer += 1
        
        if verbose:
            for _action, _idx in self.tr_action_idx_map.items():
                print(f"Charateristic Function for action {_action} \n")
                print(self.sym_tr_actions[_idx], " \n")
                if plot:
                    file_path = PROJECT_ROOT + f'/plots/{_action}_ADD_trans_func.dot'
                    file_name = PROJECT_ROOT + f'/plots/{_action}_ADD_trans_func.pdf'
                    self.manager.dumpDot([self.sym_tr_actions[_idx]], file_path=file_path)
                    gv.render(engine='dot', format='pdf', filepath=file_path, outfile=file_name)