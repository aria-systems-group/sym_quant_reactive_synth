"""
 In this script, we will build the DFA Game, compute regret-minimizing strategies in a symbolic and partitioned manner.
 1. Build Game abstarction in a compositional manner.
 2. Call SPOT or MONA to build DFA from LTL/LTlf formula.
 3. Construct the DFA Game
 3. Construct Graph of Utility (GoU) and compute Min-Min Value Iteration
    3.1 Compute Best-alternate response for every system strategy (or edge in GoU)
 4. Construct Graph of Best-response (GoBr) and Compute Regret-minimizing strategies using Min-Max Value Iteration
"""
import sys
import time
import math

from functools import reduce
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Union, Optional

from bidict import bidict
from tabulate import tabulate

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame


class SymbolicPartitionedRegretDFAGame(SymbolicPartitionedDFAGame):
    """
     This class extends the SymbolicPartitionedDFAGame class to construct Graph of Utility (GoU) and Compute Min-Min stratgies and values.
    """
    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 restricted_human_boxes: List[int],
                 budget: int,
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False):
        self.budget: int = budget
        self.uVars: List[ADD] = []
        self.prime_uVars: List[ADD] = []
        self.brVars: List[ADD] = []
        self.prime_brVars: List[ADD] = []
        self.uVar_map: List[ADD] = bidict({}) 
        self.uVar_map_sym: List[ADD] = bidict({}) 
        self.brVar_map: List[ADD] = bidict({}) 
        self.brVar_map_sym: List[ADD] = bidict({}) 
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(boxes, locs, ratio, init, goal, formula, restricted_human_locs, restricted_human_boxes, ltlf_flag=ltlf_flag, enable_reordering=enable_reordering)
        self.states_per_cost: Dict[int, ADD] = defaultdict(lambda: self.manager.addZero())
        self.uVars_transition_relation = None
        self.brVars_transition_relation = None
        self.graph_of_utility_tr = None
        self.graph_of_br_tr  = None
        
        # store ADD(s-as-s')-1 transition relation for graph of utility
        self.monolithic_valid_full_gou_trns: ADD = self.manager.addZero()
        self.monolithic_valid_full_gobr_trns: ADD = self.manager.addZero()
        self.monolithic_valid_sabr_prime_br_trns: ADD = self.manager.addZero()
        # variables for storing optimal cooperative state values and regret optimal state values
        self.cVals = None
        self.rVals = None

        # book keeping
        self.gou_game_latches = self.latches + self.uVars + self.qVars
        self.gou_game_prime_latches = self.prime_latches + self.prime_uVars + self.prime_qVars

        self.gobr_game_latches = None
        self.gobr_game_prime_latches = None


    # override the create lacthes method to include Graph of utility latches
    def create_all_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the FrankaDynamic Turn-Based Game.
          1. turn variables - tVars
          2. ratio variables - kVars
          3. predicate variables - pVars
          4. box predicate variables - bVars
          5. Utiltiy variables - uVars
        """
        offset = self.manager.size()
        self.tVar: List[ADD] = [self.manager.addVar(offset, 't0')]
        self.kVars: List[ADD] = self.create_ratio_vars()
        self.pVars, self.bVars = self.create_latches()
        self.uVars = self.create_utility_latches()
        self.create_all_maps()
        self.create_all_sym_maps()

        # create DFA latches next
        self.create_dfa_latches_and_maps()

        # create uVars map
        self.create_utiltity_var_map()
    
    
    def create_all_prime_boolean_state_vars_and_maps(self):
        """
         The main method that creates all primed version of the boolean variables for the FrankaDynamic Turn-Based Game.
          1. prime turn variables - tVars
          2. prime ratio variables - kVars
          3. prime predicate variables - pVars
          4. prime box predicate variables - bVars
          4. prime utility predicate variables - uVars
        """
        offset = self.manager.size()
        self.prime_tVar: List[ADD] = [self.manager.addVar(offset, "pt0")]
        self.prime_kVars: List[ADD] = self.create_prime_ratio_vars()
        self.prime_pVars, self.prime_bVars = self.create_prime_latches()
        self.prime_uVars: List[ADD] = self.create_prime_utility_latches()
        
        self.create_all_sym_maps(prime=True)
        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
    

    def create_all_br_vars_maps(self):
        """
         Give, the set of best-response values, this method create all latches and maps for best-alternate response variables. 
          It also creates prime latches.
        """
        # create variables for best-alternate response values
        self.brVars = self.create_br_latches()
        self.prime_brVars = self.create_prime_br_latches()
        self.create_br_var_map()
        self.gobr_game_latches = self.latches + self.uVars + self.brVars + self.qVars
        self.gobr_game_prime_latches = self.prime_latches + self.prime_uVars + self.prime_brVars + self.prime_qVars
    

    def set_init_latch(self) -> ADD:
        """
        Ovveride the base method. In Graph of Utility, the initial state also includes the utility variable set to 0.

        We need to assert that the init state should be full defined, i.e., we need to every box's conf. else the init state is a set of states. 
         This causes issue when checking for init state value after VI algorithm terminates.
        """
        assert len(self.init) == self.boxes + 1, "[Error]: The init state should be fully defined for Regret Synthesis code else the Synthesis code will not work correctly."
        init_cube = self.tVar_map_sym['robot'] & self.kVar_map_sym['k0'] & self.uVar_map_sym['u0']
        for s in self.init:
            init_cube &= self.xVar_map_sym[s]
        return init_cube
    
    
    def set_goal_latch(self) -> ADD:
        """
         Ovveride the base method. In Graph of Utility, the initial state also includes the utility variable set to 0.
        """
        return (self.dfa_handle.goal_latch & ~self.uVar_map_sym[f'u{self.budget + 1}'])


    def create_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        uVar_size = math.ceil(math.log2(self.budget + 2)) # +1 here to account for 0-bit vector and another +1 for  budget +1 which is a sink state.
        # create an additional boolean var to skip the 0-vector latch
        uVar_size = uVar_size + 1 if pow(2, uVar_size) == self.budget + 2 else uVar_size 
        uVars: List[ADD] = [self.manager.addVar(u + varsize, 'u' + str(u)) for u in range(uVar_size)]
        return uVars
    

    def create_br_latches(self):
        """
         Create best-alterante response (br) variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        brVar_size = math.ceil(math.log2(len(self.brVals)))
        # create an additional boolean var to skip the 0-vector latch
        brVar_size = brVar_size + 1 if pow(2, brVar_size) == len(self.brVals) else brVar_size 
        brVars: List[ADD] = [self.manager.addVar(br + varsize, 'br' + str(br)) for br in range(brVar_size)]
        return brVars
    

    def create_prime_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        prime_uVars: List[ADD] = [self.manager.addVar(u + varsize, 'pu' + str(u)) for u in range(len(self.uVars))]
        return prime_uVars


    def create_prime_br_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        prime_brVars: List[ADD] = [self.manager.addVar(br + varsize, 'pbr' + str(br)) for br in range(len(self.brVars))]
        return prime_brVars


    def create_utiltity_var_map(self):
        """
         Small function to create symbolic maps for the uVar_map. 
         TODO: Update cube_to_add method to cubestring_to_add for better clarity.
        """
        # budget + 1 is a state to represent budget exceeded; it will be a sink state.
        for u in range(self.budget + 2):
            ubit_str = f"{u + 1:0{len(self.uVars)}b}"
            self.uVar_map[f'u{u}'] = ubit_str
            self.uVar_map_sym[f'u{u}'] = self.cube_to_add(ubit_str, self.uVars)
    

    def create_br_var_map(self):
        """
         Small function to create symbolic maps for the brVar_map. 
        """
        for idx, val in enumerate(self.brVals):
            bit_str = f"{idx + 1:0{len(self.brVars)}b}"
            self.brVar_map[val] = bit_str
            self.brVar_map_sym[val] = self.cube_to_add(bit_str, self.brVars)
    

    def get_states_per_cost(self):
        """
         A helper function that takes in the ADD weight abd return a vector of 0-1 ADD per cost.
        """
        min_val: int = 0
        max_val: int = 1
        
        for val in range(min_val, max_val + 1, 1):
            # if val != 0:
                # self.states_per_cost[val] |= self.weight.bddInterval(val, val).toADD() & ~self.init_latch
            # else:
            self.states_per_cost[val] |= self.weight.bddInterval(val, val).toADD() & self.monolithic_relevant_box_preds
        
        # manually add the init state to cost 0
        # self.states_per_cost[1] |= self.init_latch
    

    def create_utility_transition_relation(self):
        """
         Create the transition relation for utility variables.
        """
        self.uVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.uVars}
        self.get_states_per_cost()
        valid_state_costs = [1, 0]
        for u in range(self.budget + 1):
            uConf_cube = self.uVar_map_sym[f'u{u}']
            for state_cost in valid_state_costs:
                transition_cube = uConf_cube & self.states_per_cost[state_cost]
                
                prime_u_val = u + state_cost if (u + state_cost) <= self.budget else self.budget + 1 
                uConf_prime_cube_str = self.uVar_map[f'u{prime_u_val}']
                for sidx, s in enumerate(uConf_prime_cube_str):
                    if s == '1':
                        self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube
                
                # create s a_s s' monolithic ADD which we will use later for alternate best-response computation
                self.monolithic_valid_full_gou_trns |= transition_cube & self.monolithic_valid_full_dfa_game_trns & \
                      self.uVar_map_sym[f'u{prime_u_val}'].swapVariables(self.uVars, self.prime_uVars)
        
        # add self-loop for the sink state budget + 1
        for sidx, s in enumerate(self.uVar_map[f'u{self.budget + 1}']):
            if s == '1':
                self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |=  self.uVar_map_sym[f'u{self.budget + 1}']
    

    def create_best_alternate_response_transition_relation(self):
        self.brVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.brVars}
        
        for br in self.brVals:
            brConf_cube = self.brVar_map_sym[br]
            
            for prime_br in self.brVals:
                if prime_br <= br:
                    # create the transition cube
                    transition_cube = brConf_cube & self.vector_of_br[prime_br]
                    
                    prime_brConf_cube_str = self.brVar_map[prime_br]
                    for sidx, s in enumerate(prime_brConf_cube_str):
                        if s == '1':
                            self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= transition_cube
                    
                    # bookkeeping
                    # create (s,q,u,br) ---- a_s ---> (s',q', u', br') monolithic ADD which we will use later for alternate best-response computation
                    self.monolithic_valid_full_gobr_trns |= transition_cube & self.monolithic_valid_full_gou_trns & self.brVar_map_sym[prime_br].swapVariables(self.brVars, self.prime_brVars)
                    self.monolithic_valid_sabr_prime_br_trns |= transition_cube & self.monolithic_valid_state_robot_actions & self.brVar_map_sym[prime_br].swapVariables(self.brVars, self.prime_brVars)
                else:
                    state_act_pairs = self.manager.addZero()
                    for i in self.brVals:
                        if i > br:
                            state_act_pairs |= self.vector_of_br[i]
                    
                    transition_cube = brConf_cube & state_act_pairs
                
                    prime_brConf_cube_str = self.brVar_map[br]
                    for sidx, s in enumerate(prime_brConf_cube_str):
                        if s == '1':
                            self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= transition_cube

                    self.monolithic_valid_full_gobr_trns |= transition_cube & self.monolithic_valid_full_gou_trns & self.brVar_map_sym[br].swapVariables(self.brVars, self.prime_brVars)
                    self.monolithic_valid_sabr_prime_br_trns |= transition_cube & self.monolithic_valid_state_robot_actions & self.brVar_map_sym[br].swapVariables(self.brVars, self.prime_brVars)
                    break
        
            # add that from human states, the best-alternate response remains the same
            human_transition_cube = brConf_cube & self.tVar_map_sym['human']
            prime_brConf_cube_str = self.brVar_map[br]
            for sidx, s in enumerate(prime_brConf_cube_str):
                if s == '1':
                    self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= human_transition_cube
            
            # for human transitions, the best-alternate response remains the same
            self.monolithic_valid_full_gobr_trns |= human_transition_cube & self.brVar_map_sym[br].swapVariables(self.brVars, self.prime_brVars)
            self.monolithic_valid_sabr_prime_br_trns |= human_transition_cube & self.brVar_map_sym[br].swapVariables(self.brVars, self.prime_brVars)
    

    def get_states_with_one_outgoing_transition_gou(self) -> BDD:
        """
         This method computes the set of GoU states that have exactly one outgoing robot action.
        """
        gou_state_act_count: ADD = self.count_actions_per_state_gou()
        bdd_gou_state_single_act: BDD = gou_state_act_count.bddInterval(1, 1)

        # as accepting states in DFA are sink states in GoU, we need to post-process the gou_state_act_count so that accepting states map to cardinality 1.
        bdd_gou_state_single_act |= (self.dfa_handle.goal_latch & gou_state_act_count).bddPattern()
        return bdd_gou_state_single_act

    def gou_convert_cube_to_state_ADD(self,
                                      dd: ADD, state_flag: bool = True,
                                      dfa_flag: bool = True, action: bool = False,
                                      verbose: bool = False, table_header: bool = True, print_val: bool = True) -> None:
        """
         Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] + self.tVar + self.kVars + self.uVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)

        headers = []
        if verbose:
            headers.extend(['state'])
            if action:
                headers.append('action')
            if print_val:
                headers.append('value')
        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.rVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.rVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.rVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.rVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.rVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = []
        for cube, val in cubes:
            state = None
            tConf_exist_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_exist_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_exist_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_exist_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = ((self.tVar_map.inv[tConf_exist_str],
                          self.kVar_map.inv[kConf_exist_str],
                          self.pVar_map.inv[rConf_cube_str], box_states),
                          self.dfa_handle.qVar_map.inv[qConf_exist_str], self.uVar_map.inv[uConf_exist_str])
                states_action_pairs.append([
                    (((self.tVar_map.inv[tConf_exist_str],
                       self.kVar_map.inv[kConf_exist_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_exist_str], self.uVar_map.inv[uConf_exist_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(oCube_str, self.rVars)]
                except KeyError:
                    continue
            
            if action:
                if print_val:
                    states_bookkeeping.append((state, action_str, val))
                # states_bookkeeping.append((state, action_str, val))
                else:
                    states_bookkeeping.append((state, action_str))
            else:
                if print_val:
                    states_bookkeeping.append((state, val))
                else:
                    states_bookkeeping.append(state)
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))

        return states_action_pairs
    

    def gobr_convert_cube_to_state_ADD(self,
                                       dd: ADD, state_flag: bool = True,
                                       dfa_flag: bool = True, action: bool = False,
                                       verbose: bool = False, table_header: bool = True, print_val: bool = True) -> None:
        """
        Convert Graph of Best-Response state-action cubes to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] + self.tVar + self.kVars + self.uVars + self.brVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)
        
        headers = []
        if verbose:
            headers.extend(['state'])
            if action:
                headers.append('action')
            if print_val:
                headers.append('value')

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.brVars + self.rVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.brVars + self.rVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.brVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.brVars + self.rVars)
        brConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.uVars + self.rVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.brVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.brVars + self.rVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.brVars + self.rVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = []
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_cube_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_cube_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            brConf_cube_str = cube.existAbstract(brConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = (((self.tVar_map.inv[tConf_cube_str],
                          self.kVar_map.inv[kConf_cube_str],
                          self.pVar_map.inv[rConf_cube_str], box_states),
                          self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), self.brVar_map.inv[brConf_cube_str])
                states_action_pairs.append([
                    ((((self.tVar_map.inv[tConf_cube_str],
                       self.kVar_map.inv[kConf_cube_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), self.brVar_map.inv[brConf_cube_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(oCube_str, self.rVars)]
                except KeyError:
                    continue
            
            if action:
                states_bookkeeping.append((state, action_str, val))
            else:
                states_bookkeeping.append((state, val))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))

        return states_action_pairs


    def gou_convert_full_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, dfa_flag: bool = True, action: bool = True, verbose: bool = False) -> None:
        """
        Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True.
         If you want to print the human action as well, set human_action to True.

         Here the input dd is assumed to be a fully defined cube (latches as well prime latches).
        """
        relevant_vars = [] + self.uVars + self.prime_uVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes tVars, kVars, pVars and bVars
            relevant_vars.extend(self.prime_latches) # includes tVars, kVars, pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.qVars) # includes qVars
            relevant_vars.extend(self.prime_qVars) # includes prime qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create abstraction cubes
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.rVars + self.gou_game_prime_latches)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.rVars + self.gou_game_prime_latches)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.rVars + self.gou_game_prime_latches)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.rVars + self.gou_game_prime_latches)
        # create existential abstraction cubes - rConf
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars + self.gou_game_prime_latches) 

        # create prime abstraction cubes
        prime_tConf_exist_cube = reduce(lambda a, b: a & b, self.prime_xVars + self.prime_qVars + self.prime_uVars + self.rVars + self.gou_game_latches)
        prime_kConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_xVars[len(self.prime_kVars):] + self.prime_qVars + self.prime_uVars + self.rVars + self.gou_game_latches)
        prime_qConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_uVars + self.rVars + self.gou_game_latches)
        prime_uConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_qVars + self.rVars + self.gou_game_latches)

        # create PRIME existential abstraction cube - rConf 
        prime_rConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_qVars + self.prime_uVars + self.prime_xVars[len(self.prime_kVars)+len(self.prime_pVars):] + self.rVars + self.gou_game_latches)

        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        prime_bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.rVars + self.gou_game_prime_latches)
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.prime_qVars + self.prime_uVars + self.rVars + self.gou_game_latches)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.rVars + self.gou_game_prime_latches) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.prime_qVars + self.prime_uVars + self.rVars + self.gou_game_latches) & reduce(lambda x, y: x & y, self.prime_bVars_cubes[:bidx] + self.prime_bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        state_action_prime_pairs = []
        for cube, val in cubes:
            state = None
            prime_state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_cube_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_cube_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')

            prime_tConf_cube_str = cube.existAbstract(prime_tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_rConf_cube_str = cube.existAbstract(prime_rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_kConf_cube_str = cube.existAbstract(prime_kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_uConf_cube_str = cube.existAbstract(prime_uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_qConf_cube_str = cube.existAbstract(prime_qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                # print(f"[(({self.tVar_map.inv[tConf_cube_str]}, {self.kVar_map.inv[kConf_cube_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states}), {self.dfa_handle.qVar_map.inv[qConf_cube_str]}, {self.uVar_map.inv[uConf_cube_str]}), {val}]")
                state = ((self.tVar_map.inv[tConf_cube_str],
                          self.kVar_map.inv[kConf_cube_str],
                          self.pVar_map.inv[rConf_cube_str], box_states),
                          self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str])
                states_action_pairs.append([
                    (((self.tVar_map.inv[tConf_cube_str],
                       self.kVar_map.inv[kConf_cube_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), val), None])
            except KeyError:
                continue

             # print the robot and human actions as well
            if action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(oCube_str, self.rVars)]
                except KeyError:
                    continue
            

            prime_bCube_str = []
            for e in prime_bConf_exist_cube.values():
                prime_bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                prime_box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(prime_bCube_str))
            except KeyError:
                continue
               
            try:
                prime_state = ((self.tVar_map.inv[prime_tConf_cube_str],
                                self.kVar_map.inv[prime_kConf_cube_str],
                                self.pVar_map.inv[prime_rConf_cube_str], prime_box_states),
                                self.dfa_handle.qVar_map.inv[prime_qConf_cube_str], self.uVar_map.inv[prime_uConf_cube_str])
            except KeyError:
                continue
            
            # if you made it till here then print stuff or store them
            state_action_prime_pairs.append((state, action_str, prime_state, val))
        
        if verbose:
            print(tabulate(state_action_prime_pairs, headers=['state', 'robot action', 'prime state', 'value']))
            
        return states_action_pairs
    

    def gobr_convert_full_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, dfa_flag: bool = True, action: bool = False, verbose: bool = False) -> None:
        """
        Convert Graph of Best-Response full-state-robot-action cubes to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True.
         If you want to print the human action as well, set human_action to True.

         Here the input dd is assumed to be a fully defined cube (latches as well prime latches).
        """
        relevant_vars = [] + self.uVars + self.prime_uVars + self.brVars + self.prime_brVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes tVars, kVars, pVars and bVars
            relevant_vars.extend(self.prime_latches) # includes tVars, kVars, pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.qVars) # includes qVars
            relevant_vars.extend(self.prime_qVars) # includes prime qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create abstraction cubes
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.brVars + self.rVars + self.gobr_game_prime_latches)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.brVars + self.rVars + self.gobr_game_prime_latches)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.brVars + self.rVars + self.gobr_game_prime_latches)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.brVars + self.rVars + self.gobr_game_prime_latches)
        brConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.uVars + self.rVars + self.gobr_game_prime_latches)
        # create existential abstraction cubes - rConf
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.brVars + self.uVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars + self.gobr_game_prime_latches) 

        # create prime abstraction cubes
        prime_tConf_exist_cube = reduce(lambda a, b: a & b, self.prime_xVars + self.prime_qVars + self.prime_uVars + self.prime_brVars + self.rVars + self.gobr_game_latches)
        prime_kConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_xVars[len(self.prime_kVars):] + self.prime_qVars + self.prime_uVars + self.prime_brVars + self.rVars + self.gobr_game_latches)
        prime_qConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_uVars + self.prime_brVars + self.rVars + self.gobr_game_latches)
        prime_uConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_qVars + self.prime_brVars + self.rVars + self.gobr_game_latches)
        prime_brConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_qVars + self.prime_uVars + self.rVars + self.gobr_game_latches)
        # create PRIME existential abstraction cube - rConf 
        prime_rConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_qVars + self.prime_uVars + self.prime_brVars + self.prime_xVars[len(self.prime_kVars)+len(self.prime_pVars):] + self.rVars + self.gobr_game_latches)

        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        prime_bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.brVars + self.rVars + self.gobr_game_prime_latches)
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.prime_qVars + self.prime_uVars + self.prime_brVars + self.rVars + self.gobr_game_latches)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.brVars + self.rVars + self.gobr_game_prime_latches) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.prime_qVars + self.prime_uVars + self.prime_brVars + self.rVars + self.gobr_game_latches) & reduce(lambda x, y: x & y, self.prime_bVars_cubes[:bidx] + self.prime_bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        state_action_prime_pairs = []
        for cube, val in cubes:
            state = None
            prime_state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_cube_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_cube_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            brConf_cube_str = cube.existAbstract(brConf_exist_cube).bddPattern().cubeString().replace('-', '')

            prime_tConf_cube_str = cube.existAbstract(prime_tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_rConf_cube_str = cube.existAbstract(prime_rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_kConf_cube_str = cube.existAbstract(prime_kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_uConf_cube_str = cube.existAbstract(prime_uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_qConf_cube_str = cube.existAbstract(prime_qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_brConf_cube_str = cube.existAbstract(prime_brConf_exist_cube).bddPattern().cubeString().replace('-', '')
            
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = (((self.tVar_map.inv[tConf_cube_str],
                          self.kVar_map.inv[kConf_cube_str],
                          self.pVar_map.inv[rConf_cube_str], box_states),
                          self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), self.brVar_map.inv[brConf_cube_str])
                states_action_pairs.append([
                    ((((self.tVar_map.inv[tConf_cube_str],
                       self.kVar_map.inv[kConf_cube_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), self.brVar_map.inv[brConf_cube_str]), val), None])
            except KeyError:
                continue

             # print the robot and human actions as well
            if action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(oCube_str, self.rVars)]
                except KeyError:
                    continue
            

            prime_bCube_str = []
            for e in prime_bConf_exist_cube.values():
                prime_bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                prime_box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(prime_bCube_str))
            except KeyError:
                continue
               
            try:
                prime_state = (((self.tVar_map.inv[prime_tConf_cube_str],
                                self.kVar_map.inv[prime_kConf_cube_str],
                                self.pVar_map.inv[prime_rConf_cube_str], prime_box_states),
                                self.dfa_handle.qVar_map.inv[prime_qConf_cube_str], self.uVar_map.inv[prime_uConf_cube_str]), self.brVar_map.inv[prime_brConf_cube_str])
            except KeyError:
                continue
            
            # if you made it till here then print stuff or store them
            if action:
                state_action_prime_pairs.append((state, action_str, prime_state, val))
            else:
                state_action_prime_pairs.append((state, '', prime_state, val))
        
        if verbose:
            print(tabulate(state_action_prime_pairs, headers=['state', 'robot action', 'prime state', 'value']))
            
        return states_action_pairs

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        """
         Preimage comptuation over the Graph of Utility transition relation.
        """
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars, self.prime_latches + self.prime_uVars)
        preimage = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.graph_of_utility_tr)

        return preimage

    def compute_preimage_optimized(self, curr_winning_states: ADD) -> Tuple[ADD, ADD]:
        """
         Optimized preimage comptuation over the Graph of Utility transition relation.
        """
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)

        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars, self.prime_latches + self.prime_uVars)
        pre_sys = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.sys_gou_transition_relation)
        pre_env = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.env_gou_transition_relation)

        return pre_sys, pre_env
    
    def create_gou_sys_env_transition_relations(self):
        """
         A method to create separate transition relations for the system (robot) and environment (human) actions.
        """
        # post-process to get TR for robot and Human actions separately
        self.sys_gou_transition_relation = []
        self.env_gou_transition_relation = []
        for tr_dd in self.graph_of_utility_tr:
            self.sys_gou_transition_relation.append(tr_dd & self.tVar_map_sym['robot'])
            self.env_gou_transition_relation.append(tr_dd & self.tVar_map_sym['human'])

    
    def compute_regret_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the DFA game state (s, u)
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars + self.brVars, self.prime_latches + self.prime_uVars + self.prime_brVars)
        preimage_subr: ADD = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars + self.prime_brVars, self.graph_of_br_tr)

        return preimage_subr
    

    def create_goal_nodes_with_utility_values(self, verbose: bool = False) -> ADD:
        """
         Create Graph of Utility's accting nodes with utility values.
        """
        uVars_add = self.manager.plusInfinity()
        # skip u = 0; we reason about u = 0 separately
        for u in range(1, self.budget + 1):
            uVars_add = uVars_add.min(self.uVar_map_sym[f'u{u}'].ite(self.manager.addConst(u), self.manager.plusInfinity()))

        dfa_goal_cube: ADD = self.goal_latch.ite(self.manager.addOne(), self.manager.plusInfinity())
        goal_add: ADD = dfa_goal_cube.times(uVars_add)
        final_goal_add = (self.uVar_map_sym[f'u{0}'] & self.dfa_handle.goal_latch).ite(self.manager.addZero(), goal_add)
        if verbose:
            print(final_goal_add)
            print("Initialized the goal states with Utility values!")
        return final_goal_add


    def TVI_gou_solve(self, verbose: bool = False, optimized: bool = False) -> None:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        if optimized:
            self.create_gou_sys_env_transition_relations()
        
        goal = self.create_goal_nodes_with_utility_values(verbose=False)
        curr_states =  self.manager.plusInfinity()
        curr_states = curr_states.min(goal)
        # keeps track of optimal state values
        opt_state_val: ADD = curr_states
        frontier_curr_states = curr_states
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            if optimized:
                pre_sys, pre_env = self.compute_preimage_optimized(frontier_curr_states)
                pre_sys = pre_sys.min(opt_state_val)
                pre_env = pre_env.min(opt_state_val)
                next_winning_states_sys = self.symbolic_min_abstract(pre_sys, variables_to_abstract=self.rVars)
                next_winning_states_env = self.symbolic_min_abstract(pre_env, variables_to_abstract=self.rVars)
                frontier_next_states = self.tVar[0].ite(next_winning_states_sys, next_winning_states_env)
            else:
                frontier_preimage: ADD = self.compute_preimage(frontier_curr_states)
                frontier_preimage = frontier_preimage.min(opt_state_val)
                frontier_next_states = self.symbolic_min_abstract(frontier_preimage, variables_to_abstract=self.rVars)
            frontier_next_states = frontier_next_states.min(goal)

            if opt_state_val.compare(frontier_next_states, 2):
                print("**************************Reached fixpoint**************************")
                if (self.dfa_handle.init_latch & self.init_latch) & opt_state_val != self.manager.plusInfinity():
                    if (self.dfa_handle.init_latch & self.init_latch) & opt_state_val == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & self.init_latch & opt_state_val).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.cVals = opt_state_val
                return None
            
            # any cube who's value is 0 did not change its opt. state value.
            frontier_nodes = opt_state_val - frontier_next_states
            frontier_nodes_01_add = frontier_nodes.bddPattern().toADD()
            
            # adding debugging step
            if verbose:
                print("Current Winning States:")
                frontier_sVal = frontier_nodes_01_add#.times(next_states)
                self.gou_convert_cube_to_state_ADD(frontier_sVal, action=False, verbose=True)

            # swap the winning states
            opt_state_val = frontier_next_states
            frontier_curr_states = frontier_nodes_01_add.ite(opt_state_val, self.manager.plusInfinity())
            
            # update the counter
            layer += 1
    

    def gou_solve(self, verbose: bool = False, optimized: bool = False, test: bool = False) -> Optional[ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        if optimized:
            self.create_gou_sys_env_transition_relations()
        if test:
            self.graph_of_br_tr = list(self.transition_relation.values())
            self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
            self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        
        goal = self.create_goal_nodes_with_utility_values(verbose=verbose)
        # print("Goal States with Utility values:", goal)
        curr_winning_states =  self.manager.plusInfinity()
        curr_winning_states = curr_winning_states.min(goal)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            if optimized:
                pre_sys, pre_env = self.compute_preimage_optimized(curr_winning_states)
                next_winning_states_sys = self.symbolic_min_abstract(pre_sys, variables_to_abstract=self.rVars)
                next_winning_states_env = self.symbolic_min_abstract(pre_env, variables_to_abstract=self.rVars)
                next_winning_states = self.tVar[0].ite(next_winning_states_sys, next_winning_states_env)
            elif test:
                preimage = self.compute_regret_preimage(curr_winning_states)
                next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            else:
                preimage: ADD = self.compute_preimage(curr_winning_states)
                next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.gou_convert_cube_to_state_ADD(next_winning_states, action=False, verbose=True, print_val=True)
            
            # if curr_winning_states.compare(next_winning_states, 2):
            if next_winning_states.compare(curr_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states != self.manager.plusInfinity():
                    if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.cVals = curr_winning_states
                    if optimized:
                        preimage = pre_sys.min(pre_env)
                    return preimage if init_val < math.inf else None
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def get_next_state_br(self, turn: str, curr_state_exp: List[str], curr_action_sym: ADD, **kwargs) -> ADD:
        """
         A function to compute b' values during rollout in Graph of Best-response game. b' = min{b, br(s, a)}
        """
        curr_br_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        prime_brVars_bdd = kwargs['prime_brVars_bdd']
        
        if turn == 'robot':
            next_br = (self.monolithic_valid_sabr_prime_br_trns.restrict(curr_state_sym & curr_action_sym).bddInterval(1, 1).pickOneMinterm(prime_brVars_bdd)).toADD()
            return next_br.swapVariables(self.prime_brVars, self.brVars)
        else:
            return self.brVar_map_sym[curr_br_state_val]
    

    def get_next_state(self, turn: str, curr_state_exp: List[str], act_name: str, **kwargs) -> Tuple[ADD, str]:
        # get the next state in the game in explicit form
        curr_game_state = list(curr_state_exp[0][0][0][0])
        curr_utl_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        if turn == 'robot':
            curr_game_state_sym: ADD = self.get_next_state_robot(curr_game_state, act_name, curr_state_utl=curr_utl_state_val, curr_state_sym=curr_state_sym)
        else:
            curr_game_state_sym, act_name = self.get_next_state_human(curr_game_state, act_name, curr_state_utl=curr_utl_state_val)
        
        return curr_game_state_sym, act_name
    

    def get_next_state_robot(self, curr_state: List[str], action: str, **kwargs) -> ADD:
        next_state_dfa_game = super().get_next_state_robot(curr_state, action)

        # all the operations done in the parent method. Now include the utility variable transition
        try:
            curr_state_utl: str = kwargs['curr_state_utl']
            curr_state_sym: str = kwargs['curr_state_sym']
        except KeyError:
            print("Cannot rollout the strategy without current utility value or current state in symbolic form.")
            raise ValueError("curr_state_utl_val must be provided as a keyword argument.")

        # get the state cost
        if self.weight.cofactor(curr_state_sym).isZero():
            state_cost: int = 0
        else:
            state_cost: int = int(list((self.weight.cofactor(curr_state_sym)).generate_cubes())[0][1])
        state_utl: int = int(curr_state_utl[-1])

        if state_utl + state_cost <= self.budget:
            next_uVar_sym = self.uVar_map_sym[f'u{state_utl + state_cost}']
        else:
            next_uVar_sym = self.uVar_map_sym[f'u{self.budget + 1}']

        return next_state_dfa_game & next_uVar_sym
    

    def get_next_state_human(self, curr_state: List[str], action: str, **kwargs) -> ADD:
        next_state_dfa_game, act_name = super().get_next_state_human(curr_state, action)

        # all the operations done in the parent method. Now include the utility variable transition
        try:
            curr_state_utl: str = kwargs['curr_state_utl']
        except KeyError:
            print("Cannot rollout the strategy without current utility value or current state in symbolic form.")
            raise ValueError("curr_state_utl_val must be provided as a keyword argument.")

        # from the human state the cost remains the same
        return next_state_dfa_game & self.uVar_map_sym[curr_state_utl], act_name
    
    
    def gou_roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a strategy on the graph of utility game.
        """
        curr_state_sym = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]

        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            curr_state_exp: List[str] = self.gou_convert_cube_to_state_ADD(curr_state_sym,
                                                                            state_flag=True,
                                                                            action=False,
                                                                            verbose=False,
                                                                            table_header=False,
                                                                            print_val=False)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.cVals).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            if verbose:
                # print(tabulate([(curr_state_exp[0][0][0], opt_sval)], headers=['Current State', 'Optimal State Value']))
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)])) 

            turn = 'robot' if curr_state_exp[0][0][0][0][0] == 'robot' else'human'
            act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return
           
            curr_game_state_sym, act_name = self.get_next_state(turn, curr_state_exp, act_name, curr_state_sym=curr_state_sym)
            
            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            curr_dfa_state: int = curr_state_exp[0][0][0][1]
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_state_sym = dfa_state_sym.swapVariables(self.qVars, self.prime_qVars)
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_game_state_sym)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym.swapVariables(self.prime_qVars, self.qVars)
                    break
            
            curr_state_sym: ADD = curr_game_state_sym & curr_dfa_state
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Robot Action: {act_name}") if turn == 'robot' else print(f"Human Action: {act_name}")


    def gobr_roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a strategy on the graph of utility game.
        """
        curr_state_sym = self.init_latch & self.dfa_handle.init_latch & self.brVar_map_sym[math.inf]
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        prime_brVars_bdd: List[BDD] = [var.bddPattern() for var in self.prime_brVars]

        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            curr_state_exp: List[str] = self.gobr_convert_cube_to_state_ADD(curr_state_sym,
                                                                            state_flag=True,
                                                                            action=False,
                                                                            verbose=False,
                                                                            table_header=False,
                                                                            print_val=False)

            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.rVals).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            if verbose:
                # print(tabulate([(curr_state_exp[0][0][0], opt_sval)], headers=['Current State', 'Optimal State Value']))
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)])) 

            turn = 'robot' if curr_state_exp[0][0][0][0][0][0] == 'robot' else'human'

            # for state with optimal state value of 0, the restruct operation returns zero ADD.
            act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return
           
            # get the next state in the GoU game
            curr_gou_game_state_sym, act_name = self.get_next_state(turn, curr_state_exp[0], act_name, curr_state_sym=curr_state_sym)
            curr_gobr_br_sym = self.get_next_state_br(turn, curr_state_exp, curr_state_sym=curr_state_sym, prime_brVars_bdd=prime_brVars_bdd, curr_action_sym=act_cube.toADD())
            curr_game_state_sym = curr_gou_game_state_sym & curr_gobr_br_sym
            
            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            curr_dfa_state: int = curr_state_exp[0][0][0][0][1]
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_state_sym = dfa_state_sym.swapVariables(self.qVars, self.prime_qVars)
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_game_state_sym)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym.swapVariables(self.prime_qVars, self.qVars)
                    break
            
            curr_state_sym: ADD = curr_game_state_sym & curr_dfa_state
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Robot Action: {act_name}") if turn == 'robot' else print(f"Human Action: {act_name}")


    
    def create_transition_relation(self):
        # creates game transition relation first and then dfa transition relation
        super().create_transition_relation()

        # now create utility transition relation
        self.create_utility_transition_relation()

        tic = time.time()
        strategy = self.gou_solve(verbose=False, optimized=False)
        # self.TVI_gou_solve(verbose=False, optimized=False)
        toc = time.time()
        print(f"Time to synthesize GOU values: {toc - tic} seconds")

        # if strategy is not None:
        #     self.gou_roll_out_strategy(strategy=strategy, verbose=True)
        # return

        # compute best-alternate response
        self.compute_best_alternate_response(verbose=False)
        # sys.exit(-1)

        # create boolean vars and their prime versions for Best-alternate response values computed
        self.create_all_br_vars_maps()
        
        # create br Transition Relation
        self.create_best_alternate_response_transition_relation()

        # test BR TR for sanity checking
        # self.test_pre_image()
        # self.test_br_pre_image_old_approach()
    

    def count_actions_per_state_gou(self) -> ADD:
        """
         A function that counts the numbe of actions per state in graph of utility.
        """
        prime_vars_exist_cube = reduce(lambda a, b: a & b, self.gou_game_prime_latches)
        action_cube = reduce(lambda a, b: a & b, self.rVars)
        # convert to BDD and then exist abstract
        state_action_prime_state: BDD = self.monolithic_valid_full_gou_trns.bddPattern()
        state_action_prime_state = state_action_prime_state.existAbstract(prime_vars_exist_cube.bddPattern())

        # convert to 0 - 1 ADD and exist abstract rAct cubes to get ADD(s)->|s'| 
        state_ract_count: ADD = state_action_prime_state.toADD().existAbstract(action_cube)
        return state_ract_count

    
    def compute_best_alternate_response(self, verbose: bool = False) -> None:
        """
         A method to compute the best alterante response (ba). Given, tuple (s, s'), best-alternate response is the scalar value associated with:
            Informal: What if I took any other valid edge from (s, s'') where s'' =\= s' for every Sys player state.
            Mathermatically, given cVal (cooperative value) for every state s in G, we have

            ba(s, s') = +inf if s is Env player states
            ba(s, s') = min (s, s'') {cVal(s'')} if s is Sys plaeyr states

            min(s, s'') = +inf if no s'' exists, i.e., there does not exist an alternate edge.
        
        Note: we note that our Transition function is deterministic, i.e., given (si, ai) where i \in {Env, Sys}, 
        Tr(si, ai) -> sj' where j =\= i and s' is the next state such that |sj| = 1.

        Hence, the tuple (s, s') can be replaced with (s, as) which will be useful when constructing the TR for GoBR later.
        
        Method: Output ADD(s, as)-br where br is the best-response.
        """
        robot_action_cube = reduce(lambda a, b: a & b, self.rVars)
        
        # convert cVal to prime
        prime_cVal = self.cVals.swapVariables(self.gou_game_latches, self.gou_game_prime_latches)

        # AND with ADD(s, as, s')-1 that represents valid full TR to get ADD(s, as, s')-cVal(s') (leaf values is cVal(s'))
        full_state_prime_state_tr_cval: ADD = prime_cVal & self.monolithic_valid_full_gou_trns
        
        bdd_monolithic_valid_full_gou_trns: BDD = self.monolithic_valid_full_gou_trns.bddPattern()

        # precompute the set of state-robot action pairs (s, as)-> +inf. Used in find the min value later
        prime_vars_exist_cube = reduce(lambda a, b: a & b, self.gou_game_prime_latches)
        state_action_pair: BDD = bdd_monolithic_valid_full_gou_trns.existAbstract(prime_vars_exist_cube.bddPattern())

        # now compute the best alternate response
        self.ba_per_ract = defaultdict(lambda: self.manager.plusInfinity())
        self.vector_of_br = defaultdict(lambda: self.manager.addZero())
        for ract, ract_sym in self.relevant_robot_actions_sym.items():
            print(f"Computing BR for Robot Act: {ract}")
            
            # get the states where ract (as) is a valid action
            care_states_ract: BDD = state_action_pair & ract_sym.bddPattern()
            care_states_no_ract = care_states_ract.existAbstract(robot_action_cube.bddPattern())

            alternate_ract_cube: BDD = self.relevant_robot_actions.bddPattern() & ~ract_sym.bddPattern()

            # next get the ADD(s, as') where as' are all robot actions other than as
            care_states_alt_ract = state_action_pair & care_states_no_ract & alternate_ract_cube

            # get the full valid transition ADD(s, as', s)-1 and AND with ADD(s')-(cVal(s')) so that the leaves have the cVal of (s')
            add_care_states_alt_ract: ADD = (care_states_alt_ract.toADD()).ite(self.manager.addOne(), self.manager.plusInfinity())

            add_care_states_alt_ract_prime_states = add_care_states_alt_ract & full_state_prime_state_tr_cval

            # find the min amongst all (s, as', s') and store it
            ba_per_act = self.manager.plusInfinity().min(add_care_states_alt_ract_prime_states)
            self.ba_per_ract[ract] = ba_per_act

            # chop the ADDs into vector of BDD(s-as'-s'), one for each leaf node
            lVals = {int(leaf_value) if leaf_value != math.inf else leaf_value for _, leaf_value in ba_per_act.generate_cubes()}

            for leaf_val in lVals:
                # leav_vals == inf may have invalid states into, so post-process and remove it later
                bdd_state_act = (ba_per_act.bddInterval(leaf_val, leaf_val))
                if leaf_val == math.inf:
                    # remove the current action ract from bdd_state_act as they are default set to inf. 
                    bdd_state_act = (bdd_state_act & ~ract_sym.bddPattern()).existAbstract((prime_vars_exist_cube & robot_action_cube).bddPattern()) & ract_sym.bddPattern()
                else:
                    bdd_state_act = bdd_state_act.existAbstract((prime_vars_exist_cube & robot_action_cube).bddPattern()) & ract_sym.bddPattern()
                # remove goal states from br computation; later we add them to +inf br value
                bdd_state_act = bdd_state_act & ~self.dfa_handle.goal_latch.bddPattern()  
                self.vector_of_br[leaf_val] |= bdd_state_act.toADD() & self.monolithic_valid_state_robot_actions
        
        # print stuff for debugging
        print("Done computing BR")

        # ovveride the +inf BDD. The above code works for states with one egdes. 
        # The inf vector include these states as well as valid state conf. Further, we manually all accepting states in DFA to +inf as they are sink states in GoU.
        # This is not capture in the above code. Hence, we manually override the +inf BDD here.
        if math.inf in self.vector_of_br.keys():
            inf_states: BDD = self.get_states_with_one_outgoing_transition_gou()
            inf_state_actions: BDD = inf_states & state_action_pair
            self.vector_of_br[math.inf] |= inf_state_actions.toADD()
        
        self.brVals: Set[float] = sorted(set(self.vector_of_br.keys()))

        # post-processingbrst-response to only preserve the lwer states action pair value
        # unions of all predecessors
        pre_states: ADD = reduce(lambda x, y: x | y, self.vector_of_br.values())
        self.monolithich_br: ADD = pre_states.ite(self.manager.addOne(), self.manager.plusInfinity())
        for br in sorted(self.vector_of_br.keys(), reverse=True):
            br_states: ADD = self.vector_of_br[br]
            self.monolithich_br = br_states.ite(self.manager.addConst(br), self.monolithich_br)
        
        # finally put them back in vector_of_br
        for br in self.vector_of_br.keys():
            self.vector_of_br[br] = self.monolithich_br.bddInterval(br, br).toADD()

        if verbose:
            # l, h = min(set(self.brVals) - {math.inf}), max(set(self.brVals) - {math.inf})
            # t = self.monolithich_br.bddInterval(l, h).toADD()
            # self.gou_convert_cube_to_state_ADD(t, action=True, verbose=True)
            self.gou_convert_cube_to_state_ADD(self.monolithich_br, action=True, verbose=True)
    
    def create_goal_nodes_with_regret_values(self) -> Tuple[ADD, Set[float]]:
        """
         Create Graph of Best-Response nodes with regret values.
        """
        # create ADD(s-u)-Val(u) for every acceting state in game
        uVars_add = self.manager.plusInfinity()
        for u in range(self.budget + 1):
            uVars_add = uVars_add.min(self.uVar_map_sym[f'u{u}'].ite(self.manager.addConst(u), self.manager.plusInfinity()))

        brVars_add = self.manager.plusInfinity()
        for br in self.brVals:
            if br != math.inf:
                brVars_add = brVars_add.min(self.brVar_map_sym[br].ite(self.manager.addConst(br), self.manager.plusInfinity()))

        min_utility_br: ADD = uVars_add.min(brVars_add)
        print("Done taking the min between br and utility values!")

        reg_vals_add: ADD = uVars_add.minus(min_utility_br)
        # process reg values - all invalid uvArs conf. map to +inf
        valid_uVars_add = reduce(lambda x, y: x | y, self.uVar_map_sym.values())
        valid_brVars_add = reduce(lambda x, y: x | y, self.brVar_map_sym.values())
        reg_vals_add = valid_uVars_add.ite(reg_vals_add, self.manager.plusInfinity())

        # If uVars is buget + 1, then it is a sink states and hence also maps to +inf regret value
        reg_vals_add = self.uVar_map_sym[f'u{self.budget + 1}'].ite(self.manager.plusInfinity(), reg_vals_add)
        
        # invalid br vals also map to +inf regret value
        reg_vals_add = valid_brVars_add.ite(reg_vals_add, self.manager.plusInfinity())
        rVals = {int(leaf_value) if leaf_value != math.inf else leaf_value for _, leaf_value in reg_vals_add.generate_cubes()}
        # print(reg_vals)
        print("Processed the Regret Values!")

        goal_add: ADD = self.goal_latch.ite(reg_vals_add, self.manager.plusInfinity())
        # now restrict it to the set of valid box conf.
        goal_add = self.monolithic_relevant_box_preds.ite(goal_add, self.manager.plusInfinity())
        print("Initialized the goal states with regret values!")
        return goal_add, sorted(rVals)


    def regret_solver(self, verbose: bool = False, optimized: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm For computing regret minimizing strategies. 
        """
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        # initialize goal state with respective regret values
        goal, sorted_reg_vals = self.create_goal_nodes_with_regret_values()
        curr_winning_states = goal

        # intialize the iteration counter
        layer = 0
        regret_init_latch = self.init_latch & self.brVar_map_sym[math.inf]
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        while True:
            print(f"**************************Layer: {layer}**************************")
            if optimized:
                preimage = self.manager.plusInfinity()
                # preprocess the preimage to split into "buckets" of ADD with same values
                for reg_val in sorted_reg_vals:
                    curr_winning_states_reg_val = curr_winning_states.bddInterval(reg_val, reg_val).toADD()
                    preimage_reg_val = self.compute_regret_preimage(curr_winning_states_reg_val)
                    preimage = preimage_reg_val.ite(self.manager.addConst(reg_val), preimage)

            else:
                preimage: ADD = self.compute_regret_preimage(curr_winning_states)
            
            next_winning_states = self.compute_min_max_preimage(preimage, valid_human_action_mask=valid_human_action_mask)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                print("State with regret value zero")
                self.gobr_convert_cube_to_state_ADD((next_winning_states.bddInterval(0, 0) & ~self.dfa_handle.goal_latch.bddPattern()).toADD(), action=False, verbose=True)
                print("State with regret values  positive and within budget")
                self.gobr_convert_cube_to_state_ADD(next_winning_states.bddInterval(1, self.budget).toADD(), action=False, verbose=True)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if curr_winning_states.restrict(self.dfa_handle.init_latch & regret_init_latch) != self.manager.plusInfinity():
                    if self.dfa_handle.init_latch & regret_init_latch & curr_winning_states == self.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & regret_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!! The State value is {init_val}")
                    self.rVals = curr_winning_states
                    return preimage if init_val < math.inf else None
                else:
                    print(f"No Regret-Minimizing Strategy Exists!! The State value is {math.inf}")
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states


    def new_frontier_preimage_computation(self, frontier_curr_states: ADD, opt_state_val: ADD, valid_human_action_mask: ADD) -> ADD:
        frontier_preimage: ADD = self.compute_regret_preimage(frontier_curr_states)
        robot_states = frontier_preimage.cofactor(self.tVar_map_sym['robot'])
        frontier_preimage_robots = robot_states.min(opt_state_val)
        next_winning_states_sys = self.symbolic_min_abstract(frontier_preimage_robots, self.rVars)
        
        # proprocess the opt state value ADD
        opt_state_val_human = opt_state_val.cofactor(self.tVar_map_sym['human'])
        # if the human state value is finite keep it else map all inf to -inf
        inf_human_states = opt_state_val_human.bddInterval(math.inf, math.inf).toADD()
        opt_state_val_human = inf_human_states.ite(self.manager.minusInfinity(), opt_state_val_human)

        human_states = frontier_preimage.cofactor(self.tVar_map_sym['human'])
        frontier_preimage_human =  human_states.max(opt_state_val_human)
        frontier_preimage_human = valid_human_action_mask.ite(frontier_preimage_human, self.manager.minusInfinity())
        next_winning_states_env = self.symbolic_max_abstract(frontier_preimage_human, self.rVars)
        frontier_next_states = self.tVar[0].ite(next_winning_states_sys, next_winning_states_env)

        return frontier_next_states


    def TVI_regret_solver(self, verbose: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm For computing regret minimizing strategies. 
        """
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        # initialize goal state with respective regret values
        goal, _ = self.create_goal_nodes_with_regret_values()
        curr_winning_states = goal
        print("Goal States with Regret Values: ", goal.summary())
        # keeps track of optimal state values
        opt_state_val: ADD = goal
        frontier_curr_states = goal

        # intialize the iteration counter
        layer = 0
        regret_init_latch = self.init_latch & self.brVar_map_sym[math.inf]
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        while True:
            print(f"**************************Layer: {layer}**************************")
            frontier_next_states = self.new_frontier_preimage_computation(frontier_curr_states, opt_state_val, valid_human_action_mask)

            # old approach
            preimage: ADD = self.compute_regret_preimage(curr_winning_states)
            next_winning_states = self.compute_min_max_preimage(preimage, valid_human_action_mask=valid_human_action_mask)
            next_winning_states = next_winning_states.min(goal)

            
            # frontier_next_states = self.compute_min_max_preimage(frontier_preimage, valid_human_action_mask=valid_human_action_mask)
            frontier_next_states = frontier_next_states.min(goal)
            
            if opt_state_val.compare(frontier_next_states, 2):
                print("**************************Reached fixpoint**************************")
                if opt_state_val.restrict(self.dfa_handle.init_latch & regret_init_latch) != self.manager.plusInfinity():
                    if self.dfa_handle.init_latch & regret_init_latch & opt_state_val == self.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & regret_init_latch & opt_state_val).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!! The State value is {init_val}")
                    self.test_rVals = opt_state_val
                    # return preimage if init_val < math.inf else None
                else:
                    print(f"No Regret-Minimizing Strategy Exists!! The State value is {math.inf}")
                return None
            
            # any cube who's value is 0 did not change its opt. state value.
            frontier_nodes = opt_state_val - frontier_next_states
            frontier_nodes_01_add = frontier_nodes.bddPattern().toADD()

            # adding debugging step
            if verbose:
                print("Current Frontier States:")
                self.gobr_convert_cube_to_state_ADD(frontier_nodes_01_add, action=False, verbose=True)
            
            # swap the winning states
            opt_state_val = frontier_next_states
            frontier_curr_states = frontier_nodes_01_add.ite(opt_state_val, self.manager.plusInfinity())

            # Old Approach: swap the winning states
            curr_winning_states = next_winning_states

            # update the counter
            layer += 1
    
    def preprocess_gobr_tr(self):
        self.gobr_from_to_relevant_transition_relation = defaultdict(dict) 
        for from_u in range(self.budget + 1): 
            from_u_sym = self.uVar_map_sym[f'u{from_u}']
            for cost in range(0, 2):
                to_u = from_u + cost
                # first find the concerned states
                for sidx, s in enumerate(self.uVar_map[f'u{to_u}']):
                    concerned_states = self.manager.addOne()
                    if s == '1':
                        concerned_states &= (self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()]) & from_u_sym
            
                # now remove the unconcerned states from the transition relation
                gobr_relevant_transition_relation = []
                for tr_dd in self.graph_of_br_tr:
                    gobr_relevant_transition_relation.append(tr_dd & concerned_states)
                self.gobr_from_to_relevant_transition_relation[from_u][to_u] = gobr_relevant_transition_relation
    

    def compute_regret_preimage_over_relevant_tr(self, curr_winning_states: ADD, relevant_tr: List[ADD]) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
                    
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the DFA game state (s, u)
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars + self.brVars, self.prime_latches + self.prime_uVars + self.prime_brVars)
        preimage_subr: ADD = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars + self.prime_brVars, relevant_tr)
        return preimage_subr


    def new_TVI_regret_solver(self, verbose: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm For computing regret minimizing strategies. 
        """
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        # initialize goal state with respective regret values
        goal, _ = self.create_goal_nodes_with_regret_values()
        tic = time.time()
        self.preprocess_gobr_tr()
        toc = time.time()
        print(f"Time to preprocess GoBR TR: {toc - tic} seconds")
        self.test_rVals = None
        # curr_winning_states = goal
        # keeps track of optimal state values
        opt_state_val: ADD = goal
        # intialize the iteration counter
        regret_init_latch = self.init_latch & self.brVar_map_sym[math.inf]
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        while True:
            # we do back up in the topological order given by the utility vars. We start by backing up the highest utility value first
            for u in reversed(range(self.budget + 1)):
                print(f"**************************Working on Utility Group: u{u}**************************")
                goal_u = self.uVar_map_sym[f'u{u}'].ite(goal, self.manager.plusInfinity())
                curr_winning_states_u = self.uVar_map_sym[f'u{u + 1}'].ite(opt_state_val, self.manager.plusInfinity())
                curr_winning_states_u = curr_winning_states_u.min(goal_u)
                # get the appropriate transition relation
                graph_of_br_tr_from_to_u = self.gobr_from_to_relevant_transition_relation[u][u]
                graph_of_br_tr_from_to_u_p1 = self.gobr_from_to_relevant_transition_relation[u][u + 1]
                relevant_tr = []
                for e1, e2 in zip(graph_of_br_tr_from_to_u, graph_of_br_tr_from_to_u_p1):
                    relevant_tr.append(e1 | e2)
                
                # running local value iteration on the current utility layer
                while True:
                    # preimage: ADD = self.compute_regret_preimage(curr_winning_states_u)
                    preimage = self.compute_regret_preimage_over_relevant_tr(curr_winning_states_u, relevant_tr)
                    # mask preimage to remove lower utility values as we will reason over them later.
                    next_winning_states = self.compute_min_max_preimage(preimage, valid_human_action_mask=valid_human_action_mask)
                    next_winning_states = next_winning_states.min(opt_state_val)
                    if curr_winning_states_u.compare(next_winning_states, 2):
                        break
                    curr_winning_states_u = next_winning_states
                opt_state_val = opt_state_val.min(next_winning_states)
            
            # if opt_state_val.compare(frontier_next_states, 2):
            print("**************************Reached fixpoint**************************")
            if opt_state_val.restrict(self.dfa_handle.init_latch & regret_init_latch) != self.manager.plusInfinity():
                if self.dfa_handle.init_latch & regret_init_latch & opt_state_val == self.manager.addZero():
                    init_val: int = 0
                else:
                    init_val: int = list((self.dfa_handle.init_latch & regret_init_latch & opt_state_val).generate_cubes())[0][1]
                print(f"A Winning Strategy Exists!! The State value is {init_val}")
                self.test_rVals = opt_state_val
                # return preimage if init_val < math.inf else None
            else:
                print(f"No Regret-Minimizing Strategy Exists!! The State value is {math.inf}")
            return None
            


    
    def test_pre_image(self):
        # set thigs up
        latch_vars_exist_cube: BDD = reduce(lambda a, b: a & b, self.latches).bddPattern()
        uvars_exist_cube: BDD = reduce(lambda a, b: a & b, self.uVars).bddPattern()
        uVars_add = self.manager.plusInfinity()
        for u in range(self.budget + 1):
            uVars_add = uVars_add.min(self.uVar_map_sym[f'u{u}'].ite(self.manager.addConst(u), self.manager.plusInfinity()))

        # extende the DFA game latches
        graph_of_utility_tr = list(self.transition_relation.values()) #.extend(list(self.uVars_transition_relation.values()))
        graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        # goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['ready l1'] & self.xVar_map_sym['b0 l1'] & self.dfa_handle.goal_latch & self.uVar_map_sym['u5']
        uvars_goal_cube = self.uVar_map_sym['u0'] 
        dfa_game_goal_cube = self.tVar_map_sym['human'] & self.kVar_map_sym['k0'] & self.xVar_map_sym['holding l1'] & self.xVar_map_sym['b0 l0'] & self.dfa_handle.init_latch
        goal_cube = dfa_game_goal_cube & uvars_goal_cube
        # goal state is b0 and l0 and ready l0
        print('Goal state:', goal_cube)

        # first evolve over the DFA
        dfa_preimage = self.preimage_test(From=goal_cube, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation.values()))
        print('Preimage (Only DFA): ', dfa_preimage)
        # self.gou_convert_cube_to_state_ADD(dfa_preimage, human_action=False, robot_action=False)
        
        # then evolve over the game
        print("************Old way of computing preimage:************")
        old_dfa_game_preimage = self.preimage_test(From=uvars_goal_cube,
                                               latches=self.latches + self.uVars,
                                               prime_latches=self.prime_latches + self.prime_uVars,
                                               ts_action=graph_of_utility_tr)
        print('DFA Game Preimage: ', old_dfa_game_preimage)
        self.gou_convert_cube_to_state_ADD(old_dfa_game_preimage, action=False, verbose=True)
        print("************************************************************")

        # parse the goal cube into game and utility cubes
        goal_bdd = goal_cube.bddPattern()
        uls_bdd = goal_bdd.existAbstract(latch_vars_exist_cube)
        dfa_game_bdd = goal_bdd.existAbstract(uvars_exist_cube)

        # try using the min() method to extract utilities ADD
        utls_add = goal_cube.min(uVars_add)

        print("************Old way of computing preimage:************")
        # test this alternate preimage computation
        utls_preimage = self.preimage_test(From=uls_bdd.toADD(),
                                                    latches=self.uVars,
                                                    prime_latches=self.prime_uVars,
                                                    ts_action=list(self.uVars_transition_relation.values()))
        # print('Preimage (Only Utilities): ', utls_preimage)
        # # self.gou_convert_cube_to_state_ADD(dfa_game_utls_preimage, human_action=False, robot_action=False, verbose=True)

        dfa_game_preimage = self.preimage_test(From=dfa_game_bdd.toADD(),
                                               latches=self.latches,
                                               prime_latches=self.prime_latches,
                                               ts_action=list(self.transition_relation.values()))
        # print('Preimage (Only GAME): ', dfa_game_preimage)
        # self.convert_cube_to_state_ADD(dfa_game_preimage, human_action=False, robot_action=False)

        # final preimage
        preimage_full = dfa_game_preimage & utls_preimage
        print('Preimage (GOU GAME): ', preimage_full)
        self.gou_convert_cube_to_state_ADD(preimage_full, action=False, verbose=True)

        assert old_dfa_game_preimage.compare(preimage_full, 2), "[Error]: Preimage computation mismatch between old and new way of computing preimage in GOU game."

    

    def test_br_pre_image_old_approach(self):
        # s = self.tVar_map_sym['robot'] & self.xVar_map_sym['holding l2'] & self.xVar_map_sym['b0 l0'] & self.uVar_map_sym['u2'] & self.kVar_map_sym['k0'] & self.qVar_map_sym[1] #& self.brVar_map_sym[math.inf]
        # sprime = self.tVar_map_sym['human'] & self.xVar_map_sym['ready l2'] & self.xVar_map_sym['b0 l2'] & self.uVar_map_sym['u3'] & self.kVar_map_sym['k0'] & self.qVar_map_sym[1] #& self.brVar_map_sym[math.inf]
        # s2 = self.tVar_map_sym['robot'] & self.xVar_map_sym['ready l2'] & self.xVar_map_sym['b0 l1'] & self.uVar_map_sym['u3'] & self.kVar_map_sym['k0'] & self.qVar_map_sym[2]  # hmove cooperative
        # s3 = self.tVar_map_sym['human'] & self.xVar_map_sym['in-transit b0'] & self.xVar_map_sym['b0 l2'] & self.uVar_map_sym['u1'] & self.kVar_map_sym['k0'] & self.qVar_map_sym[1] # no hmove - adversarial
        state = self.tVar_map_sym['human'] & self.xVar_map_sym['holding l2'] & self.xVar_map_sym['b0 l0'] & self.uVar_map_sym['u1'] & self.kVar_map_sym['k0'] & self.qVar_map_sym[1] #& self.brVar_map_sym[2]
        # br_cube = self.brVar_map_sym[4]#.ite(self.manager.addOne(), self.manager.plusInfinity()) # self.brVar_map_sym[math.inf]
        # br_cube = self.brVar_map_sym[math.inf]
        br_cube = self.brVar_map_sym[math.inf]
        sprime = state

        goal_cube = sprime & br_cube
        print('Goal state:\n', goal_cube)
        # goal = goal_cube.ite(self.manager.addZero(), self.manager.plusInfinity())
        
        print("**********************************************************")
        print("Old method using Compose Operation for BR Cube")
        From = goal_cube.swapVariables(self.gobr_game_latches, self.gobr_game_prime_latches)
        preimage_br = From.vectorCompose(self.prime_brVars, list(self.brVars_transition_relation.values()))
        print("Evolved over BR TR: \n", preimage_br)
        # self.gobr_convert_cube_to_state_ADD(preimage_br, robot_action=False, human_action=False, verbose=True)

        # then evolve over the game
        goal_cube = goal_cube.swapVariables(self.brVars, self.prime_brVars) # as the method below does not swap the br vars
        preimage_su = self.compute_preimage(goal_cube)
        self.gou_convert_cube_to_state_ADD(preimage_su, action=False, verbose=True)
        robot_states = preimage_su & self.tVar_map_sym['robot']
        humans_states = preimage_su & self.tVar_map_sym['human']
        preimage_full_2 = self.manager.addZero()
        preimage_full_1 = self.manager.addZero()
        if not robot_states.isZero():
            preimage_full_1 = preimage_br & preimage_su
        if not humans_states.isZero():
            preimage_full_2 = goal_cube.swapVariables(self.gobr_game_latches, self.gobr_game_prime_latches) & preimage_su 
        
        preimage_full = preimage_full_1 | preimage_full_2
        # preimage_full = preimage_su
        print_cube = preimage_full #& goal_cube.swapVariables(self.gobr_game_latches, self.gobr_game_prime_latches)
        print('Preimage over GoBR TR: ', print_cube)
        # self.gobr_convert_full_cube_to_state_ADD(print_cube, robot_action=False, verbose=True)
        # print('DFA Game Preimage: ', dfa_game_preimage)
        self.gobr_convert_cube_to_state_ADD(print_cube, action=False, verbose=True)

        # here use the monolithic valid full GoBR TR to compute preimage of BR
        print("**********************************************************")
        print("New method using AND Operation wihtout computing preimage of BR Cube")
        # goal_cube = goal_cube.swapVariables(self.brVars, self.prime_brVars) # as the method below does not swap the br vars
        new_preimage_su = self.compute_preimage(goal_cube)

        # TODO: Should this & or should it be restrict?
        new_preimage_full = new_preimage_su & self.monolithic_valid_sabr_prime_br_trns
        print('Preimage over GoBR TR: \n', new_preimage_full)
        # remove dependency on prime br vars
        tmp_preimage = self.manager.addZero()
        for br in self.brVar_map_sym:
            tmp_preimage |= new_preimage_full.cofactor(self.brVar_map_sym[br].swapVariables(self.brVars, self.prime_brVars))
        
        print('Preimage over GoBR TR (no prime vars): \n', tmp_preimage)
        self.gobr_convert_cube_to_state_ADD(tmp_preimage, action=False, verbose=True)
        # assert preimage_full.compare(new_preimage_full, 2), "[Error]: Preimage computation mismatch between old and new way of computing preimage in GoBR game."
        print("**********************************************************")