"""
 In this script, we will build the DFA Game, copute regret-minimizing strategies in a symbolic and partitioned manner.
 1. Build Game abstarction in a compositional manner.
 2. Call SPOT or MONA to build DFA from LTL/LTlf formula.
 3. Construct the DFA Game
 3. Construct Graph of Utility and compute Min-Min Value Iteration
    3.1 Compute BEst-response for every system strategy
 4. Construct Graph of Best-response and compute Min-Max Value Iteration
"""
import math

from functools import reduce
from typing import List, Dict
from collections import defaultdict

from bidict import bidict
from cudd import Cudd, ADD, BDD

from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame


class SymbolicPartitionedRegretDFAGame(SymbolicPartitionedDFAGame):
    """
     This class extends the SymbolicPartitionedDFAGame class to compute regret-minimizing strategies.
    """
    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 budget: int,
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False):
        self.budget: int = budget
        self.uVars: List[ADD] = []
        self.prime_uVars: List[ADD] = []
        self.uVar_map: List[ADD] = bidict({}) 
        self.uVar_map_sym: List[ADD] = bidict({}) 
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(boxes, locs, ratio, init, goal, formula, restricted_human_locs, ltlf_flag=ltlf_flag, enable_reordering=enable_reordering)
        self.states_per_cost: Dict[int, ADD] = defaultdict(lambda: self.manager.addZero())
        self.uVars_transition_relation = None


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
    
    
    def create_all_prime_boolean_state_vars(self):
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
        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars


    def create_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        uVar_size = math.ceil(math.log2(self.budget + 1)) # +1 here to account for 0-bit vector
        # create an additional boolean var to skip the 0-vector latch
        uVar_size = uVar_size + 1 if pow(2, uVar_size) == self.budget + 1 else uVar_size 
        uVars: List[ADD] = [self.manager.addVar(u + varsize, 'u' + str(u)) for u in range(uVar_size)]
        return uVars
    

    def create_prime_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        prime_uVars: List[ADD] = [self.manager.addVar(u + varsize, 'pu' + str(u)) for u in range(len(self.uVars))]
        return prime_uVars


    def create_utiltity_var_map(self):
        """
         Small function to create symbolic maps for the uVar_map. 
         TODO: Update cube_to_add method to cubestring_to_add for better clarity.
        """
        for u in range(self.budget + 1):
            ubit_str = f"{u + 1:0{len(self.uVars)}b}"
            self.uVar_map[f'u{u}'] = ubit_str
            self.uVar_map_sym[f'u{u}'] = self.cube_to_add(ubit_str, self.uVars)
    

    def get_dd_per_cost(self):
        """
         A helper function that takes in the ADD weight abd return a vector of 0-1 ADD per cost.
        """
        # min_val: int = self.weight.findMin() # must be equal to 0
        # max_val: int = self.weight.findMax() # must be equal to infinity
        min_val: int = 0
        max_val: int = 1
        
        for val in range(min_val, max_val + 1, 1):
            self.states_per_cost[val] |= self.weight.bddInterval(val, val).toADD()
    

    def create_utlity_transition_relation(self):
        """
         Create the transition relation for utility variables.
        """
        self.uVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.uVars}
        self.get_dd_per_cost()
        valid_state_costs = [1, 0]
        for u in range(self.budget + 1):
            for state_cost in valid_state_costs:
                if u + state_cost <= self.budget:
                    uConf_cube = self.uVar_map_sym[f'u{u}']
                    uConf_prime_cube = self.uVar_map[f'u{u + state_cost}']
                    transition_cube = uConf_cube & self.states_per_cost[state_cost]
                    
                    for sidx, s in enumerate(uConf_prime_cube):
                        if s == '1':
                            self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube
    

    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, dfa_flag: bool = True, robot_action: bool = False, human_action: bool = False) -> None:
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
        if robot_action:
            relevant_vars.extend(self.oVars) # robot action vars (oVars)
        if human_action:
            relevant_vars.extend(self.iVars) # env action vars (iVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])
        start_ivar_idx, end_ivar_idx = self.manager.addVariables().index(self.iVars[0]), self.manager.addVariables().index(self.iVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.oVars + self.iVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.oVars + self.iVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.oVars + self.iVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.oVars + self.iVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.oVars + self.iVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.oVars + self.iVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.oVars + self.iVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        for cube, val in cubes:
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
                print(f"[(({self.tVar_map.inv[tConf_exist_str]}, \
                           {self.kVar_map.inv[kConf_exist_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states}), \
                           {self.dfa_handle.qVar_map.inv[qConf_exist_str]}, {self.uVar_map.inv[uConf_exist_str]}), {val}]")
                states_action_pairs.append([
                    (((self.tVar_map.inv[tConf_exist_str],
                       self.kVar_map.inv[kConf_exist_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_exist_str], self.uVar_map.inv[uConf_exist_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if robot_action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    rAction_str = self.rAction_map_sym.inv[self.cube_to_add(oCube_str, self.oVars)]
                except KeyError:
                    continue
            if human_action:
                iCube_str = cube.bddPattern().cubeString()[start_ivar_idx:end_ivar_idx + 1].replace('-', '')
                try:
                    eAction_str = self.eAction_map_sym.inv[self.cube_to_add(iCube_str, self.iVars)]
                except KeyError:
                    continue
            if robot_action or human_action:    
                action = ", ".join(filter(None, [rAction_str if robot_action else None, eAction_str if human_action else None]))
                print(f"    -- Actions: ({action})")
        
        return states_action_pairs


    
    def create_transition_relation(self):
        super().create_transition_relation()

        # now create utility transition relation
        self.create_utlity_transition_relation()
    
    def test_pre_image(self):
        # extende the DFA game latches
        graph_of_utility_tr = list(self.transition_relation.values()) #.extend(list(self.uVars_transition_relation.values()))
        graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['ready l1'] & self.xVar_map_sym['b0 l1'] & self.dfa_handle.goal_latch #& self.uVar_map_sym['u2']
        # goal state is b0 and l0 and ready l0
        print('Goal state:', goal_cube)

        # first evolve over the DFA
        dfa_preimage = self.preimage_test(From=goal_cube, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation.values()))
        print('DFA Preimage: ', dfa_preimage)
        # self.convert_cube_to_state_ADD(dfa_preimage, human_action=False, robot_action=False)
        
        # then evolve over the game
        dfa_game_preimage = self.preimage_test(From=dfa_preimage,
                                               latches=self.latches + self.uVars,
                                               prime_latches=self.prime_latches + self.prime_uVars,
                                               ts_action=graph_of_utility_tr)
        print('DFA Game Preimage: ', dfa_game_preimage)
        self.convert_cube_to_state_ADD(dfa_game_preimage, human_action=False, robot_action=False)
    
                    