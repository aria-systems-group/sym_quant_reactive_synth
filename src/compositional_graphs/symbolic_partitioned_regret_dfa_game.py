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
from typing import List, Dict, Tuple
from collections import defaultdict

from bidict import bidict
from tabulate import tabulate

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
        # store s a_s s' transition relation for graph of utility
        self.monolithic_valid_full_gou_trns: ADD = self.manager.addZero()


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
    

    def set_init_latch(self) -> ADD:
        """
         Ovveride the base method. In Graph of Utility, the initial state also includes the utility variable set to 0.
        """
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
        # budget + 1 is a state to represent budget exceeded; it will be a sink state.
        for u in range(self.budget + 2):
            ubit_str = f"{u + 1:0{len(self.uVars)}b}"
            self.uVar_map[f'u{u}'] = ubit_str
            self.uVar_map_sym[f'u{u}'] = self.cube_to_add(ubit_str, self.uVars)
    

    def get_states_per_cost(self):
        """
         A helper function that takes in the ADD weight abd return a vector of 0-1 ADD per cost.
        """
        # min_val: int = self.weight.findMin() # must be equal to 0
        # max_val: int = self.weight.findMax() # must be equal to infinity
        min_val: int = 0
        max_val: int = 1
        
        for val in range(min_val, max_val + 1, 1):
            # if val != 0:
                # self.states_per_cost[val] |= self.weight.bddInterval(val, val).toADD() & ~self.init_latch
            # else:
            self.states_per_cost[val] |= self.weight.bddInterval(val, val).toADD()
        
        # manually add the init state to cost 0
        # self.states_per_cost[1] |= self.init_latch
    

    def create_utlity_transition_relation(self):
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
                # if u + state_cost <= self.budget:
                uConf_prime_cube_str = self.uVar_map[f'u{prime_u_val}']
                for sidx, s in enumerate(uConf_prime_cube_str):
                    if s == '1':
                        self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube
                # else:
                    # uConf_prime_cube_str = self.uVar_map[f'u{self.budget + 1}']
                    # for sidx, s in enumerate(uConf_prime_cube_str):
                    #     if s == '1':
                    #         self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube
                
                # create s a_s s' monolithic ADD which we will use later for alternate best-response computation
                self.monolithic_valid_full_gou_trns |= transition_cube & self.monolithic_valid_full_dfa_game_trns & \
                      self.uVar_map_sym[f'u{prime_u_val}'].swapVariables(self.uVars, self.prime_uVars)
        
        # add self-loop for the sink state budget + 1
        for sidx, s in enumerate(self.uVar_map[f'u{self.budget + 1}']):
            if s == '1':
                self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |=  self.uVar_map_sym[f'u{self.budget + 1}']
    

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
                print(f"[(({self.tVar_map.inv[tConf_exist_str]}, {self.kVar_map.inv[kConf_exist_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states}), {self.dfa_handle.qVar_map.inv[qConf_exist_str]}, {self.uVar_map.inv[uConf_exist_str]}), {val}]")
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


    def convert_full_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, dfa_flag: bool = True, robot_action: bool = False, verbose: bool = False) -> None:
        """
        Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True.
         If you want to print the human action as well, set human_action to True.

         Here the input dd is assumed to be a fully defined cube (latches as well prime latches).
        """
        gou_game_latches = self.latches + self.qVars + self.uVars
        gou_game_prime_latches = self.prime_latches + self.prime_qVars + self.prime_uVars
        relevant_vars = [] + self.uVars + self.prime_uVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes tVars, kVars, pVars and bVars
            relevant_vars.extend(self.prime_latches) # includes tVars, kVars, pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.qVars) # includes qVars
            relevant_vars.extend(self.prime_qVars) # includes prime qVars
        if robot_action:
            relevant_vars.extend(self.oVars) # robot action vars (oVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])

        # create abstraction cubes
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.oVars + self.iVars + gou_game_prime_latches)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.oVars + self.iVars + gou_game_prime_latches)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.oVars + self.iVars + gou_game_prime_latches)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.oVars + self.iVars + gou_game_prime_latches)
        # create existential abstraction cubes - rConf
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.oVars + self.iVars + gou_game_prime_latches) 

        # create prime abstraction cubes
        prime_tConf_exist_cube = reduce(lambda a, b: a & b, self.prime_xVars + self.prime_qVars + self.prime_uVars + self.oVars + self.iVars + gou_game_latches)
        prime_kConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_xVars[len(self.prime_kVars):] + self.prime_qVars + self.prime_uVars + self.oVars + self.iVars + gou_game_latches)
        prime_qConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_uVars + self.oVars + self.iVars + gou_game_latches)
        prime_uConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars + self.prime_qVars + self.oVars + self.iVars + gou_game_latches)

        # create PRIME existential abstraction cube - rConf 
        prime_rConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_qVars + self.prime_uVars + self.prime_xVars[len(self.prime_kVars)+len(self.prime_pVars):] + self.oVars + self.iVars + gou_game_latches)

        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        prime_bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.oVars + self.iVars + gou_game_prime_latches)
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.prime_qVars + self.prime_uVars + self.oVars + self.iVars + gou_game_latches)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.oVars + self.iVars + gou_game_prime_latches) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.prime_qVars + self.prime_uVars + self.oVars + self.iVars + gou_game_latches) & reduce(lambda x, y: x & y, self.prime_bVars_cubes[:bidx] + self.prime_bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        state_action_prime_pairs = []
        for cube, val in cubes:
            state = None
            prime_state = None
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
            if robot_action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    rAction_str = self.rAction_map_sym.inv[self.cube_to_add(oCube_str, self.oVars)]
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
            # print(state, f'--({rAction_str})-->', prime_state, sep="      ")
            state_action_prime_pairs.append((state, rAction_str, prime_state))
        
        if verbose:
            print(tabulate(state_action_prime_pairs, headers=['state', 'robot action', 'prime state']))
            
        return states_action_pairs


    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.latches + self.qVars + self.uVars, self.prime_latches + self.prime_qVars + self.prime_uVars)
        
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))

        # then evolve over the game
        preimage = dfa_preimage.vectorCompose(self.prime_latches + self.prime_uVars, self.graph_of_utility_tr)

        return preimage
    

    def solve(self, verbose: bool = False, cooperative_game: bool = False) -> Dict[str, ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values()) #.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        
        return super().solve(verbose=verbose, cooperative_game=cooperative_game)
    

    def get_next_state(self, turn: str, curr_state_exp: List[str], act_name: str, **kwargs) -> Tuple[ADD, str]:
        # get the next state in the game in explicit form
        curr_game_state = list(curr_state_exp[0][0][0][0])
        curr_utl_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        # act_name = ''
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


    
    def create_transition_relation(self):
        # creates game transition relation first and then dfa transition relation
        super().create_transition_relation()

        # now create utility transition relation
        self.create_utlity_transition_relation()

        # print Sys transitions for sanity checking
        # self.convert_full_cube_to_state_ADD(dd=self.monolithic_valid_full_gou_trns, robot_action=True, verbose=True)
    
    def test_pre_image(self):
        # extende the DFA game latches
        graph_of_utility_tr = list(self.transition_relation.values()) #.extend(list(self.uVars_transition_relation.values()))
        graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['ready l1'] & self.xVar_map_sym['b0 l1'] & self.dfa_handle.goal_latch & self.uVar_map_sym['u5']
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
    
                    