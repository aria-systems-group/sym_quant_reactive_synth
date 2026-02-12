"""
This script implements a pure turn-based game between robot and human. 

In frankadynamic.py, we have the robot and humnan move in concurrent fashion, i.e., the successor state is determined by both
 the robot action and human action. Thi

Here we implement a pure turn-based game, where the robot moves first, then the human moves. After each robot turns, we evolve a Env state. 
From the Env state, we evolve to Sys state by applying all possible human moves.
"""
import sys
import time
import math

from typing import List, Dict, Tuple, Set, Union, Optional
from functools import reduce
from itertools import product
from collections import defaultdict

from bidict import bidict
from tabulate import tabulate

from cudd import Cudd, ADD, BDD, REORDER_GROUP_SIFT_CONV
from src.compositional_graphs.logger import CustomLogger


class FrankaWorldDynamicRatioTurnBased():

    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, restricted_human_locs: List[int],
                 restricted_human_boxes: List[int],
                 enable_reordering: bool = False,
                 only_reachable_states: bool = False):
        self.boxes: int = boxes
        self.locs: int = locs
        self.ratio: int = ratio
        self.human_locs: List[int] = restricted_human_locs
        self.human_boxes: List[int] = restricted_human_boxes
        self.only_reachable_states = only_reachable_states
        self.restricted_human_locs: Set[int] = set([0, self.locs] + [*range(1, self.locs + 1)]) - set(self.human_locs) 
        self.misc_preds = ['ready', 'in-transit', 'in-transfer' 'to-obj', 'holding']
        self.robot_actions: List[str] = ['transit', 'transfer', 'grasp', 'release']
        self.init = init
        self.goal = goal
        self.manager: Cudd = Cudd(maxMem=16000000000)
        # Predicate to Str maps - needed for lookup of the states corresponding to cubesstring
        self.pVar_map = bidict({})
        self.kVar_map = bidict({})
        self.xVar_map = dict()
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}
        self.tVar_map = bidict({'robot': '1', 'human': '0'})  # fixed turn variable map

        # Predicate to Cube maps - needed for symbolic operations; also avoid multiple calls to cube_to_add()
        self.xVar_map_sym = dict()
        self.bVar_map_sym = dict()
        self.prime_xVar_map_sym = dict()
        self.prime_bVar_map_sym = dict()

        # create latches - tVars + kVars + pVars + bVars
        self.create_all_boolean_state_vars_and_maps()
        self.set_latches()
        
        # create prime latches - prime tVars + prime kVars + prime pVars + prime bVars
        self.create_all_prime_boolean_state_vars_and_maps()
        self.set_prime_latches()
        
        # different from frankaworld, we need a turn variable map
        self.tVar_map_sym = bidict({'robot': self.cube_to_add(self.tVar_map['robot'], self.tVar),
                                    'human': self.cube_to_add(self.tVar_map['human'], self.tVar)})
        
        self.prime_tVar_map_sym = bidict({'robot': self.cube_to_add(self.tVar_map['robot'], self.prime_tVar),
                                          'human': self.cube_to_add(self.tVar_map['human'], self.prime_tVar)})
        
        # now that the maps are initialized we create init and goal states
        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()
        self.states_per_cost: Dict[int, ADD] = defaultdict(lambda: self.manager.bddZero())

        # monolithic transition relation
        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
        self.ts_bdd_transition_fun_list: List[List[BDD]] = []

        # create robot and human action vars and maps
        self.rVars: List[ADD] = self.create_action_vars()
        self.human_action: List[str] = ['hmove']
        self.action_map = bidict({})
        self.relevant_robot_actions = None
        self.relevant_robot_actions_sym: ADD = defaultdict(lambda: self.manager.addZero()) 
        self.create_action_map()
        self.action_map_sym = bidict({k: self.cube_to_add(v, self.rVars) for k, v in self.action_map.items()})
        self.rVars_cube = reduce(lambda x, y: x & y, self.rVars)
        self.action_cube_list: List[ADD] = list(self.action_map_sym.values())
        
        self.weight_dict: Dict[str, int] = {'transit': 1, 'transfer': 1, 'grasp': 1, 'release': 1}
        self.symbolic_weight_dict: Dict[str, ADD] = defaultdict(lambda: self.manager.addOne())
        self.create_sym_weight_dict()

        # precompute cubes of valid Robot and Env actions - needed for synthesis
        self.env_action_cube_list = []
        self.sys_action_cube_list = []
        for act_str, act_dd in self.action_map_sym.items():
            if act_str.startswith('hmove'):
                if act_str.startswith('hmove noop') and len(act_str.split(' ')) > 2:
                    continue
                self.env_action_cube_list.append(act_dd)
            else:
                self.sys_action_cube_list.append(act_dd)
        
        self.env_action_cube_list_bdd: List[BDD] = [act_dd.bddPattern() for act_dd in self.env_action_cube_list]
        self.sys_action_cube_list_bdd: List[BDD] = [act_dd.bddPattern() for act_dd in self.sys_action_cube_list]
        self.miscellanoues_helper_stuff()

        # create logger instance
        self.logger = CustomLogger()
        

        if enable_reordering:
            self.manager.autodynEnable()
    
    @property
    def human_boxes(self):
        return self._human_boxes
    
    @property
    def human_locs(self):
        return self._human_locs

    @human_boxes.setter
    def human_boxes(self, hboxes: List[int]):
        assert set(hboxes).issubset(set(range(self.boxes))), "[Error] Human boxes should be a subset of all boxes."
        assert len(hboxes) > 0 , "[Error] Human boxes cannot be empty. If you need enforce no human movement set ratio vars to 0."
        self._human_boxes = hboxes  

    @human_locs.setter
    def human_locs(self, hlocs: List[int]):
        assert set(hlocs).issubset(set(range(1, self.locs + 1))), "[Error] Human locs should be a subset of all locs."
        self._human_locs = hlocs  
    

    def create_all_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the FrankaDynamic Turn-Based Game.
          1. turn variables - tVars
          2. ratio variables - kVars
          3. predicate variables - pVars
          4. box predicate variables - bVars
        """
        offset = self.manager.size()
        self.tVar: List[ADD] = [self.manager.addVar(offset, 't0')]
        self.kVars: List[ADD] = self.create_ratio_vars()
        self.pVars, self.bVars = self.create_latches()
        self.create_all_maps()
        self.create_all_sym_maps()
    

    def create_all_prime_boolean_state_vars_and_maps(self):
        """
         The main method that creates all primed version of the boolean variables for the FrankaDynamic Turn-Based Game.
          1. prime turn variables - tVars
          2. prime ratio variables - kVars
          3. prime predicate variables - pVars
          4. prime box predicate variables - bVars
        """
        offset = self.manager.size()
        self.prime_tVar: List[ADD] = [self.manager.addVar(offset, "pt0")]
        self.prime_kVars: List[ADD] = self.create_prime_ratio_vars()
        self.prime_pVars, self.prime_bVars = self.create_prime_latches()
        self.create_all_sym_maps(prime=True)
    

    def log_game_details(self) -> Dict[str, int]:
        sys_states, env_states = self.get_number_of_states(False)
        abs_dict = {
            'total_latches': len(self.latches) + len(self.prime_latches) + len(self.rVars),
            'latches': len(self.latches),
            'prime_latches': len(self.prime_latches),
            'action_vars': len(self.rVars),
            'turn_vars': len(self.tVar),
            'ratio_vars': len(self.kVars),
            'state_vars': len(self.pVars) + len(reduce(lambda x, y: x + y, self.bVars)),
            'total_states': sys_states + env_states,
            'sys_states': sys_states,
            'env_states': env_states,
            'num_opt_sVals': self.comp_winning_states.countLeaves()
            }
        return abs_dict
    

    def create_latches(self) -> Tuple[List[ADD], List[ADD], List[ADD]]:
        """
         For ready; to-obj; and holding we create a dedicated set of latches
         For every on b predicate we will create a set of latch for all boxes and for all locs. 
        """
        # create holding, ready, to-obj vars
        pVars = self.create_ready_holding_to_obj_vars()
        
        loc_var_size = math.ceil(math.log2(self.locs + 1))  # the +1 is for end-effector (ee) location
        # create an additional boolean var to skip the 0-vector latch
        loc_var_size = loc_var_size + 1 if pow(2, loc_var_size) == self.locs + 1 else loc_var_size 
        
        bVars: List[List[ADD]] = []
        for b in range(self.boxes):
            varsize = self.manager.size()
            bVars.append([self.manager.addVar(j + varsize, f'b{b}{j}') for j in range(loc_var_size)])

        return pVars, bVars

    
    def create_ratio_vars(self) -> List[ADD]:
        """
         Given ratio K (Human can intervene K times repeatedly before robot can force its move), we create log(K) boolean variables.
        """
        varsize = self.manager.size()
        kVar_size = math.ceil(math.log2(self.ratio + 1)) # +1 here to accrount k0 and up to k-ratio
        # create an additional boolean var to skip the 0-vector latch
        kVar_size = kVar_size + 1 if pow(2, kVar_size) == self.ratio + 1 else kVar_size 
        kVars: List[ADD] = [self.manager.addVar(k + varsize, 'k' + str(k)) for k in range(kVar_size)]
        return kVars
        

    def create_ready_holding_to_obj_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        # num. of preds = ready x |locs| + to-obj x |boxes| + holding x |locs| + 1 (to account for l0 being end effector loc)
        # additional preds: in-transit x |locs + 1| x |boxes| + in-transfer x |locs| x |locs  - 1|
        num_of_preds = 2*self.locs + self.boxes + 2 + 1 # +1 for ready-else state
        num_of_preds += (self.locs + 1) * self.boxes # in-transit preds +1 for the else location
        num_of_preds += self.locs * (self.locs - 1) # in-transfer preds
        vars_size: int = math.ceil(math.log2(num_of_preds))
        vars_size = vars_size + 1 if pow(2, vars_size) == num_of_preds else vars_size
        Vars: List[ADD] = [self.manager.addVar(k + varsize, 'p' + str(k)) for k in range(vars_size)]
        return Vars

    
    def create_prime_latches(self) -> Tuple[List[ADD], List[ADD]]:
        """
         Create a copy of prime variables for the latches.
        """
        varsize = self.manager.size()
        pVars_prime: List[ADD] = [self.manager.addVar(k + varsize, 'pp' + str(k)) for k in range(len(self.pVars))]
        bVars_prime: List[List[ADD]] = []
        for b_idx, b in enumerate(self.bVars):
            varsize = self.manager.size()
            bVars_prime.append([self.manager.addVar(j + varsize, f'pb{b_idx}{j}') for j in range(len(b))])

        return pVars_prime, bVars_prime
    
    def create_prime_ratio_vars(self) -> List[ADD]:
        """
         Create a copy of prime variables for the ratio latches.
        """
        varsize = self.manager.size()
        kVars_prime: List[ADD] = [self.manager.addVar(k + varsize, 'pk' + str(k)) for k in range(len(self.kVars))]
        return kVars_prime
    
    def create_loc_empty_constraint(self):
        """
         A function that create cubes that enforce that a location l is empty.
        """
        for l in range(1, self.locs + 1):
            l_empty_cube = self.manager.addOne()
            for b in range(self.boxes):
                l_empty_cube &= ~self.xVar_map_sym[f'b{b} l{l}']
            self.locs_empty_constraints[f'l{l}'] = l_empty_cube
    

    def create_xVar_map(self):
        """
         Create a map for the predicate variables.
        """
        # for misc preds ready and holding we create all locs.
        offset = 0
        self.pVar_map[f'ready l{self.locs + 1}'] = f"{1:0{len(self.pVars)}b}"
        self.xVar_map[f'ready l{self.locs + 1}'] = f"{1:0{len(self.pVars)}b}"
        offset += 1
        for pidx, pred in enumerate(['ready', 'holding']):
            # +1 to include end-effector location
            for loc in range(1, self.locs + 1):
                bit_str = f"{offset + loc:0{len(self.pVars)}b}"
                self.xVar_map[pred + ' l' + str(loc)] = bit_str
                self.pVar_map[pred + ' l' + str(loc)] = bit_str
            offset += self.locs
        
        offset = 2*(self.locs) + 2 # 1 for the offset from the 0-vector; another 1 for the ready-else state
        # for misc pred to-obj we create all boxes
        for b in range(self.boxes):
            bit_str = f"{b + offset:0{len(self.pVars)}b}"
            self.xVar_map['to-obj b' + str(b)] = bit_str
            self.pVar_map['to-obj b' + str(b)] = bit_str
        
        # create preds for in-transit
        offset += self.boxes
        for b in range(self.boxes):
            # +2 to account for else location
            for loc in range(1, self.locs + 2):
                # -1 because loc starts from 1
                bit_str = f"{offset + loc - 1:0{len(self.pVars)}b}"
                self.xVar_map[f'in-transit l{loc} b{b}'] = bit_str
                self.pVar_map[f'in-transit l{loc} b{b}'] = bit_str
            offset += self.locs + 1
        
        # create preds for in-transfer
        for from_loc in range(1, self.locs + 1):
            for to_loc in range(1, self.locs + 1):
                if from_loc == to_loc:
                    continue
                bit_str = f"{offset + to_loc - 1:0{len(self.pVars)}b}"
                self.xVar_map[f'in-transfer l{from_loc} l{to_loc}'] = bit_str
                self.pVar_map[f'in-transfer l{from_loc} l{to_loc}'] = bit_str
            offset += self.locs

        # for each boxes we create |locs| boolean vars
        for b in range(self.boxes):
            for l in range(self.locs + 1):
                bit_str = f"{l + 1:0{len(self.bVars[b])}b}"
                self.xVar_map['b' + str(b) + ' l' + str(l)] = bit_str
                self.bVars_map[b]['b' + str(b) + ' l' + str(l)] = bit_str
    
    def create_ratio_var_map(self):
        """
         Create a map for the ratio variables.
        """
        offset = 1
        for r in range(self.ratio + 1):
            bit_str = f"{r + offset:0{len(self.kVars)}b}"
            self.kVar_map[f'k{r}'] = bit_str
    

    def create_all_maps(self):
        """
         A tiny method to create all maps for the variables. We only create maps for non-primed boolean variables.
        """
        self.create_xVar_map()
        self.create_ratio_var_map()
    

    def create_all_sym_maps(self, prime: bool = False):
        """
         A tiny method to create all symbolic maps for the variables. We only create symbolic maps for non-primed boolean variables.
        """
        # more bookeeping stuff
        if prime:
            self.prime_kVar_map_sym = bidict({r: self.cube_to_add(v, self.prime_kVars) for r, v in self.kVar_map.items()})
        else:
            self.kVar_map_sym = bidict({r: self.cube_to_add(v, self.kVars) for r, v in self.kVar_map.items()})
        self.create_symbolic_maps(prime=prime)
    

    def miscellanoues_helper_stuff(self):
        """
         Some miscellanoues helper stuff that are used in multiple places.
        """
        # need these cubes for printing states from cubes
        self.bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.bVars]
        self.prime_bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.prime_bVars]
        # create relevant env and robot actions; boxes
        self.relevant_box_preds_sym = defaultdict(lambda: self.manager.addZero())
        self.create_relevant_box_predicates()
        self.monolithic_relevant_box_preds: ADD = reduce(lambda x, y: x & y, self.relevant_box_preds_sym.values())
        self.monolithic_valid_state_robot_actions: ADD = self.manager.addZero()
        self.monolithic_valid_state_robot_actions_prime_state: ADD = self.manager.addZero()
        self.monolithic_valid_state_human_actions_prime_state: ADD = self.manager.addZero()
        # set of all valid state-action-next_state triples for Sys and Env players
        self.monolithic_state_action_prime_state: ADD = None
        
        # state invariance constraint - end-effector empty cube - used in transit and grasp actions
        self.ee_empty_cube: ADD = self.create_ee_empty_cube()
        self.create_monoltithic_box_conf_cube()
        self.create_hmove_not_b()
        self.create_valid_state_constraints()
        self.preprocess_monolithic_valid_state_robot_actions()
        
        self.locs_empty_constraints = defaultdict(lambda: self.manager.addZero())
        self.create_loc_empty_constraint()
        self.kVal_cube = reduce(lambda x, y: x | y, self.kVar_map_sym.values())
    

    def create_action_vars(self) -> List[ADD]:
        """
         Create a single method wehre we create robot action and env actions using the same of variables.
         
         This will lead to savings in the # of boolean vars needed. 
         This approach will require log(num_robot_actions + num_env_actions) boolean vars.
         Old approach required log(num_robot_actions) + log(num_env_actions) boolean vars.
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes + self.locs + 2 #+ 1 # +1 to offset the 0-vector
        num_of_eActions = len(self.human_boxes) * len(self.human_locs) + 1 # +1 is for no-op action
        rVars_size = math.ceil(math.log2(num_of_rActions + num_of_eActions)) #if (num_of_rActions > 1 else 1
        rVars: List[ADD] =  [self.manager.addVar(r + varsize , 'r' + str(r)) for r in range(rVars_size)]
        return rVars
    
    def create_symbolic_maps(self, prime: bool = False):
        """
         Small function to create symbolic maps for the xVar_map, rAction_map and eAction_map
        """
        for k, v in self.pVar_map.items():
            if prime:
                self.prime_xVar_map_sym[k] = self.cube_to_add(v, self.prime_pVars)
            else:
                self.xVar_map_sym[k] = self.cube_to_add(v, self.pVars)
        
        for bidx, d in self.bVars_map.items():
            for k, v in d.items():
                if prime:
                    self.prime_xVar_map_sym[k] = self.cube_to_add(v, self.prime_bVars[bidx])
                    self.prime_bVar_map_sym[k] = self.cube_to_add(v, self.prime_bVars[bidx])
                else:
                    self.xVar_map_sym[k] = self.cube_to_add(v, self.bVars[bidx])
                    self.bVar_map_sym[k] = self.cube_to_add(v, self.bVars[bidx])
    
    def create_action_map(self) -> None:
        # robot actions - testing with no offset
        self.relevant_robot_actions: ADD = self.manager.addZero()
        for ract in self.robot_actions:
            if ract == 'transit':
                for b in range(self.boxes):
                    act_str = f'{ract} b{b}'
                    rbit_str = f"{b:0{len(self.rVars)}b}"
                    self.action_map[act_str] = rbit_str
                    self.relevant_robot_actions_sym[act_str] |= self.cube_to_add(rbit_str, self.rVars)
                    self.relevant_robot_actions |= self.cube_to_add(rbit_str, self.rVars)
            elif ract == 'transfer':
                for l in range(1, self.locs + 1):
                    act_str = f'{ract} l{l}'
                    # -1 offset the loc 0 str
                    rbit_str = f"{self.boxes + l -1:0{len(self.rVars)}b}"
                    self.action_map[act_str] = rbit_str
                    self.relevant_robot_actions_sym[act_str] |= self.cube_to_add(rbit_str, self.rVars)
                    self.relevant_robot_actions |= self.cube_to_add(rbit_str, self.rVars)
        rbit_str = f"{self.boxes + self.locs:0{len(self.rVars)}b}"
        self.action_map['grasp'] = rbit_str
        self.relevant_robot_actions_sym['grasp'] |= self.cube_to_add(rbit_str, self.rVars)
        self.relevant_robot_actions |= self.cube_to_add(rbit_str, self.rVars)
        rbit_str = f"{self.boxes + self.locs + 1:0{len(self.rVars)}b}"
        self.action_map['release'] = rbit_str
        self.relevant_robot_actions_sym['release'] |= self.cube_to_add(rbit_str, self.rVars)
        self.relevant_robot_actions |= self.cube_to_add(rbit_str, self.rVars)

        # human actions
        # Add a no-op action for the human
        offset = len(self.action_map.keys())
        self.action_map[f'{self.human_action[0]} noop'] = f"{offset:0{len(self.rVars)}b}"
        offset += 1
        for b in self.human_boxes:
            for l in self.human_locs:
                act_str = f'{self.human_action[0]} b{b} l{l}'
                hbit_str = f"{offset:0{len(self.rVars)}b}"
                self.action_map[act_str] = hbit_str
                offset += 1
        
        # the rest of them map to human noop as well.
        for i in range(offset, pow(2, len(self.rVars))):
            hbit_str = f"{offset:0{len(self.rVars)}b}"
            self.action_map[f'{self.human_action[0]} noop {i}'] = hbit_str
            offset += 1


    def create_sym_weight_dict(self) -> None:
        for ract, dd in self.action_map_sym.items():
            if ract.startswith('hmove'):
                continue
            # extract the name
            act_name: str = ract.split(' ')[0]
            w = self.weight_dict[act_name]
            self.symbolic_weight_dict[ract] = dd.ite(self.manager.addConst(w), self.manager.addZero()) & self.tVar_map_sym['robot']
        
        self.weight = reduce(lambda x, y: x | y, self.symbolic_weight_dict.values())
    
    
    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add


    def set_latches(self):
        self.xVars: List[ADD] = self.kVars + self.pVars + [var for box_adds in self.bVars for var in box_adds]
        self.latches: List[ADD] = self.tVar + self.xVars
        self.latches_bdd: List[BDD] = [latch.bddPattern() for latch in self.latches]
    

    def set_prime_latches(self):
        self.prime_xVars: List[ADD] = self.prime_kVars + self.prime_pVars + [var for box_adds in self.prime_bVars for var in box_adds]
        self.prime_latches: List[ADD] = self.prime_tVar + self.prime_xVars
        self.prime_latches_bdd: List[BDD] = [latch.bddPattern() for latch in self.prime_latches]
    

    def get_number_of_states(self, verbose: bool = True):
        """
        A method to to compute the |Sys States| and |Env states| in the game.
         Sys States = Robot Configurations (ready, holding, to-obj) x Box Configurations x |turn variables|
         Env States = Robot Configurations (in-transit, in-transfer) x Box Configurations x |turn variables|

         Box conf. = (|locs + 1|)! / (|locs + 1| - |boxes|)! (locs = locations; +1 for end-effector loc)
         |ready| = |locs|; |holding| = |locs|; |to-obj| = |boxes| + 1 (for the 0-offset)
         |in-transit| = |boxes|*(|locs| + 1); |in-transfer| = |locs|*(|locs|-1);
        """
        sys_states = (self.ratio + 1)*(2*self.locs + self.boxes + 1)*(math.factorial(self.locs + 1) // math.factorial(self.locs + 1 - self.boxes))
        env_states = (self.ratio + 1)*((self.boxes*(self.locs + 1) + self.locs*(self.locs - 1)))*(math.factorial(self.locs + 1) // math.factorial(self.locs + 1 - self.boxes))
        if verbose:
            print(f'Number of States in Game: \n Sys States: {sys_states:,} \n Env States: {env_states:,} \n Total States: {sys_states + env_states:,}')
        return sys_states, env_states



    def set_init_latch(self) -> ADD:
        assert len(self.init) == self.boxes + 1, "[Error]: The init state should be fully defined for the Synthesis code else the Synthesis code will not work correctly."
        init_cube = self.tVar_map_sym['robot'] & self.kVar_map_sym['k0']
        for s in self.init:
            init_cube &= self.xVar_map_sym[s]
        return init_cube
    

    def set_goal_latch(self) -> ADD:
        mono_goal_cube = self.manager.addZero()
        for state in self.goal:
            goal_cube = self.manager.addOne()
            for s in state:
                goal_cube &= self.xVar_map_sym[s]
            mono_goal_cube |= goal_cube
        return mono_goal_cube
    

    def create_ee_empty_cube(self) -> ADD:
        # cube that implies that end-effector location empty
        bConf_cube = self.manager.addOne()
        for b in range(self.boxes):
            bConf_cube &= ~self.xVar_map_sym['b' + str(b) + ' l0']
        return bConf_cube & self.monolithic_relevant_box_preds
    
    def create_only_b_at_l_cube(self, curr_box: int, curr_loc: str, bConf_cube: ADD) -> ADD:
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.xVar_map_sym['b' + str(ob) + f' {curr_loc}']
        return bConf_cube


    def create_relevant_box_predicates(self):
        """
         A tiny method to create relevant box predicates for each box.
        """
        for box_str, box_add in self.bVar_map_sym.items():
            box_id = int(box_str.split(' ')[0][-1])
            assert isinstance(box_id, int) and box_id in range(self.boxes), "Error in extracting box id. Fix this!!!"
            self.relevant_box_preds_sym[box_id] |= box_add
    

    def create_hmove_not_b(self):
        """
         A method that create a cube that consists of all hmvoves that are moving any box except the box b.
        """
        self.hmove_not_b = defaultdict(lambda: self.manager.addZero())
        for b in range(self.boxes):
            for act_str, act_add in self.action_map_sym.items():
                # first check if it is a human move
                if act_str.startswith('hmove'):
                    if f'b{b}' not in act_str and not act_str.startswith('hmove noop'):
                        self.hmove_not_b[b] |= act_add

    def create_monoltithic_box_conf_cube(self):
        """
        Let try to use ITE method to create valid set of box configurations. Basically, monolithic_relevant_box_preds variable capturre all possible
          combinations of box configuration. Within this set, we need to enforce that no two boxes can be at the same location.
        """
        # add this to monolithic relevant box preds
        for b in range(self.boxes):
            for l in range(0, self.locs + 1):
                self.monolithic_relevant_box_preds &= self.bVar_map_sym[f'b{b} l{l}'].ite(self.create_only_b_at_l_cube(curr_box=b, curr_loc=f'l{l}', bConf_cube=self.manager.addOne()), self.manager.addOne())


    def create_valid_state_constraints(self):
        """
         A method to create valid state constraints that capture the relationship between robot configuration and box configurations.
        """
        # now lets add constraints that is rConf is ready then no box is at ee-location
        for b in range(self.boxes):
            for at_loc in range(1, self.locs + 2):
                valid_rConf_for_grasp_cube: ADD = self.xVar_map_sym[f'ready l{at_loc}'] | self.xVar_map_sym[f'in-transit l{at_loc} b{b}'] | self.xVar_map_sym[f'to-obj b{b}'] 
                self.monolithic_relevant_box_preds &= valid_rConf_for_grasp_cube.ite(self.ee_empty_cube, self.manager.addOne())
            
        # now lets add constraints that is rConf is holding then some box is at ee-location
        some_box_at_ee: ADD = reduce(lambda x, y: x | y, [self.xVar_map_sym[f'b{b} l0'] for b in range(self.boxes)])
        for from_loc in range(1, self.locs + 1):
            for to_loc in range(1, self.locs + 1):
                if from_loc == to_loc:
                    continue
                self.monolithic_relevant_box_preds &= (self.xVar_map_sym[f'in-transfer l{from_loc} l{to_loc}']).ite(some_box_at_ee, self.manager.addOne())
        
        for at_loc in range(1, self.locs + 1):
            self.monolithic_relevant_box_preds &= (self.xVar_map_sym[f'holding l{at_loc}']).ite(some_box_at_ee, self.manager.addOne())
        

        # map all invalid Rconf to zero
        valid_rconf: ADD = reduce(lambda x, y: x | y, [self.cube_to_add(i, self.pVars) for i in self.pVar_map.values()])
        self.monolithic_relevant_box_preds = valid_rconf.ite(self.monolithic_relevant_box_preds, self.manager.addZero())
    

    def preprocess_monolithic_valid_state_robot_actions(self):
        """
         monolithic_valid_state_robot_actions variable is uses to keep track of all valid robot actions under valid robot states.
         For preds that are in-transfer and in-transit, we just addOne() as these preds do not have any restriction on robot actions.

         This is needed because in the post_process_transition_relation method, we need to restrict the transition relation to only valid robot states and actions.
         For human moves, we do not need to do this as human action validity is excatly (precisly) determined by the respective human action constrcution methods.
        """
        # for in-transit and in-transfer preds, just addOne()
        for from_loc in range(1, self.locs + 2):
            for b in range(self.boxes):
                self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'in-transit l{from_loc} b{b}']).ite(self.manager.addOne(), self.manager.addZero())
        
        # now lets add constraints that is rConf is holding then some box is at ee-location
        for from_loc in range(1, self.locs + 1):
            for to_loc in range(1, self.locs + 1):
                if from_loc == to_loc:
                    continue
                self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'in-transfer l{from_loc} l{to_loc}']).ite(self.manager.addOne(), self.manager.addZero())
        
        for loc in range(1, self.locs + 2):
            if loc < self.locs + 1:
                self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'holding l{loc}']).ite(self.manager.addOne(), self.manager.addZero())
                self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'ready l{loc}']).ite(self.manager.addOne(), self.manager.addZero())
            else:
                self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'ready l{loc}']).ite(self.manager.addOne(), self.manager.addZero())


    def post_process_transition_relation(self):
        """
         A method to post-process the transition relation after all action rules and frame axioms have been added.
        """
        for tr_key, tr_dd in self.transition_relation.items():
            self.transition_relation[tr_key] &= self.monolithic_relevant_box_preds & self.monolithic_valid_state_robot_actions
    

    def debug_monolithic_state_action_prime_state(self):
        """
         A method to debug the monolithic_state_action_prime_state variable. I think are some actiosn missing from this ADD. 
         I need to find out which one are these.
        """
        all_state_action_dd = self.manager.addZero()
        for tr_dd in self.transition_relation.values():
            all_state_action_dd |= tr_dd
        
        action_cube = reduce(lambda a, b: a & b, self.rVars)
        prime_vars_exist_cube = reduce(lambda a, b: a & b, self.prime_latches)
        
        monolithic_state_action_prime_state_bdd = self.monolithic_state_action_prime_state.bddPattern()
        # monolithic_state_prime_state_bdd = monolithic_state_action_prime_state_bdd.existAbstract(action_cube.bddPattern())
        monolithic_state_bdd = monolithic_state_action_prime_state_bdd.existAbstract(prime_vars_exist_cube.bddPattern())
        
        # convert to BDD
        state_acts_not_in_monolithic = all_state_action_dd.bddPattern() & ~(monolithic_state_bdd)
        # remove invalid conf from current state and next state
        # state_acts_not_in_monolithic &= self.monolithic_relevant_box_preds.bddPattern()
        # state_acts_not_in_monolithic &= self.monolithic_relevant_box_preds.swapVariables(self.latches, self.prime_latches).bddPattern()
        # robot_to_human_cube = self.tVar_map_sym['robot'].bddPattern() & self.prime_tVar_map_sym['human'].bddPattern()
        # human_to_robot_cube = self.tVar_map_sym['human'].bddPattern() & self.prime_tVar_map_sym['robot'].bddPattern()
        # state_acts_not_in_monolithic &= robot_to_human_cube | human_to_robot_cube

        # rConstraint_cube = self.manager.addZero()
        # for kVal in self.kVar_map.keys():
        #     rConstraint_cube |= self.kVar_map_sym[kVal].ite(self.prime_kVar_map_sym[kVal], self.manager.addZero())
        # state_acts_not_in_monolithic &= rConstraint_cube.bddPattern()
        # state_acts_not_in_monolithic &= (self.action_map_sym['hmove noop']).ite(self.prime_kVar_map_sym['k0'], self.manager.addOne()).bddPattern()
        if state_acts_not_in_monolithic.isZero():
            print("No state-action-next_state triples are missing.")
            return
        print("Some state-action-next_state triples are missing from the monolithic_state_action_prime_state variable.")
        
        print(state_acts_not_in_monolithic.toADD())
        sys.exit(1)

    def postprocess_monolithic_valid_state_robot_actions_prime_state(self):
        """
         A method to post-process the transition relation after all action rules and frame axioms have been added.
        """
        self.monolithic_valid_state_robot_actions_prime_state &= self.tVar_map_sym['robot'].ite(self.prime_tVar_map_sym['human'], self.manager.addZero())
        self.monolithic_valid_state_human_actions_prime_state &= self.tVar_map_sym['human'].ite(self.prime_tVar_map_sym['robot'], self.manager.addZero())
        
        rConstraint_cube = self.manager.addZero()
        for kVal in self.kVar_map.keys():
            rConstraint_cube |= self.kVar_map_sym[kVal].ite(self.prime_kVar_map_sym[kVal], self.manager.addZero())
        
        self.monolithic_valid_state_robot_actions_prime_state &= rConstraint_cube
        self.monolithic_valid_state_human_actions_prime_state &= (self.action_map_sym['hmove noop']).ite(self.prime_kVar_map_sym['k0'], self.manager.addOne())
        # remove invalid box configurations form the current states 
        self.monolithic_valid_state_human_actions_prime_state &= self.monolithic_relevant_box_preds
        # remove invalid box configurations form the next states
        self.monolithic_valid_state_human_actions_prime_state &= self.monolithic_relevant_box_preds.swapVariables(self.latches, self.prime_latches)

        # put the robot and human edges together
        self.monolithic_state_action_prime_state = self.tVar_map_sym['robot'].ite(self.monolithic_valid_state_robot_actions_prime_state, self.monolithic_valid_state_human_actions_prime_state)
    
    def add_turn_var_update_rule(self):
        """
         A method to add turn variable update rule. Irrespective of the action taken, after every turn, the turn variable is flipped.
        """
        curr_pred = [self.tVar_map_sym['robot'], self.tVar_map_sym['human']]
        next_pred_str = [self.tVar_map['human'], self.tVar_map['robot']]
        for turn_bit, turn_prime_string in zip(curr_pred, next_pred_str):
            for sidx, s in enumerate(turn_prime_string):
                if s == '1':
                    self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= turn_bit
    
    def add_hmove_var_update_rule_from_robot_states(self):
        """
         A method to add K Var variable update rule. K here correspond to the number of human interventions taken so far in the current robot turn. 
          From any robot state, the turn var does no change its value and its value is only updated after the human turn. Thus, k <-> k' where k' = k holds
          true under all robot actions from all robot states.
        """
        for kVal, kVal_str in self.kVar_map.items():
            for sidx, s in enumerate(kVal_str):
                if s == '1':
                    self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |=  self.tVar_map_sym['robot'] & self.kVar_map_sym[kVal]


    def create_transition_relation(self):
        """
         A method to construct the transition relation for the Franka World Dynamic domain. 
         This constructs the transition system that consists of actions controlled by the robot (robot actions) and human (env actions).

         After every turn, the turn variable is flipped and the game evolves to the other player.
        """
        # first we create grasp actions
        self.create_grasp_actions()

        # next we create the release actions
        self.create_release_actions()

        # next we create the transit actions
        self.create_transit_actions()

        # next we create the transfer actions
        self.create_transfer_actions()

        # finally, we add frame axioms for all boxes that enforce state invariance constraint
        self.add_robot_frame_axioms()
        if self.boxes > 1 and self.only_reachable_states:
            self.add_robot_s_sprime_frame_axioms()
        
        self.create_human_move_transit()
        self.create_human_move_transfer()
        self.create_human_move_action_grasp()
        self.create_human_move_actions()

        self.add_human_frame_axiom()

        self.add_turn_var_update_rule()
        self.add_hmove_var_update_rule_from_robot_states()
        
        # keep only the valid robot states and actions in the transition relation
        self.post_process_transition_relation()

        if self.only_reachable_states:
            self.postprocess_monolithic_valid_state_robot_actions_prime_state()
            self.care_states = self.compute_reachable_states(monolithic_trans_dd=self.monolithic_state_action_prime_state,
                                                             latches=self.latches, prime_latches=self.prime_latches,
                                                             act_vars=self.rVars, verbose=False, print_states=False)
        
        # print s a_s s' transition function that we created for sanity checking
        # self.convert_full_cube_to_state_ADD(self.monolithic_valid_state_human_actions_prime_state, action=True, verbose=True)
        # self.count_actions_per_state()
        

    def create_grasp_actions(self) -> None:
        """
        Here we create grasp actions for the robot. The precondition for the robot to grasp box b at location l is as follows:
        Preconditions:
         1. The robot is at location l where box b is (b @ l) - (to-obj) (b l) predicates are true at current state
        Effects+:
         2. The robot is holding box b: (holding l) (b l0) predicates are true at next state (Note: l0 is reserved for end-effector location)
        Effects-:
         3. The robot's end effector is not empty: ~(to-obj) ~(b l) predicates are true at next state
        """
        # need to enforce that the end-effector is empty
        turn_bit: ADD = self.tVar_map_sym['robot']
        state_constraint_cube = self.ee_empty_cube
        robot_act_cube = self.action_map_sym['grasp']
        for b in range(self.boxes):
            rConf_cube = self.xVar_map_sym[f'to-obj b{b}']

            # for a given box, it can be at any location, so we iterate over all locations
            for loc in range(1, self.locs + 1):
                rConf_cube_ready = self.xVar_map_sym[f'ready l{loc}']
                curr_box_pred = f"b{b} l{loc}"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(loc), bConf_cube=bConf_cube) & self.monolithic_relevant_box_preds

                robot_transition_cube = turn_bit & self.kVal_cube & bConf_cube & state_constraint_cube & robot_act_cube & (rConf_cube | rConf_cube_ready)

                # update the valid robot moves
                self.monolithic_valid_state_robot_actions |= (turn_bit & (rConf_cube | rConf_cube_ready) & bConf_cube & state_constraint_cube).ite(robot_act_cube, self.manager.addZero())
                
                # this is fixed
                pred_clause_prime_string = self.xVar_map['holding l' + str(loc)]
                box_clause_prime_string = self.xVar_map[f'b{b} l0']
                
                # now we add the transition where the human does all the valid move and the robot grasps the box
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube
                
                # create s a_s s' transitions
                prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym['holding l' + str(loc)] & self.prime_xVar_map_sym[f'b{b} l0']
                self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())


    def create_release_actions(self) -> None:
        """
        Here we create release actions for the robot. The precondition for the robot to release box b at location l is as follows:
        Preconditions:
         1. The robot is at location l where no box (b') is located, i.e., (b @ l0) & ~(b' @ l) :- 
            (holding l) ~(b' l) predicates are true at current state
        Effects+:
         2. The robot is ready at location l: (ready l) (b l) predicates are true at next state
        Effects-:
         3. The robot's end effector is empty and not holding b: ~(holding l) ~(b l0) (for all b) predicates are true at next state
        """
        turn_bit = self.tVar_map_sym['robot']
        robot_act_cube = self.action_map_sym['release']
        for loc in range(1, self.locs + 1):
            rConf_cube = self.xVar_map_sym[f'holding l{loc}']

            # for a given location, it can be any location, so we iterate over all locations
            for b in range(self.boxes):
                curr_box_pred = f"b{b} l0"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l0', bConf_cube=bConf_cube) & self.monolithic_relevant_box_preds

                robot_transition_cube = turn_bit & self.kVal_cube & rConf_cube & bConf_cube & robot_act_cube & self.locs_empty_constraints[f'l{loc}']

                # update the valid robot moves
                self.monolithic_valid_state_robot_actions |= (turn_bit & rConf_cube & bConf_cube & self.locs_empty_constraints[f'l{loc}']).ite(robot_act_cube, self.manager.addZero())

                # this is fixed
                next_box_pred = f"b{b} l{loc}"
                pred_clause_prime_string = self.xVar_map['ready l' + str(loc)]
                box_clause_prime_string = self.xVar_map[next_box_pred]
                
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube

                # create s a_s s' transitions
                prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym['ready l' + str(loc)] & self.prime_xVar_map_sym[next_box_pred]
                self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())    

    def create_transit_actions(self) -> None:
        """
        Here we create transit actions for the robot. The precondition for the robot is box b is at location l:
         Preconditions:
            1. The robot is at location l (l =\= l') and end-effector is empty - (ready l) ~(b l0) (for all boxes) predicates are true at current state
         Effects+:
            2. The robot is at location l': (to-obj b) (b l') predicates are true at next state
         Effects-:
            3. The robot's location has changed: ~(ready l) predicate is true at next state
        """
        state_constraint_cube = self.ee_empty_cube
        turn_bit = self.tVar_map_sym['robot']
        for b in range(self.boxes):
            robot_act_cube = self.action_map_sym[f"transit b{b}"]

            for from_loc in range(1, self.locs + 2):
                rConf_cube = self.xVar_map_sym[f'ready l{from_loc}']

                for to_loc in range(1, self.locs + 1):
                    if from_loc == to_loc:
                        continue
                    curr_box_pred = f"b{b} l{to_loc}"
                    bConf_cube = self.xVar_map_sym[curr_box_pred]
                    bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc=f'l{to_loc}', bConf_cube=bConf_cube) & self.monolithic_relevant_box_preds

                    robot_transition_cube = turn_bit & self.kVal_cube & rConf_cube & state_constraint_cube & robot_act_cube & bConf_cube

                    # update the valid robot moves
                    self.monolithic_valid_state_robot_actions |= (turn_bit & rConf_cube & bConf_cube & state_constraint_cube).ite(robot_act_cube, self.manager.addZero())

                    pred_clause_prime_string = self.xVar_map[f"in-transit l{from_loc} b{b}"]
                    
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    # create s a_s s' transitions
                    prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym[f"in-transit l{from_loc} b{b}"] & self.prime_xVar_map_sym[curr_box_pred]
                    self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())


    def create_transfer_actions(self) -> None:
        """
        Here we create transfer actions for the robot. The precondition for the robot is box b is at location l0 (end effector location):
         Preconditions:
            1. The robot is at location l and is holding box b - (holding l) (b l0) predicates are true at current state
         Effects+:
            2. The robot is at location l': (in-transfer l l') (b l0) predicates are true at next state
         Effects-:
            3. The robot's location has changed: ~(holding l) predicate is true at next state
        """
        turn_bit = self.tVar_map_sym['robot']
        for b in range(self.boxes):
            curr_box_pred = f'b{b} l0'
            bConf_cube = self.xVar_map_sym[curr_box_pred]
            bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l0', bConf_cube=bConf_cube) & self.monolithic_relevant_box_preds
        
            for from_loc in range(1, self.locs + 1):
                rConf = f'holding l{from_loc}'
                rConf_cube = self.xVar_map_sym[rConf]
                for to_loc in range(1, self.locs + 1):
                    # skip transferring to the same location
                    if from_loc == to_loc:
                        continue
                    robot_act_cube = self.action_map_sym[f'transfer l{to_loc}']

                    robot_transition_cube = turn_bit & self.kVal_cube & rConf_cube & robot_act_cube & bConf_cube

                    # update the valid robot moves
                    self.monolithic_valid_state_robot_actions |= (turn_bit & rConf_cube & bConf_cube).ite(robot_act_cube, self.manager.addZero())

                    # next state clause - (in-transfer from_loc to_loc); box location does not change
                    pred_clause_prime_string = self.xVar_map[f'in-transfer l{from_loc} l{to_loc}']
                    
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    box_clause_prime_string = self.xVar_map[curr_box_pred]
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    # create s a_s s' transitions
                    prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym[f'in-transfer l{from_loc} l{to_loc}'] & self.prime_xVar_map_sym[curr_box_pred]
                    self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())
    

    def create_human_move_transfer(self) -> None:
        """
         From the in-transfer states, if human moves a box, the robot conf and the box conf. change. 
          If the human does not move a box, the robot conf. evolves to holding org-destination and box conf. remain the same for all the boxes.
          If the human does move a box, the robot conf. goes back to holding org-src and the box conf. of the moved box changes. 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for from_loc in range(1, self.locs + 1):
                for to_loc in range(1, self.locs + 1):
                    if from_loc == to_loc:
                        continue
                    # for all in-transfer preds
                    invalid_hmove_cube = self.manager.addZero()
                    rConf_cube = self.xVar_map_sym[f'in-transfer l{from_loc} l{to_loc}']

                    for human_box in self.human_boxes:
                        for human_to_loc in self.human_locs:
                            ##### VALID MOVE CASE #####
                            hmove_cube = turn_bit & kVal_cube & \
                            self.action_map_sym[f'hmove b{human_box} l{human_to_loc}'] & rConf_cube
                            
                            
                            constraint_cube = self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube &= ~self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            hmove_cube &= constraint_cube & self.monolithic_relevant_box_preds
                            
                            ##### INVALID MOVE CASE #####
                            constraint_cube = ~self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube |= self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            
                            invalid_hmove_cube |=  turn_bit & kVal_cube & rConf_cube & \
                                self.action_map_sym[f'hmove b{human_box} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds

                            if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                                pred_clause_prime_string = self.xVar_map[f'holding l{from_loc}']
                                for sidx, s in enumerate(pred_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                                
                                # update k var - increment k by 1
                                kVal_prime_str = self.kVar_map[f'k{k + 1}']
                                for sidx, s in enumerate(kVal_prime_str):
                                    if s == '1':
                                        self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                                
                                # create s a_s s' transitions - human
                                prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'holding l{from_loc}'] & self.prime_kVar_map_sym[f'k{k + 1}']
                                self.monolithic_valid_state_human_actions_prime_state |= hmove_cube.ite(prime_state_cube, self.manager.addZero())
                            else:
                                invalid_hmove_cube |= hmove_cube

                            
                    # add the human noop action here - hmove noop is always a valid human move
                    pred_clause_prime_string = self.xVar_map[f'holding l{to_loc}']                    
                    hmove_cube = turn_bit & kVal_cube & (self.action_map_sym['hmove noop'] | invalid_hmove_cube) & rConf_cube
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    for sidx, s in enumerate(self.kVar_map['k0']):
                        if s == '1':
                            self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    # create s a_s s' transitions - human
                    prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'holding l{to_loc}'] & self.prime_kVar_map_sym['k0']
                    self.monolithic_valid_state_human_actions_prime_state |= hmove_cube.ite(prime_state_cube, self.manager.addZero())

    

    def create_human_move_transit(self) -> None:
        """
         From the in-transit states, if human moves a box, the robot conf and the box conf. change. 
          If the human does not move a box, the robot conf. evolves to to-obj boc and box conf. remain the same for all the boxes. 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for b in range(self.boxes):
                for from_loc in range(1, self.locs + 2):
                    # for all in-transit preds
                    invalid_hmove_cube = self.manager.addZero()
                    rConf_cube = self.xVar_map_sym[f'in-transit l{from_loc} b{b}']

                    for human_box in self.human_boxes:
                        for human_to_loc in self.human_locs:
                            ##### VALID MOVE CASE #####
                            hmove_cube = turn_bit & kVal_cube & \
                                self.action_map_sym[f'hmove b{human_box} l{human_to_loc}'] & rConf_cube & self.ee_empty_cube
                            
                            constraint_cube = self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube &= ~self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            
                            hmove_cube &= constraint_cube & self.monolithic_relevant_box_preds

                            ##### INVALID MOVE CASE (not because of reaching the max human intervention) #####
                            constraint_cube = ~self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube |= self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            invalid_hmove_cube |=  turn_bit & kVal_cube & \
                                self.action_map_sym[f'hmove b{human_box} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds

                            # If the human can still intervene then add it set of valis moves and increment k by 1
                            if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                                pred_clause_prime_string = self.xVar_map[f'ready l{from_loc}']
                                for sidx, s in enumerate(pred_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube

                                # update k var - increment k by 1
                                kVal_prime_str = self.kVar_map[f'k{k + 1}']
                                for sidx, s in enumerate(kVal_prime_str):
                                    if s == '1':
                                        self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                            
                                # create s a_s s' transitions - human
                                prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'ready l{from_loc}'] & self.prime_kVar_map_sym[f'k{k + 1}']
                                self.monolithic_valid_state_human_actions_prime_state |= hmove_cube.ite(prime_state_cube, self.manager.addZero())
                            # the human has reached the max number of interventions in this robot turn;
                            # all they can do is noop so we add all hmove to invalid move case
                            else:
                                invalid_hmove_cube |= hmove_cube
                        
                    # add the human noop action here - hmove noop is always a valid human move                    
                    hmove_cube = turn_bit & kVal_cube & (self.action_map_sym['hmove noop'] | invalid_hmove_cube) & rConf_cube & self.ee_empty_cube
                    pred_clause_prime_string = self.xVar_map[f'to-obj b{b}']
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    for sidx, s in enumerate(self.kVar_map['k0']):
                        if s == '1':
                            self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    # create s a_s s' transitions - human
                    prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'to-obj b{b}'] & self.prime_kVar_map_sym['k0']
                    self.monolithic_valid_state_human_actions_prime_state |= hmove_cube.ite(prime_state_cube, self.manager.addZero())


    def create_human_move_actions(self) -> None:
        """
        Here we create human move actions for the environment. The precondition for the human to move box b to location l is as follows:
         Preconditions:
            1. The human can move box b to location l only if location l is empty - (b @ l') & ~(b @ l) predicates are true at current state 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        parent_hmove_cube = self.manager.addZero()
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for hb in self.human_boxes:
                # boxes can be "grounded" at any location (except for 0 and else := |locs| + 1 location)
                for from_loc in range(1, self.locs + 1):
                    invalid_hmove_cube = self.manager.addZero()
                    box_pred = f"b{hb} l{from_loc}"
                    bConf_cube = self.xVar_map_sym[box_pred]
                    for human_to_loc in self.human_locs:
                        ##### VALID MOVE CASE #####
                        hmove_cube = turn_bit & kVal_cube & self.action_map_sym[f'hmove b{hb} l{human_to_loc}']
                        
                        constraint_cube = self.locs_empty_constraints[f'l{human_to_loc}']
                        for restricted_loc in self.restricted_human_locs:
                            constraint_cube &= ~self.xVar_map_sym[f'b{hb} l{restricted_loc}']
                        hmove_cube &= constraint_cube & self.monolithic_relevant_box_preds

                        ##### INVALID MOVE CASE #####
                        # if it not a valid move, then human action should not have any affect
                        constraint_cube = ~self.locs_empty_constraints[f'l{human_to_loc}']
                        for restricted_loc in self.restricted_human_locs:
                            constraint_cube |= self.xVar_map_sym[f'b{hb} l{restricted_loc}']
                        invalid_hmove_cube |= turn_bit & kVal_cube &  \
                            self.action_map_sym[f'hmove b{hb} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds
                        

                        if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                            moved_box_prime_str = self.xVar_map[f'b{hb} l{human_to_loc}'] 
                            for sidx, s in enumerate(moved_box_prime_str):
                                if s == '1':
                                    self.transition_relation[self.bVars[hb][sidx].bddPattern().__str__()] |= hmove_cube
                            
                            # update k var - increment k by 1
                            kVal_prime_str = self.kVar_map[f'k{k + 1}']
                            for sidx, s in enumerate(kVal_prime_str):
                                if s == '1':
                                    self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                            
                            # create s a_s s' transitions - human
                            prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'b{hb} l{human_to_loc}'] & self.prime_kVar_map_sym[f'k{k + 1}']
                            self.monolithic_valid_state_human_actions_prime_state &= hmove_cube.ite(prime_state_cube, self.manager.addOne())
                            parent_hmove_cube |= hmove_cube.ite(prime_state_cube, self.manager.addZero())
                        # the human has reached the max number of interventions in this robot turn;
                        # all they can do is noop so we add all hmove to invalid move case
                        else:
                            invalid_hmove_cube |= hmove_cube
                    
                    
                    hmove_cube = turn_bit & kVal_cube & (self.action_map_sym['hmove noop'] | invalid_hmove_cube) & bConf_cube
                    for sidx, s in enumerate(self.xVar_map[box_pred]):
                        if s == '1':
                            self.transition_relation[self.bVars[hb][sidx].bddPattern().__str__()] |= hmove_cube
                    
                    for sidx, s in enumerate(self.kVar_map['k0']):
                        if s == '1':
                            self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    # create s a_s s' transitions - human
                    prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[box_pred] & self.prime_kVar_map_sym['k0']
                    # self.monolithic_valid_state_human_actions_prime_state &= hmove_cube.ite(prime_state_cube, self.manager.addZero())
                    parent_hmove_cube = hmove_cube.ite(prime_state_cube, self.manager.addZero())
    
        self.tmp_parent_hmove_cube = parent_hmove_cube            
    
    def create_human_move_action_grasp(self) -> None:
        """
        Here we create human move actions for the environment during robot grasp action.
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for hb in self.human_boxes:
                # boxes can be "grounded" at any location (except for 0 and else := |locs| + 1 location)
                for from_loc in range(1, self.locs + 1):
                    invalid_hmove_cube = self.manager.addZero()
                    
                    rConf_cubes_list = [self.xVar_map_sym[f'holding l{from_loc}'], self.xVar_map_sym[f'ready l{from_loc}'] & self.ee_empty_cube]
                    pred_clause_prime_string_list = [self.xVar_map[f'holding l{from_loc}'], self.xVar_map[f'ready l{from_loc}']]
                    pred_clause_prime_sym_list = [self.prime_xVar_map_sym[f'holding l{from_loc}'], self.prime_xVar_map_sym[f'ready l{from_loc}']]
                    for rConf_cube, pred_clause_prime_string, pred_clause_prime_sym in zip(rConf_cubes_list, pred_clause_prime_string_list, pred_clause_prime_sym_list):
                        for human_to_loc in self.human_locs:
                            ##### VALID MOVE CASE #####
                            hmove_cube = turn_bit & kVal_cube & self.action_map_sym[f'hmove b{hb} l{human_to_loc}'] & rConf_cube
                            
                            constraint_cube = self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube &= ~self.xVar_map_sym[f'b{hb} l{restricted_loc}']
                            hmove_cube &= constraint_cube & self.monolithic_relevant_box_preds

                            ##### INVALID MOVE CASE #####
                            # if it not a valid move, then human action should not have any affect
                            constraint_cube = ~self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube |= self.xVar_map_sym[f'b{hb} l{restricted_loc}']
                            
                            invalid_hmove_cube |= turn_bit & kVal_cube & rConf_cube & \
                                self.action_map_sym[f'hmove b{hb} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds
                            
                            if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                                for sidx, s in enumerate(pred_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube 
                                
                                # update k var - increment k by 1
                                kVal_prime_str = self.kVar_map[f'k{k + 1}']
                                for sidx, s in enumerate(kVal_prime_str):
                                    if s == '1':
                                        self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                                
                                # create s a_s s' transitions - human
                                prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & pred_clause_prime_sym & self.prime_kVar_map_sym[f'k{k + 1}']
                                self.monolithic_valid_state_human_actions_prime_state |= hmove_cube.ite(prime_state_cube, self.manager.addZero())
                            
                            # the human has reached the max number of interventions in this robot turn;
                            # all they can do is noop so we add all hmove to invalid move case
                            else:
                                invalid_hmove_cube |= hmove_cube
                        
                        hmove_cube = turn_bit & kVal_cube & (self.action_map_sym['hmove noop'] | invalid_hmove_cube) & rConf_cube
                        # pred_clause_prime_string = self.xVar_map[f'holding l{from_loc}']
                        for sidx, s in enumerate(pred_clause_prime_string):
                            if s == '1':
                                self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                        
                        for sidx, s in enumerate(self.kVar_map['k0']):
                            if s == '1':
                                self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                        
                        # create s a_s s' transitions - human
                        prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & pred_clause_prime_sym & self.prime_kVar_map_sym['k0']
                        self.monolithic_valid_state_human_actions_prime_state |= hmove_cube.ite(prime_state_cube, self.manager.addZero())
    

    def add_robot_s_sprime_frame_axioms(self):
        """
         Adding frame axioms that ensure that boxes not being moved by the robot remain in the same location.

         This method is for curr vars --- (robot action) ---> next vars. 

         It is different from add_robot_frame_axioms() method that adds frame axioms using the negation operator which causes isuess for ADD.
        """
        transit_cube = self.manager.addZero()
        transfer_cube = self.manager.addZero()
        for rAct, rAct_cube in self.action_map_sym.items():
            if rAct.startswith('transit'):
                transit_cube |= rAct_cube
            elif rAct.startswith('transfer'):
                transfer_cube |= rAct_cube

        parent_constraint = self.manager.addZero()
        for i in range(0, 2):
            transit_constraint = self.manager.addOne()
            for b in range(self.boxes):
                transit_constraint_b = self.manager.addZero()
                for l in range(1, self.locs + 1):                    
                    curr_state_cube: ADD = self.tVar_map_sym['robot'] & self.kVal_cube & self.xVar_map_sym[f'b{b} l{l}']
                    transit_constraint_b |= curr_state_cube.ite(self.prime_xVar_map_sym[f'b{b} l{l}'], self.manager.addZero())
                    if i == 1 and l == self.locs:
                        curr_state_cube: ADD = self.tVar_map_sym['robot'] & self.kVal_cube & self.xVar_map_sym[f'b{b} l0']
                        transit_constraint_b |= curr_state_cube.ite(self.prime_xVar_map_sym[f'b{b} l0'], self.manager.addZero())

                transit_constraint &= transit_constraint_b
            
            parent_constraint |= transit_constraint & transit_cube if i == 0 else transit_constraint & transfer_cube 
        
        print("Added Transit & Transfer constraints to robot frame axioms.")
        
        for b in range(self.boxes):
            for grasp_loc in range(1, self.locs + 1):
                rConf_cube = self.xVar_map_sym[f'ready l{grasp_loc}']
                
                for non_grasp_loc in range(1, self.locs + 1):
                    if grasp_loc == non_grasp_loc:
                        continue
                    parent_constraint |= (self.tVar_map_sym['robot'] & self.kVal_cube & rConf_cube & self.action_map_sym['grasp'] & \
                        self.xVar_map_sym[f'b{b} l{non_grasp_loc}']).ite(self.prime_xVar_map_sym[f'b{b} l{non_grasp_loc}'], self.manager.addZero())  
        
        for b in range(self.boxes):
            rConf_cube = self.xVar_map_sym[f'to-obj b{b}']
            for other_b in range(self.boxes):
                if b == other_b:
                    continue
                for loc in range(1, self.locs + 1):                
                    parent_constraint |= (self.tVar_map_sym['robot'] & self.kVal_cube & rConf_cube & self.action_map_sym['grasp'] & \
                        self.xVar_map_sym[f'b{other_b} l{loc}']).ite(self.prime_xVar_map_sym[f'b{other_b} l{loc}'], self.manager.addZero()) 
        print("Added Grasp constraint to robot frame axioms.")

        # finally we add the release constraint
        # TODO: Check if constraint is correct or not
        for grasp_loc in range(1, self.locs + 1):
            rConf_cube = self.xVar_map_sym[f'holding l{grasp_loc}']
            for non_grasp_loc in range(1, self.locs + 1):
                if grasp_loc == non_grasp_loc:
                    continue
                for b in range(self.boxes):
                    parent_constraint |= (self.tVar_map_sym['robot'] & self.kVal_cube & rConf_cube & self.action_map_sym['release'] & \
                            self.xVar_map_sym[f'b{b} l0']).ite(self.prime_xVar_map_sym[f'b{b} l{non_grasp_loc}'], self.manager.addZero())
        
        self.monolithic_valid_state_robot_actions_prime_state &= parent_constraint
        print("Added Release constraint to robot frame axioms.")
    
    def assert_one_s_prime_s_relation(self, dd_full_trans_rel: ADD):
        """
        Given the monolithic ADD(s-as-s') where,
         s : is the current state
         as : is a valid robot action from s
         s': is the next state
        this helper function asserts that |s-s'| = 1, i.e., there exactly one robot action for every s-s' transition function.
        """
        action_cube = reduce(lambda a, b: a & b, self.rVars)
        state_action_prime_state: BDD = dd_full_trans_rel.bddPattern()                                                         
        state_action_prime_state: BDD = state_action_prime_state.existAbstract(action_cube.bddPattern())

        # convert to 0 - 1 ADD 
        add_state_action_prime_state  = state_action_prime_state.toADD()
        max_cardinality: ADD = add_state_action_prime_state.findMax()
        assert max_cardinality == self.manager.addOne(), "Error: Issue Constructing monolithic ADD(s-as-s'). There is non-determinism in robot action transitin in the game."
        print("Verified: ADD(s-as-s') is correct")


    def count_actions_per_state(self):
        """
         A function that counts the numbe of actions per state
        """
        prime_vars_exist_cube = reduce(lambda a, b: a & b, self.prime_latches)
        action_cube = reduce(lambda a, b: a & b, self.rVars)
        # convert to BDD and then exist abstract
        # state_action_prime_state: BDD = self.monolithic_valid_state_robot_actions_prime_state.bddPattern()
        # state_action_prime_state: BDD = self.monolithic_valid_state_human_actions_prime_state.bddPattern()
        state_action_prime_state: BDD = self.monolithic_state_action_prime_state.bddPattern()
        state_action_prime_state = state_action_prime_state.existAbstract(prime_vars_exist_cube.bddPattern())

        # convert to 0 - 1 ADD 
        add_state_action_prime_state  = state_action_prime_state.toADD()

        # exist abstract the ADD - this sums all leaves (1s) from a state
        # we get ADD where each state maps to leaf node with vlaue = # of valid robot actions
        test = add_state_action_prime_state.existAbstract(action_cube)

        # print it
        # self.convert_cube_to_state_ADD(test, state_flag=True, verbose=True)



    def add_robot_frame_axioms(self):
        """
        A helper function to test the frame axioms. Frame axioms ensure that the boxes that
          are not being moved by the robot do not change their location. This is like a state invariance constraint.
        
        Here, we add frame axioms for all boxes that are currently "grounded" (i.e., not being moved by the robot). 
          For the box that is at end-effector (ee) location (l0), we add the state invariance constraint during the action construction.
        """
        grasp_action_cube = self.action_map_sym['grasp']
        release_action_cube = self.action_map_sym['release']
        for b in range(self.boxes):
            for l in range(1, self.locs + 1):
                box_pred = f"b{b} l{l}"
                constraint_cube = self.manager.addOne()
                not_grasp_cube = ~((self.xVar_map_sym[f'to-obj b{b}'] | self.xVar_map_sym[f'ready l{l}']) & grasp_action_cube)
                not_release_cube = ~(self.xVar_map_sym[f'holding l{l}'] & release_action_cube)
                constraint_cube &= not_grasp_cube & not_release_cube \
                    & self.relevant_robot_actions & self.monolithic_valid_state_robot_actions 
                
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= self.tVar_map_sym['robot'] & self.kVal_cube \
                            & constraint_cube & self.xVar_map_sym[box_pred]
                
    def add_human_frame_axiom(self):
        """
        This helper method add frame axioms for the human move actions. 
         Boxes that the human does not move should remain in the same location in the next state.
        """
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        for b in range(self.boxes):
            for l in range(0, self.locs + 1):
                box_pred = f"b{b} l{l}"
                if (l in self.restricted_human_locs) or (b not in self.human_boxes):
                    # this transition exists for all valid human edges
                    haction_cube = self.tVar_map_sym['human'] & self.xVar_map_sym[box_pred] & self.kVal_cube & valid_human_action_mask
                else:
                    haction_cube = self.tVar_map_sym['human'] \
                        & self.xVar_map_sym[box_pred] &  (self.action_map_sym['hmove noop'] | self.hmove_not_b[b]) & self.kVal_cube
                    constraint_cube = self.manager.addOne()
                    for restricted_loc in self.restricted_human_locs:
                        constraint_cube &= ~self.xVar_map_sym[f'b{b} l{restricted_loc}']
                    haction_cube &= constraint_cube & self.monolithic_relevant_box_preds

                # box remmains in the same location if human does not move it
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= haction_cube
                
                # create s a_s s' transitions - human
                prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[box_pred] & self.kVal_cube.swapVariables(self.kVars, self.prime_kVars)
                self.monolithic_valid_state_human_actions_prime_state &= haction_cube.ite(prime_state_cube, self.manager.addOne())
                self.tmp_parent_hmove_cube |= haction_cube.ite(prime_state_cube, self.manager.addZero())

    
    def get_all_cubes(self, dd: ADD, relevant_vars: List[ADD]) -> List[Tuple[ADD, float]]:
        cubes = []
        for cube_list, val in dd.generate_cubes():
            if val == math.inf:
                continue
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube_list):
                if self.manager.addVar(_idx) not in relevant_vars:
                    continue

                if var == 2:
                    _amb_var.append([self.manager.addVar(_idx), ~self.manager.addVar(_idx)])
                elif var == 0:
                    var_list.append(~self.manager.addVar(_idx))
                elif var == 1:
                    var_list.append(self.manager.addVar(_idx))
                else:
                    print("CUDD ERRROR, A variable is assigned an unaccounted integer assignment. FIX THIS!!")
                    sys.exit(-1)
            
            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    cubes.append((reduce(lambda a, b: a & b, var_list), val))
                    var_list = list(set(var_list) - set(_ele))
            else:
                cubes.append((reduce(lambda a, b: a & b, var_list), val))
        
        return cubes


    def get_all_states_interval(self, upper: int, dd: ADD, lower: int = 0) -> BDD:
        """
         Helper function to get all the states below between Lower and Upper (both inclusive). 
         Note Strict inludes the threshold value as well.
        """
        # returnns BDD of all the states with state value greater than lower
        bdd_sgtl = dd.bddStrictThreshold(lower)
        # returnns BDD of all the states with state value greater than upper
        bdd_gtu = dd.bddThreshold(upper)
        # this include cubes corresponding to the upper values as well
        bdd_ltu = ~bdd_gtu

        return bdd_ltu & ~bdd_sgtl
    
    
    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, action: bool = False, verbose: bool = False, table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches)
        if action:
            relevant_vars.extend(self.rVars)
        
        headers = []
        if verbose and action:
            headers = ['state', 'action', 'value']
        elif verbose and not action:
            headers = ['state', 'value']

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.rVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.rVars)
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.rVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.rVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = [] 
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_cube_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = (self.tVar_map.inv[tConf_cube_str], self.kVar_map.inv[kConf_cube_str], self.pVar_map.inv[rConf_cube_str], box_states)
                states_action_pairs.append([((self.tVar_map.inv[tConf_cube_str], self.kVar_map.inv[kConf_cube_str], self.pVar_map.inv[rConf_cube_str], box_states), val), None])
            except KeyError:
                continue
            
            if action:
                rCube_str = cube.bddPattern().cubeString()[start_rvar_idx:end_rvar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(rCube_str, self.rVars)]
                except KeyError:
                    continue
            
            # if you made it till here then print stuff or store them
            if action:
                states_bookkeeping.append((state, action_str, val))
            else:
                states_bookkeeping.append((state, val))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))

        
        return states_action_pairs


    def convert_full_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, action: bool = False, verbose: bool = False) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
        Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the action as well, set action flag to True. 

         Here the input dd is assumed to be a fully defined cube (latches as well prime latches).
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches)
            relevant_vars.extend(self.prime_latches)
        if action:
            relevant_vars.extend(self.rVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.rVars + self.prime_latches)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.rVars + self.prime_latches)
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars + self.prime_latches)

        # create prime turn abstraction cube
        prime_tConf_exist_cube = reduce(lambda a, b: a & b, self.prime_xVars + self.rVars + self.latches)
        prime_kConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_xVars[len(self.prime_kVars):] + self.rVars + self.latches)
        
        # create existential abstraction cubes
        prime_rConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars[len(self.prime_kVars)+len(self.prime_pVars):] + self.rVars + self.latches)

        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        prime_bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.rVars + self.prime_latches)
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.rVars + self.latches)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.rVars + self.prime_latches) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.rVars + self.latches) & reduce(lambda x, y: x & y, self.prime_bVars_cubes[:bidx] + self.prime_bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        prime_states_action_pairs = []
        state_action_prime_pairs = []
        
        for cube, _ in cubes:
            state = None
            prime_state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_cube_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_tConf_cube_str = cube.existAbstract(prime_tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_rConf_cube_str = cube.existAbstract(prime_rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            prime_kConf_cube_str = cube.existAbstract(prime_kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = (self.tVar_map.inv[tConf_cube_str], self.kVar_map.inv[kConf_cube_str], self.pVar_map.inv[rConf_cube_str], box_states)
                states_action_pairs.append([(self.tVar_map.inv[tConf_cube_str], self.kVar_map.inv[kConf_cube_str], self.pVar_map.inv[rConf_cube_str], box_states), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if action:
                rCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(rCube_str, self.rVars)]
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
                prime_states_action_pairs.append([(self.tVar_map.inv[prime_tConf_cube_str], self.kVar_map.inv[prime_kConf_cube_str], self.pVar_map.inv[prime_rConf_cube_str], prime_box_states), None])
                prime_state = (self.tVar_map.inv[prime_tConf_cube_str], self.kVar_map.inv[prime_kConf_cube_str], self.pVar_map.inv[prime_rConf_cube_str], prime_box_states)
            except KeyError:
                continue

            # if you made it till here then print stuff
            state_action_prime_pairs.append((state, action_str, prime_state))
        
        if verbose:
            print(tabulate(state_action_prime_pairs, headers=['state', 'robot action', 'prime state']))
        
        return states_action_pairs, prime_states_action_pairs



    def check_valid_human_move(self, curr_state: List[str], action: str) -> bool:
        """
         A helper function to check if a human move action is valid given the current state and human action.
         Things to check:
         1. if human is moving a box to a location that is already occupied by another box, return hmove noop
         2. if human is moving a box that is currently at a restricted location, return hmove noop
         3. if human is moving a box to a restricted location, return hmove noop
         4. if human has reached the max number of interventions in this robot turn, return hmove noop
         4. else return the human move action as is.

        """
        box_idx = 3
        human_move_idx = 1
        if action.startswith('hmove noop'):
            return 'hmove noop'
        elif action.startswith('hmove'):
            # Chekcing point 1.
            b_idx = action.split(' ')[1]
            hmove_to_loc = action.split(' ')[2]
            # the box str will be of the form b0 l1, b1 l3, etc..
            split_str = curr_state[box_idx].split(', ')
            # check if the location is already occupied by another box
            for s in split_str:
                loc = s.split(' ')[1]
                if loc == hmove_to_loc:
                    return 'hmove noop'
            
            # checking point 2
            curr_box__loc = split_str[int(b_idx[-1])].split(' ')[1]
            if int(curr_box__loc[1:]) in self.restricted_human_locs:
                return 'hmove noop'
            
            # checking point 3
            if int(hmove_to_loc[1:]) in self.restricted_human_locs:
                return 'hmove noop'

            # checking point 4
            kval = int(curr_state[human_move_idx][-1])
            if (self.ratio == 0 and kval == 0) or kval >= self.ratio:
                return 'hmove noop'
        else:
            print("Unknown human action. Cannot proceed!!")
            sys.exit(-1)
        
        return action

    def get_next_state_human(self, curr_state: List[str], action: str) -> Tuple[ADD, str] :
        """
         A helper function to get the next state under human action given the current state and human action.
        """
        turn_var_idx = 0
        human_move_idx = 1
        rConf_idx = 2
        box_idx = 3
        action: str = self.check_valid_human_move(curr_state=curr_state, action=action)
        if action.startswith('hmove noop'):
            # boxes do not change location but we need to update robot confguration
            if curr_state[rConf_idx].startswith('in-transit'):
                # the rConf state is of the form in-transit from_loc b_idx
                to_box = curr_state[rConf_idx].split(' ')[2]
                curr_state[rConf_idx] = f'to-obj {to_box}'
            elif curr_state[rConf_idx].startswith('in-transfer'):
                # the rConf state is of the form in-transfer from_loc to_loc
                to_loc = curr_state[rConf_idx].split(' ')[2]
                curr_state[rConf_idx] = f'holding {to_loc}'
            elif curr_state[rConf_idx].startswith('ready') or curr_state[rConf_idx].startswith('holding'):
                pass
            else:
                print("Unknown robot configuration during human move action. Cannot proceed!!")
                sys.exit(-1)
            
            # if did not move any box then the K var resets to 0
            curr_state[human_move_idx] = 'k0'
        
        elif action.startswith('hmove'):
            # update box configuration
            b_idx = action.split(' ')[1]
            kval = int(curr_state[human_move_idx][-1])
            hmove_to_loc = action.split(' ')[2]
            # the box str will be of the form b0 l1, b1 l3, etc..
            split_str = curr_state[box_idx].split(', ')
            split_str[int(b_idx[-1])] = f'{b_idx} {hmove_to_loc}'
            curr_state[box_idx] = ', '.join(split_str)
            # update robot configuration based on current robot configuration if the human moves a box        
            if curr_state[rConf_idx].startswith('in-transit'):
                # the rConf state is of the form in-transit from_loc b_idx
                from_loc = curr_state[rConf_idx].split(' ')[1]
                curr_state[rConf_idx] = f'ready {from_loc}'
            elif curr_state[rConf_idx].startswith('in-transfer'):
                # the rConf state is of the form in-transfer from_loc to_loc
                from_loc = curr_state[rConf_idx].split(' ')[1]
                curr_state[rConf_idx] = f'holding {from_loc}'
            elif curr_state[rConf_idx].startswith('ready') or curr_state[rConf_idx].startswith('holding'):
                pass
            else:
                print("Unknown robot configuration during human move action. Cannot proceed!!")
                sys.exit(-1)
            # update K var - increment K by 1
            if kval < self.ratio:
                curr_state[human_move_idx] = f'k{kval + 1}'
            
        # update state turn
        curr_state[turn_var_idx] = 'human' if curr_state[turn_var_idx] == 'robot' else 'robot'
        # convert the string of boxes location to sperate state
        split_str = curr_state[box_idx].split(', ')
        return self.tVar_map_sym[curr_state[turn_var_idx]] & self.kVar_map_sym[curr_state[human_move_idx]] & \
              self.xVar_map_sym[curr_state[rConf_idx]] & reduce(lambda a, b: a & b, [self.xVar_map_sym[s] for s in split_str]), action


    def get_next_state_robot(self, curr_state: List[str], action: str, **kwargs) -> ADD:
        """
         A helper function to get the next state under robot action given the current state and robot action.
        """
        turn_var_idx = 0
        human_move_idx = 1
        rConf_idx = 2
        box_idx = 3
        # if action is transit then, update the robot configuration
        if action.startswith('transit'):
            assert curr_state[rConf_idx].startswith('ready'), "Make sure the robot is ready to transit!!!"
            from_loc = curr_state[rConf_idx].split(' ')[1]
            b_idx = action.split(' ')[1]
            # from ready you evolve to in-transit
            curr_state[rConf_idx] = f'in-transit {from_loc} {b_idx}'
        
        # if action is grasp then, update the robot configuration and box configuration
        elif action.startswith('grasp'):
            if curr_state[rConf_idx].startswith('to-obj'):
                box: str = curr_state[rConf_idx].split(' ')[1]
                b_idx = int(box[-1])
                # the box str will of the form b0 l1, b1 l3, etc..
                split_str = curr_state[box_idx].split(', ')
                l_idx = split_str[b_idx].split(' ')[1] 
                split_str[b_idx] = f'{box} l0'
                curr_state[box_idx] = ', '.join(split_str)
                # update the robot configuration
                curr_state[rConf_idx] = f'holding {l_idx}'
            elif curr_state[rConf_idx].startswith('ready'):
                l_idx = curr_state[rConf_idx].split(' ')[1]
                curr_state[rConf_idx] = f'holding {l_idx}'

                # need to find which box is at l_idx
                split_str = curr_state[box_idx].split(', ')
                found_box: bool = False
                for bidx, b in enumerate(split_str):
                    if b.endswith(l_idx):
                        box = b.split(' ')[0]
                        split_str[bidx] = f'{box} l0'
                        found_box = True
                        break
                if not found_box:
                    print("No box found at the location where robot is trying to grasp. Cannot proceed!!")
                    sys.exit(-1)
                curr_state[box_idx] = ', '.join(split_str)
            else:
                print("Unknown robot configuration during grasp action. Cannot proceed!!")
                sys.exit(-1)

        # if action is release then, update the robot configuration and box configuration
        elif action.startswith('release'):
            assert curr_state[rConf_idx].startswith('holding'), "Make sure the robot is in holding status when releasing!!!"
            l_idx = curr_state[rConf_idx].split(' ')[1]
            # change the box location from l0 to l_idx
            split_str = curr_state[box_idx].split(', ')
            for bidx, b in enumerate(split_str):
                if b.endswith('l0'):
                    box = b.split(' ')[0]
                    split_str[bidx] = f'{box} {l_idx}'
                    break
            # update the robot configuration
            curr_state[box_idx] = ', '.join(split_str)
            curr_state[rConf_idx] = f'ready {l_idx}'
        
        # if action is transfer then, update the robot configuration 
        elif action.startswith('transfer'):
            assert curr_state[rConf_idx].startswith('holding'), "Make sure the robot is holding when transfering to another loc!!!"
            from_loc = curr_state[rConf_idx].split(' ')[1]
            to_loc = action.split(' ')[1]
            curr_state[rConf_idx] = f'in-transfer {from_loc} {to_loc}'

        else:
            print("Unknown action. Cannot compute next state!!")
            sys.exit(-1)

        # update state turn
        curr_state[turn_var_idx] = 'human' if curr_state[turn_var_idx] == 'robot' else 'robot'
        # convert the string of boxes location to sperate state
        split_str = curr_state[box_idx].split(', ')
        return self.tVar_map_sym[curr_state[turn_var_idx]] & self.kVar_map_sym[curr_state[human_move_idx]] &  \
              self.xVar_map_sym[curr_state[rConf_idx]] & reduce(lambda a, b: a & b, [self.xVar_map_sym[s] for s in split_str])
    
    
    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a give strategy
        """
        curr_state = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]

        while (curr_state & self.goal_latch.existAbstract(self.tVar[0])).isZero():
            curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state, state_flag=True, action=False, table_header=False, verbose=verbose)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval: int = list((curr_state & self.comp_winning_states).generate_cubes())[0][1]
            except IndexError:
                opt_sval: int = 0

            # get the action to be taken at the current state
            act_cube: BDD = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string]
            except KeyError:
                print("No robot action found!!")
                return
        
            turn = 'robot' if curr_state_exp[0][0][0][0] == 'robot' else'human'
           
            # get the next state
            if turn == 'robot':
                curr_state: ADD = self.get_next_state_robot(list(curr_state_exp[0][0][0]), act_name)
            elif turn == 'human':
                curr_state, act_name = self.get_next_state_human(list(curr_state_exp[0][0][0]), act_name)
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Robot Action: {act_name}") if turn == 'robot' else print(f"Human Action: {act_name}")
    

    def symbolic_min_abstract(self, add_function, variables_to_abstract: List[ADD]):
        """
        Eliminates variables by taking the minimum of the cofactor branches.
         This replaces explicit loops over action lists.
        """
        result_add = add_function

        for var_add in variables_to_abstract:
            pos_cofactor = result_add.cofactor(var_add)
            neg_cofactor = result_add.cofactor((~var_add))
            result_add = pos_cofactor.min(neg_cofactor) 
            
        return result_add
    

    def symbolic_max_abstract(self, add_function, variables_to_abstract: List[ADD]) -> ADD:
        """
        Eliminates variables by taking the maximum of the cofactor branches.
         This replaces explicit loops over action lists.
        """
        result_add = add_function

        for var_add in variables_to_abstract:
            pos_cofactor = result_add.cofactor(var_add)
            neg_cofactor = result_add.cofactor((~var_add))
            result_add = pos_cofactor.max(neg_cofactor) 
            
        return result_add

    
    def compute_reachable_states(self,
                                 monolithic_trans_dd: ADD,
                                 latches: List[ADD], prime_latches: List[ADD],
                                 act_vars: List[ADD], verbose: bool = False,
                                 print_states: bool = False) -> ADD:
        """
         A method to compute the set of reachable states in the Graph.
        """
        latches_cube = reduce(lambda a, b: a & b, latches)
        action_cube = reduce(lambda a, b: a & b, act_vars)
        
        latches_cube_bdd = latches_cube.bddPattern()
        action_cube_bdd = action_cube.bddPattern()
        curr_state_action_cube_bdd: BDD = latches_cube_bdd & action_cube_bdd
        game_latches_list_bdd: BDD = [var.bddPattern() for var in latches]
        game_prime_latches_list_bdd: BDD = [var.bddPattern() for var in prime_latches]
        open_list = []
        closed = self.manager.bddZero()

        # maintain a common layering number
        layer_num = 0
        open_list.append(self.init_latch.bddPattern())

        # convert the monolithic ADD into a BDD for faster operations
        bdd_monolithic_valid_full_gobr_trns: BDD = monolithic_trans_dd.bddPattern()
        if verbose:
            print("********************Starting Game Reachability Computation********************")
        
        # perform BFS like exploration
        while True:
            # remove all states that have been explored
            open_list[layer_num] = open_list[layer_num] & ~closed
            if not open_list[layer_num].isZero():
                if verbose:
                    print(f"********************Layer: {layer_num}**************************")
                    if print_states:
                        self.convert_cube_to_state_ADD(open_list[layer_num].toADD(), action=False, verbose=True)
                # Add states to be expanded next to already expanded states
                closed |= open_list[layer_num]
            
                # preimage: ADD = self.compute_regret_preimage(reachable_states)
                image_prime: BDD = bdd_monolithic_valid_full_gobr_trns.andAbstract(open_list[layer_num], curr_state_action_cube_bdd)
                image: BDD = image_prime.swapVariables(game_prime_latches_list_bdd, game_latches_list_bdd)
                open_list.append(image)

                layer_num += 1
            else:
                if verbose:
                    print("********************Done Computing Game Reachable States********************")
                return closed.toADD()
    

    def create_sys_env_transition_relations(self):
        """
         A method to create separate transition relations for the system (robot) and environment (human) actions.
        """
        # post-process to get TR for robot and Human actions separately
        self.sys_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
        self.env_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
        for tr_key, tr_dd in self.transition_relation.items():
            self.sys_transition_relation[tr_key] = tr_dd & self.tVar_map_sym['robot']
            self.env_transition_relation[tr_key] = tr_dd & self.tVar_map_sym['human']
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.latches, self.prime_latches)
        preimage = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        self.iteration_bookkeeping.append([curr_winning_states_primed.size(), preimage.size()])

        return preimage

    
    def decompose_winning_states(self, monolithic_add: ADD, add_threshold: int, layer: int, c_max: int, debug: bool = False) -> List[ADD]:
        """
         A method tthat decomposes the monolithic ADD into vectors of ADDs of smaller size.  
         
         Given the threhold, we use leaf values to obtain the partitioning of the monolithic ADD. 
         This is a heuristic method to decompose the monolithic ADD.
        """
        winning_states_decomposed = defaultdict(lambda: self.manager.plusInfinity()) 
        bucket_iter = 0
        curr_bucket_add = self.manager.plusInfinity()
        # prev_bucket_add = self.manager.plusInfinity()
        for sval in range(0, (layer * c_max) + 1):
            decomposed_add: BDD = monolithic_add.bddInterval(sval, sval)
            if not decomposed_add.isZero():
                curr_bucket_add = decomposed_add.toADD().ite(self.manager.addConst(sval), winning_states_decomposed[bucket_iter])
                if curr_bucket_add.size() < add_threshold:
                    winning_states_decomposed[bucket_iter] = curr_bucket_add
                else:
                    bucket_iter += 1
                    winning_states_decomposed[bucket_iter] = decomposed_add.toADD().ite(self.manager.addConst(sval), winning_states_decomposed[bucket_iter])
                    if debug:
                        print(f"Size of the decomposed add in bucket {bucket_iter} is {winning_states_decomposed[bucket_iter - 1].size()}")
        winning_states_decomposed[bucket_iter] = curr_bucket_add
        return winning_states_decomposed


    def compute_preimage_decomposed(self, curr_winning_states: Dict[int, ADD], add_max_size: int, add_threshold: int, layer: int, debug: bool = False) -> Dict[int, ADD]:
        if curr_winning_states.size() < add_max_size:
            return self.compute_preimage(curr_winning_states)
        
        if debug:
            print("Decomposing the winning states ADD as it is above the threshold size")
            print("Size of the winning states ADD before decomposition is ", curr_winning_states.size())
        c_max: int = int(list(self.weight.findMax().generate_cubes())[0][1])
        decomposed_add = self.decompose_winning_states(curr_winning_states, add_threshold, layer=layer, c_max=c_max, debug=debug)
        preimage = self.manager.plusInfinity()
        for decomposed_curr_winning_states in decomposed_add.values():
            preimage = preimage.min(self.compute_preimage(decomposed_curr_winning_states))
        return preimage

    

    def compute_preimage_optimized(self, curr_winning_states: ADD) -> Tuple[ADD, ADD]:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.latches, self.prime_latches)
        pre_sys = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.sys_transition_relation.values()))
        pre_env = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.env_transition_relation.values()))

        return pre_sys, pre_env
    

    def post_process_transition_relation_reachable(self):
        """
         A method to post-process the transition relation after all action rules and frame axioms have been added. HEre we restrict the TR to the set of state that are reachable in the game.
        """
        print("******************Post-processing the transition relation to only include reachable states******************")
        for tr_key in self.transition_relation.keys():
            self.transition_relation[tr_key] &= self.care_states
    

    def solve(self, verbose: bool = False, cooperative_game: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.
        """
        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states = self.manager.plusInfinity().min(goal)
        if self.only_reachable_states:
            self.post_process_transition_relation_reachable()

        # print the initial winning states
        if verbose:
            print("Initial Winning States:")
            # by default generate cubes does not retuen cubes that point to 0 leaf. 
            # So, we manually convert the 0 leaf to a cube with leaf value 1 here for printing.
            self.convert_cube_to_state_ADD(curr_winning_states.bddInterval(0, 0).toADD(), action=False, verbose=True)
        
        # intialize the iteration counter
        layer = 0
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        # iteration bookkeeping
        self.iteration_bookkeeping = []

        while True:
            print(f"**************************Layer: {layer}**************************")
            # lets add nodes in the graph before and after and read the peak node count
            # preimage: ADD = self.compute_preimage(curr_winning_states)

            # let try the new decomposition preimage method
            preimage: ADD = self.compute_preimage_decomposed(curr_winning_states, add_max_size=2000, add_threshold=2000, layer=layer, debug=False)
            # add the action costs associated with the robot actions
            # assert preimage.compare(preimage_test, 2), "Make sure the new decomposed preimage method is correct!!"   
            preimage = preimage + self.weight
            # print("*****************************Current Preimage:*****************************")
            # self.convert_cube_to_state_ADD(preimage, state_flag=True, action=True, verbose=verbose)
            
            # take min over Sys player states; as invalid actions and human action are mapped to inf, they will not affect the min operation
            if cooperative_game:
                next_winning_states = self.symbolic_min_abstract(preimage, self.rVars)
            else:
                next_winning_states = self.compute_min_max_preimage(preimage, valid_human_action_mask=valid_human_action_mask)
            
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, action=False, verbose=verbose)
            
            if curr_winning_states.compare(next_winning_states, 2):
                self.logger.comp_time['Iterations'] = layer
                self.logger.comp_time['Preimage_size'] = {idx: e for idx, e in enumerate(self.iteration_bookkeeping)}
                print("**************************Reached fixpoint**************************")
                if curr_winning_states.restrict(self.init_latch) != self.manager.plusInfinity():
                    if curr_winning_states.restrict(self.init_latch) == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    # return preimage if init_val < math.inf else None
                    if init_val < math.inf:
                        return preimage.min(goal), curr_winning_states
                    else:
                        return None, None
                else:
                    print(f"No Winning Strategy Exists!! The State value is {math.inf}")
                return None, None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states

    
    def compute_min_max_preimage(self, preimage: ADD, valid_human_action_mask: ADD) -> ADD:
        # robot_states = preimage.restrict(self.tVar_map_sym['robot'])
        # robot_states = preimage & self.tVar_map_sym['robot']
        robot_states = preimage.cofactor(self.tVar_map_sym['robot'])
        next_winning_states_robot = self.symbolic_min_abstract(robot_states, self.rVars)
        
        # take max over Env player states; but first map the invalid human actions and robot action from these stares to -inf
        # human_states = preimage.restrict(self.tVar_map_sym['human'])
        # human_states = preimage & self.tVar_map_sym['human']
        human_states = preimage.cofactor(self.tVar_map_sym['human'])
        preimage_for_max = valid_human_action_mask.ite(human_states, self.manager.minusInfinity()) 
        next_winning_states_env = self.symbolic_max_abstract(preimage_for_max, self.rVars)

        next_winning_states = self.tVar[0].ite(next_winning_states_robot, next_winning_states_env)
        return next_winning_states


    def solve_optimized(self, verbose: bool = False, cooperative_game: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.
        """
        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states = self.manager.plusInfinity().min(goal)
        if self.only_reachable_states:
            self.post_process_transition_relation_reachable()
        
        self.create_sys_env_transition_relations()
        # intialize the iteration counter
        layer = 0
        valid_env_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        while True:
            print(f"**************************Layer: {layer}**************************")
            pre_sys, pre_env = self.compute_preimage_optimized(curr_winning_states)

            # add the action costs associated with the robot actions   
            pre_sys += self.weight
            pre_env += self.weight

            if cooperative_game:
                next_winning_states_sys = self.symbolic_min_abstract(pre_sys, self.rVars)
                next_winning_states_env = self.symbolic_min_abstract(pre_env, self.rVars)
            else:
                next_winning_states_sys = self.symbolic_min_abstract(pre_sys, self.rVars)
                pre_env_masked = valid_env_action_mask.ite(pre_env, self.manager.minusInfinity())
                next_winning_states_env = self.symbolic_max_abstract(pre_env_masked, self.rVars)
            
            next_winning_states = self.tVar[0].ite(next_winning_states_sys, next_winning_states_env)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, action=False, verbose=verbose)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if curr_winning_states.restrict(self.init_latch) != self.manager.plusInfinity():
                    if curr_winning_states.restrict(self.init_latch) == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    preimage = pre_sys.min(pre_env)
                    if init_val < math.inf:
                        return preimage.min(goal), curr_winning_states
                    else:
                        return None, None
                else:
                    print(f"No Winning Strategy Exists!! The State value is {math.inf}")
                return None, None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def convert_mono_tr_to_action_tr(self):
        # loop throught the transition relation and separate them based on action
        for act_dd in self.action_map_sym.values():
            # loop over the tr and rertain these action
            act_ls = []
            for tr_bdd in self.transition_relation.values():
                ts_action_dd = tr_bdd & act_dd
                act_ls.append(ts_action_dd.bddPattern())
            self.ts_bdd_transition_fun_list.append(act_ls)
    

    def convert_monolithic_add_to_bdd_buckets(self, monolithic_add: ADD, layer: int, c_max: int) -> Dict[int, BDD]:    
        win_state_bucket: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        
        # convert the winning states into buckets of BDD
        _max_interval_val = layer * c_max
        for sval in range(_max_interval_val + 1):
            # get the states with state value equal to sval and store them in their respective bukcets
            win_sval = monolithic_add.bddInterval(sval, sval)

            if not win_sval.isZero():
                win_state_bucket[sval] |= win_sval
        
        return win_state_bucket
    

    def iros23_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        bookkeeping_size = defaultdict(lambda: defaultdict(list))
        for idx, tr_action in enumerate(self.ts_bdd_transition_fun_list):
            # we get from the new weightr dictionary
            for sval, succ_states in win_state_bucket.items():
                succ_states_prime = succ_states.swapVariables(self.latches_bdd, self.prime_latches_bdd)
                pre_states: BDD = succ_states_prime.vectorCompose(self.prime_latches_bdd, tr_action)

                if not pre_states.isZero():
                    assert pre_buckets[sval] & pre_states == self.manager.bddZero(), "Make sure there are no overlapping states in the pre buckets..."
                    pre_buckets[sval] |= pre_states
                    bookkeeping_size[idx][sval] = [succ_states_prime.size(), pre_states.size()]
                                            
        self.iteration_bookkeeping.append(bookkeeping_size)
        # unions of all predecessors
        if not return_bdd:
            preimage = self.manager.plusInfinity()
            for sval, add_bucket in pre_buckets.items():
                preimage = add_bucket.toADD().ite(self.manager.addConst(sval), preimage)
            
            return preimage
        return pre_buckets
    

    def get_states_per_cost(self):
        """
         A helper function that takes in the ADD weight abd return a vector of 0-1 BDD per cost.
        """
        for w in self.weight_dict.values():
            self.states_per_cost[w] = self.weight.bddInterval(w, w)
        
        # manually add human action to cost zero
        self.states_per_cost[0] |= self.tVar_map_sym['human'].bddPattern() & reduce(lambda x, y: x | y, self.env_action_cube_list_bdd)
        

    def compute_min_preimage_pure_bdd(self, preimage: Dict[int, BDD]) -> Dict[int, BDD]:
        minmin_preimage = defaultdict(self.manager.bddZero) 
        rVars_cube_bdd = self.rVars_cube.bddPattern()
        states_action_pairs: BDD = reduce(lambda x, y: x | y, preimage.values())
        states: BDD = states_action_pairs.existAbstract(rVars_cube_bdd)
        for sval in sorted(preimage.keys()):
            # intersect with finite valued states for Sys and Env player
            sval_to_keep = preimage[sval].existAbstract(rVars_cube_bdd) & states
            minmin_preimage[sval] |= sval_to_keep
            states &= ~sval_to_keep
        assert states.isZero() == True, "Error in computing min for system and env states"

        return minmin_preimage


    def compute_min_max_preimage_pure_bdd(self, preimage: Dict[int, BDD], debug: bool = False) -> Dict[int, BDD]:
        minmax_preimage = defaultdict(self.manager.bddZero)
        rVars_cube_bdd = self.rVars_cube.bddPattern()
        human_turn_bdd: BDD = self.tVar_map_sym['human'].bddPattern()
        states_action_pairs: BDD = reduce(lambda x, y: x | y, preimage.values())
        # now take univ abstraction to remove edges to states with infinity value
        states: BDD = states_action_pairs.existAbstract(rVars_cube_bdd)
        
        robot_state_bdd: BDD = states & ~human_turn_bdd

        # remove Env state that do not have a finite value value under all actions
        human_state_w_inf_val = human_turn_bdd & reduce(lambda x, y: x | y, self.env_action_cube_list_bdd) & ~states_action_pairs
        human_state_w_inf_val = human_state_w_inf_val.existAbstract(rVars_cube_bdd)
        
        # debug states without inf value are
        human_finite_valued_states = states & human_turn_bdd & ~human_state_w_inf_val
        
        for sval in sorted(preimage.keys(), reverse=True):
            # intersect with human finite valued states
            sval_to_keep = preimage[sval].existAbstract(rVars_cube_bdd) & human_finite_valued_states
            minmax_preimage[sval] |= sval_to_keep
            human_finite_valued_states &= ~sval_to_keep
        
        if debug:
            assert human_finite_valued_states.isZero() == True, "Error in computing max for human states"

        for sval in sorted(preimage.keys()):
            # intersect with human finite valued states
            sval_to_keep = preimage[sval].existAbstract(rVars_cube_bdd) & robot_state_bdd
            minmax_preimage[sval] |= sval_to_keep
            robot_state_bdd &= ~sval_to_keep
        
        if debug:
            assert robot_state_bdd.isZero() == True, "Error in computing min for system states"

        return minmax_preimage


    def compute_min_goal_states(self, preimage: Dict[int, BDD], goal: Dict[int, BDD]) -> Dict[int, BDD]:
        for goal_sval in sorted(goal.keys()):
            for sval in sorted(preimage.keys()):
                # if there exists states in goal state, then we override the state value in preimage
                sval_to_update = preimage[sval] & goal[goal_sval]
                preimage[sval] &= ~sval_to_update
                preimage[goal_sval] |= sval_to_update
        return preimage


    def check_reached_fixpoint_bdd(self, curr_winning_states: Dict[int, BDD], next_winning_states: Dict[int, BDD]) -> bool:
        if set(next_winning_states.keys()) != set(curr_winning_states.keys()):
            return False
        else:
            for new_bdd, pre_bdd in zip(next_winning_states.values(), curr_winning_states.values()):
                if not pre_bdd.compare(new_bdd, 2):
                    return False
        return True
    

    def convert_vector_of_bdd_to_add(self, bdd_vector: Dict[int, BDD]) -> ADD:
        """
         A helper function that converts a vector of BDDs to an ADD. USed in puree BDD solver method for 
          (1) printing the winning states
          (2) returning the preimage strategy
        """
        result_add = self.manager.plusInfinity()
        for sval, sbdd in bdd_vector.items():
            result_add = sbdd.toADD().ite(self.manager.addConst(sval), result_add)
        return result_add
    

    def hybrid_solve(self, verbose: bool = False, cooperative_game: bool = False) ->  Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.

          This mwthod replace is different from te pure ADD approach as it first converts the current winning states 
           into BDD buckets based on the state values and then computes the preimage using BDD vectorCompose operation.
        """
        # create Partitiotned TR based on actions
        self.convert_mono_tr_to_action_tr()

        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states = self.manager.plusInfinity().min(goal)
        next_winning_states = self.manager.plusInfinity()
        # intialize the iteration counter
        layer = 0
        c_max: int = 1        
        
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        # iteration bookkeeping
        self.iteration_bookkeeping = []

        while True:
            print(f"**************************Layer: {layer}**************************")
            win_state_bucket = self.convert_monolithic_add_to_bdd_buckets(monolithic_add=curr_winning_states, layer=layer, c_max=c_max)
            preimage = self.iros23_compute_preimage(win_state_bucket=win_state_bucket)
                    
            # add the action costs associated with the robot actions
            preimage = preimage + self.weight
            # take min over Sys player states; as invalid actions and human action are mapped to inf, they will not affect the min operation
            if cooperative_game:
                next_winning_states = self.symbolic_min_abstract(preimage, self.rVars)
            else:
                next_winning_states = self.compute_min_max_preimage(preimage=preimage, valid_human_action_mask=valid_human_action_mask)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, action=False, verbose=verbose)
            
            
            if next_winning_states.compare(curr_winning_states, 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                self.logger.comp_time['action_list'] = list(self.action_map_sym.keys())
                self.logger.comp_time['Iterations'] = layer
                self.logger.comp_time['Preimage_size'] = {idx: e for idx, e in enumerate(self.iteration_bookkeeping)}
                if curr_winning_states.restrict(self.init_latch) != self.manager.plusInfinity():
                    if curr_winning_states.restrict(self.init_latch) == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    strategy = preimage.min(goal)
                    # return preimage if init_val < math.inf else None
                    if init_val < math.inf:
                        return strategy, curr_winning_states
                    else:
                        return None, None
                else:
                    print(f"No Winning Strategy Exists!! The State value is {math.inf}")
                return None, None
            
            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    
    
    def pure_bdd_solve(self, verbose: bool = False, cooperative_game: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.
        """
        # create Partitioned TR based on actions
        self.convert_mono_tr_to_action_tr()
        self.get_states_per_cost()
        goal_states_buckets = defaultdict(lambda: self.manager.bddZero())

        # initialize goal state with 0 state value and add it to the winning region
        curr_winning_states = defaultdict(lambda: self.manager.bddZero())
        next_winning_states = defaultdict(lambda: self.manager.bddZero())
        
        # initialize the goal state bucket with 0 cost
        curr_winning_states[0] |= self.goal_latch.bddPattern()
        goal_states_buckets[0] |= self.goal_latch.bddPattern()
       
        # intialize the iteration counter
        layer = 0

        # iteration bookkeeping
        self.iteration_bookkeeping = []

        while True:
            print(f"**************************Layer: {layer}**************************")
            # compute preimage
            vector_preimage: Dict[int, BDD] = self.iros23_compute_preimage(win_state_bucket=curr_winning_states, return_bdd=True)
            
            # add the action costs associated with the robot actions
            for sCost, sbdd in self.states_per_cost.items():
                for pre_sVal, pre_sbdd in vector_preimage.items():
                    total_cost: int = sCost + pre_sVal
                    
                    common_states: BDD = sbdd & pre_sbdd
                    if not common_states.isZero():
                        next_winning_states[total_cost] |= common_states

            # take min over Sys player states; as invalid actions and human action are mapped to inf, they will not affect the min operation
            if cooperative_game:
                next_winning_states_opt = self.compute_min_preimage_pure_bdd(preimage=next_winning_states)
            else:
                next_winning_states_opt = self.compute_min_max_preimage_pure_bdd(preimage=next_winning_states, debug=False)
            # retain the min over goal states - here all goal states are at 0 cost
            next_winning_states_opt = self.compute_min_goal_states(preimage=next_winning_states_opt, goal=goal_states_buckets)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                # unions of all predecessors along with their state values - ADD used for easy printing only
                preimage = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states_opt)
                self.convert_cube_to_state_ADD(preimage, action=False, verbose=verbose)
            
            
            if self.check_reached_fixpoint_bdd(curr_winning_states=curr_winning_states, next_winning_states=next_winning_states_opt):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                self.logger.comp_time['action_list'] = list(self.action_map_sym.keys())
                self.logger.comp_time['Iterations'] = layer
                self.logger.comp_time['Preimage_size'] = {idx: e for idx, e in enumerate(self.iteration_bookkeeping)}
                init_val = math.inf
                for sval, sbdd in curr_winning_states.items():
                    if sbdd & self.init_latch.bddPattern() != self.manager.bddZero():
                        init_val: int = sval
                        print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                        break
                self.comp_winning_states = self.convert_vector_of_bdd_to_add(bdd_vector=curr_winning_states)
                # post process the strategy to return as monolithic ADD that corresponds to strategy
                strategy: ADD = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states)
                goal: ADD = self.convert_vector_of_bdd_to_add(bdd_vector=goal_states_buckets)
                if init_val < math.inf:
                    return strategy.min(goal), self.comp_winning_states
                else:
                    print(f"No Winning Strategy Exists!! The State value is {math.inf}")
                    return None, None
            
            # update the counter
            layer += 1

            # swap the winning states; can't do  curr_winning_states = next_winning_states_opt as python is pass by value of reference
            curr_winning_states = defaultdict(lambda: self.manager.bddZero())
            for sval in next_winning_states_opt.keys():
                curr_winning_states[sval] |= next_winning_states_opt[sval]


    
    def test_pre_image_restricted_human_moves(self):
        self.create_sys_env_transition_relations()
        # goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['in-transit b0'] & self.xVar_map_sym['b0 l1'] #& self.kVar_map_sym['k0'] #& self.xVar_map_sym['b1 l3']
        # goal_cube = self.tVar_map_sym['human'] & self.kVar_map_sym['k0'] & self.xVar_map_sym['in-transfer l2'] & self.xVar_map_sym['b0 l0']
        goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['in-transit b0'] & self.xVar_map_sym['b0 l1'] #& self.kVar_map_sym['k1']
        print('Goal state:', goal_cube)
        # compute preimage 
        From = goal_cube.swapVariables(self.latches, self.prime_latches)
        preimage = From.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        print("************************Using Old TR*******************************")
        print('Preimage (Old TR): ', preimage)
        self.convert_cube_to_state_ADD(preimage, action=True, verbose=True)

        print("************************Using New TR*******************************")
        new_preimage = From.vectorCompose(self.prime_latches, list(self.sys_transition_relation.values()))
        print('Preimage (New TR): ', new_preimage)
        self.convert_cube_to_state_ADD(new_preimage, action=True, verbose=True)
        print("Done Computing Preimage")



if __name__ == "__main__":
    # test_dynamic_franka_world()
    # sys.exit(0)
    
    # setting things up
    boxes = 2
    locs = 10
    ratio = 1
    # init = ['ready l2', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5']
    # goal = [['b0 l1']]
    init = ['ready l2', 'b0 l2', 'b1 l4']
    goal = [['b0 l1']]
    # goal = ['holding l1', 'b0 l0']
    # goal = [['b0 l1', 'b1 l3'], ['b0 l1', 'b1 l4']]
    # human_locs = range(1, locs + 1)
    # human_locs =  [3, 4] #range(1, locs + 1)
    human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = []
    fw_tb = FrankaWorldDynamicRatioTurnBased(boxes=boxes, locs=locs, ratio=ratio, init=init, goal=goal, restricted_human_locs=human_locs, enable_reordering=True)

    print('****************xVars Map:****************')
    for k, v in fw_tb.xVar_map.items():
        print(f"{k} : {v}")

    print('****************Action Map:****************')
    for k, v in fw_tb.action_map.items():
        print(f"{k} : {v}")

    print("*****************Ratio Map:*****************")
    for k, v in fw_tb.kVar_map.items():
        print(f"{k} : {v}")
    
    # print("*****************Valid State Conf*****************")
    # fw_tb.convert_cube_to_state_ADD(fw_tb.monolithic_relevant_box_preds, robot_action=False, human_action=False)
    # sys.exit(-1)
    # print the number of explicit states
    sys_states = (ratio + 1)*(pow(locs + 1, 3) + boxes)*(math.factorial(locs+1) // math.factorial(locs+1 - boxes))
    env_states = (ratio + 1)*(pow(locs + 1, 2) + boxes*(locs+1))*(math.factorial(locs+1) // math.factorial(locs+1 - boxes))
    print("Total num of explicit states: ", env_states + sys_states)


    print("Total num of latches: ", len(fw_tb.latches))
    print("Total num of prime latches: ", len(fw_tb.prime_latches))
    print("Total boolean vars: ", len(fw_tb.latches) + len(fw_tb.prime_latches))

    tic = time.time()
    fw_tb.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # fw_tb.test_pre_image_restricted_human_moves()
    tic = time.time()
    strategy = fw_tb.solve(verbose=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        fw_tb.roll_out_strategy(strategy=strategy, verbose=True)