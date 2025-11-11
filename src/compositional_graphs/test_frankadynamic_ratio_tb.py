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

from typing import List, Dict, Tuple, Set, Union
from functools import reduce
from itertools import product
from collections import defaultdict


from bidict import bidict
# from .symbolic_partitioned_dfa import SymbolicPartitionedDFA
from cudd import Cudd, ADD, BDD, REORDER_GROUP_SIFT_CONV



class FrankaWorldDynamicRatioTurnBased():

    def __init__(self, boxes: int, locs: int, ratio: int, init: tuple, goal: tuple, restricted_human_locs: List[int], enable_reordering: bool = False):
        self.boxes: int = boxes
        self.locs: int = locs
        self.ratio: int = ratio
        self.human_locs: List[int] = restricted_human_locs
        self.restricted_human_locs: Set[int] = set([0, self.locs] + [*range(1, self.locs + 1)]) - set(self.human_locs) 
        self.misc_preds = ['ready', 'in-transit', 'in-transfer' 'to-obj', 'holding']
        self.robot_actions: List[str] = ['transit', 'transfer', 'grasp', 'release']
        self.init = init
        self.goal = goal
        self.manager: Cudd = Cudd()
        # Predicate to Str maps - needed for lookup of the states corresponding to cubesstring
        self.pVar_map = bidict({})
        self.kVar_map = bidict({})
        self.xVar_map = dict()
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}
        self.rAction_map = bidict({})
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
        
        self.oVars: List[ADD] = self.create_output_vars()
        self.create_rAction_map()
        self.rAction_map_sym = bidict({k: self.cube_to_add(v, self.oVars) for k, v in self.rAction_map.items()})

        # now that the maps are initialized we create init and goal states
        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()

        # monolithic transition relation for the robot actions
        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}

        # create env move related vars and maps
        self.human_action: List[str] = ['hmove']
        self.iVars: List[ADD] = self.create_input_vars()
        self.eAction_map = bidict({})
        self.create_eAction_map()
        self.eAction_map_sym = bidict({k: self.cube_to_add(v, self.iVars) for k, v in self.eAction_map.items()})
        self.iVars_cube: ADD = reduce(lambda x, y: x & y, self.iVars)

        self.weight_dict: Dict[str, int] = {'transit': 1, 'transfer': 1, 'grasp': 1, 'release': 1}
        self.symbolic_weight_dict: Dict[str, ADD] = defaultdict(lambda: self.manager.addOne())
        self.create_sym_weight_dict()

        # precompute cubes of valid Robot and Env actions - needed for synthesis
        self.robot_action_cube_list: List[ADD] = [self.cube_to_add(r, self.oVars) for r in self.rAction_map.values()]
        self.env_action_cube_list: List[ADD] = [self.cube_to_add(e, self.iVars) for e in self.eAction_map.values()]

        self.miscellanoues_helper_stuff()

        if enable_reordering:
            self.enable_variable_reordering()            
    

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
        # num. of preds = ready x |locs| + to-obj x |boxes| + holding x |locs| + 1 (to account for l0 being end effector loc) + grasp + release
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
        # varsize = self.manager.size()
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
        rVars_prime: List[ADD] = [self.manager.addVar(k + varsize, 'pk' + str(k)) for k in range(len(self.kVars))]
        return rVars_prime
    
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
        # self.kVar_map[f'k0'] = f"{offset:0{len(self.kVars)}b}"
        # offset += 1
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
        # kVars = self.kVars if not prime else self.prime_kVars
        if prime:
            self.prime_kVar_map_sym = bidict({r: self.cube_to_add(v, self.prime_kVars) for r, v in self.kVar_map.items()})
        else:
            self.kVar_map_sym = bidict({r: self.cube_to_add(v, self.kVars) for r, v in self.kVar_map.items()})
        self.create_symbolic_maps(prime=prime)
    

    def enable_variable_reordering(self):
        """
          This method enable variable reordering - as VectorCompose and Restrict/Cofactor operate over mutually exclusive set
          of variables, we create groups for the variables to avoid reordering across groups. We use CUDD's Tree Nodes for this.
           
           NOTE: Enabling reordering seems to slow down the synthesis algorithm. So, we disable it by default. 
          This also a hint that our variable ordering is not too bad after all.
        """
        # self.manager.reduceHeap()
        self.manager.enableReorderingReporting()
        self.manager.makeTreeNode(0, len(self.latches))
        self.manager.makeTreeNode(len(self.latches), len(self.prime_latches))
        self.manager.makeTreeNode(2*len(self.latches), len(self.oVars))
        self.manager.makeTreeNode(2*len(self.latches) + len(self.oVars), len(self.iVars))
        self.manager.reduceHeap(REORDER_GROUP_SIFT_CONV)
        print('order:', ' '.join(self.manager.bddOrder()))
    

    def miscellanoues_helper_stuff(self):
        """
         Some miscellanoues helper stuff that are used in multiple places.
        """
        # need these cubes for printing states from cubes
        self.bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.bVars]
        self.prime_bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.prime_bVars]
        # create relevant env and robot actions; boxes
        self.monolithic_hnoop = reduce(lambda x, y: x | y, [act for act_str, act in self.eAction_map_sym.items() if act_str.startswith('hmove noop')])
        self.relevant_env_actions: ADD = reduce(lambda x, y: x | y, self.eAction_map_sym.values())
        self.relevant_robot_actions: ADD = reduce(lambda x, y: x | y, self.rAction_map_sym.values())
        self.relevant_env_actions_per_box = defaultdict(lambda: self.manager.addZero())
        self.relevant_box_preds_sym = defaultdict(lambda: self.manager.addZero())
        self.create_relevant_env_actions_per_box()
        self.create_relevant_box_predicates()
        self.monolithic_relevant_box_preds: ADD = reduce(lambda x, y: x & y, self.relevant_box_preds_sym.values())
        self.monolithic_valid_state_robot_actions: ADD = self.manager.addZero()
        self.monolithic_valid_state_robot_actions_prime_state: ADD = self.manager.addZero()
        
        # state invariance constraint - end-effector empty cube - used in transit and grasp actions
        self.ee_empty_cube: ADD = self.create_ee_empty_cube()
        self.create_monoltithic_box_conf_cube()
        self.create_hmove_not_b()
        self.create_valid_state_constraints()
        self.preprocess_monolithic_valid_state_robot_actions()
        self.preprocess_monolithic_valid_state_robot_actions_prime_state()
        
        self.locs_empty_constraints = defaultdict(lambda: self.manager.addZero())
        self.create_loc_empty_constraint()
        self.kVal_cube = reduce(lambda x, y: x | y, self.kVar_map_sym.values())


    def create_output_vars(self) -> List[ADD]:
        """
         Num. of robot actions = transit x |boxes| + transfer x |locs| + grasp + release
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes + self.locs + 2 + 1 # +1 to offset the 0-vector
        oVars_size = math.ceil(math.log2(num_of_rActions))
        oVars: List[ADD] =  [self.manager.addVar(r + varsize , 'o' + str(r)) for r in range(oVars_size)]
        return oVars
    
    def create_input_vars(self) -> List[ADD]:
        """
         Num. of human actions = |boxes| x |locs| + 1 (for no-op action)
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes * len(self.human_locs) + 1
        iVars_size = math.ceil(math.log2(num_of_rActions)) if num_of_rActions > 1 else 1
        iVars: List[ADD] =  [self.manager.addVar(h + varsize , 'i' + str(h)) for h in range(iVars_size)]
        return iVars
    
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

    def create_rAction_map(self) -> None:
        for ract in self.robot_actions:
            if ract == 'transit':
                for b in range(self.boxes):
                    act_str = f'{ract} b{b}'
                    rbit_str = f"{b + 1:0{len(self.oVars)}b}"
                    self.rAction_map[act_str] = rbit_str
            elif ract == 'transfer':
                for l in range(1, self.locs + 1):
                    act_str = f'{ract} l{l}'
                    rbit_str = f"{self.boxes + l:0{len(self.oVars)}b}"
                    self.rAction_map[act_str] = rbit_str
            # else:
        rbit_str = f"{self.boxes + self.locs + 1:0{len(self.oVars)}b}"
        self.rAction_map['grasp'] = rbit_str
        rbit_str = f"{self.boxes + self.locs + 2:0{len(self.oVars)}b}"
        self.rAction_map['release'] = rbit_str

    
    def create_eAction_map(self) -> None:
        # Add a no-op action for the human
        self.eAction_map[f'{self.human_action[0]} noop'] = f"{0:0{len(self.iVars)}b}"
        offset = 1
        for b in range(self.boxes):
            for l in self.human_locs:
                act_str = f'{self.human_action[0]} b{b} l{l}'
                hbit_str = f"{offset:0{len(self.iVars)}b}"
                self.eAction_map[act_str] = hbit_str
                offset += 1
        
        # the rest of them map to human noop as well.
        for i in range(offset, pow(2, len(self.iVars))):
            # hbit_str = f"{i:0{len(self.iVars)}b}"
            hbit_str = f"{offset:0{len(self.iVars)}b}"
            self.eAction_map[f'{self.human_action[0]} noop {i}'] = hbit_str
            offset += 1

    def create_sym_weight_dict(self) -> None:
        for ract, dd in self.rAction_map_sym.items():
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
    

    def set_prime_latches(self):
        self.prime_xVars: List[ADD] = self.prime_kVars + self.prime_pVars + [var for box_adds in self.prime_bVars for var in box_adds]
        self.prime_latches: List[ADD] = self.prime_tVar + self.prime_xVars #self.prime_kVars + self.prime_pVars + [var for box_adds in self.prime_bVars for var in box_adds] #+ self.prime_bVars


    def set_init_latch(self) -> ADD:
        init_cube = self.tVar_map_sym['robot'] & self.kVar_map_sym['k0']
        for s in self.init:
            init_cube &= self.xVar_map_sym[s]
        return init_cube
    

    def set_goal_latch(self) -> ADD:
        mono_goal_cube = self.manager.addZero()
        for state in self.goal:
            # goal_cube = self.tVar_map_sym['robot']
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
    

    def create_relevant_env_actions_per_box(self):
        """
         A tiny method to create relevant env actions for the human moves for each box.
        """
        for b in range(self.boxes):
            for act_str, act_add in self.eAction_map_sym.items():
                if f'b{b}' in act_str:
                    self.relevant_env_actions_per_box[b] |= act_add


    def create_relevant_box_predicates(self):
        """
         A tiny method to create relevant box predicates for each box.
        """
        # for b in range(self.boxes):
        for box_str, box_add in self.bVar_map_sym.items():
            box_id = int(box_str.split(' ')[0][-1])
            assert isinstance(box_id, int) and box_id in range(self.boxes), "Error in extracting box id. Fix this!!!"
            # if f'b{b}' in act_str:
            self.relevant_box_preds_sym[box_id] |= box_add
    

    def create_hmove_not_b(self):
        """
         A method that create a cube that consists of all hmvoves that are moving any box except the box b.
        """
        self.hmove_not_b = defaultdict(lambda: self.manager.addZero())
        for b in range(self.boxes):
            for act_str, act_add in self.eAction_map_sym.items():
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
                self.monolithic_valid_state_robot_actions |= self.xVar_map_sym[f'in-transit l{from_loc} b{b}'].ite(self.manager.addOne(), self.manager.addZero())
        
        for from_loc in range(1, self.locs + 1):
            for to_loc in range(1, self.locs + 1):
                if from_loc == to_loc:
                    continue
                self.monolithic_valid_state_robot_actions |= self.xVar_map_sym[f'in-transfer l{from_loc} l{to_loc}'].ite(self.manager.addOne(), self.manager.addZero())
    
    def preprocess_monolithic_valid_state_robot_actions_prime_state(self):
        """
         A helper function that preprocesses the monolithic_valid_state_robot_actions_prime_state variable. 
         This vairables catptues tuple (s, a_s, s') where s is a valid robot state, a_s is a valid robot action from state s and
         s' is the next state after applying action a_s from state s. 
         
         s' must be a valid human state.
        """
        # here we ass the constraint that kVar reamins constant after applying robot action
        for kVal in self.kVar_map.keys():
            self.monolithic_valid_state_robot_actions_prime_state |= self.kVar_map_sym[kVal].ite(self.prime_kVar_map_sym[kVal], self.manager.addZero())


    def post_process_transition_relation(self):
        """
         A method to post-process the transition relation after all action rules and frame axioms have been added.
        """
        for tr_key, tr_dd in self.transition_relation.items():
            self.transition_relation[tr_key] = tr_dd & self.monolithic_relevant_box_preds & self.monolithic_valid_state_robot_actions


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
          From any robot state, the turn var does no change it value and it's valus is only updated ftaer the human turn. Thus, k <-> k' where k' = k holds
          true under all robot actions and all robot states.
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

        # add robot frame axioms
        self.add_robot_frame_axioms()

        # keep only the relvant states and actions
        self.monolithic_valid_state_robot_actions_prime_state &= self.tVar_map_sym['robot'].ite(self.prime_tVar_map_sym['human'], self.manager.addZero()) & \
              self.monolithic_relevant_box_preds & self.monolithic_valid_state_robot_actions

        # finally, we add frame axioms for all boxes that enforce state invariance constraint
        self.create_human_move_transit()
        self.create_human_move_transfer()
        self.create_human_move_action_grasp()
        self.create_human_move_actions()

        self.add_human_frame_axiom()

        self.add_turn_var_update_rule()
        self.add_hmove_var_update_rule_from_robot_states()
        
        # keep only the valid robot states and actions in the transition relation
        self.post_process_transition_relation()
        
        # print s a_s s' transition function that we created for sanity checking
        self.convert_full_cube_to_state_ADD(self.monolithic_valid_state_robot_actions_prime_state, robot_action=True)
        

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
        robot_act_cube = self.rAction_map_sym['grasp']
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
                self.monolithic_valid_state_robot_actions |= ((rConf_cube | rConf_cube_ready) & self.xVar_map_sym[curr_box_pred]).ite(robot_act_cube, self.manager.addZero())
                
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
        robot_act_cube = self.rAction_map_sym['release']
        for loc in range(1, self.locs + 1):
            rConf_cube = self.xVar_map_sym[f'holding l{loc}']

            # for a given location, it can be any location, so we iterate over all locations
            for b in range(self.boxes):
                curr_box_pred = f"b{b} l0"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l0', bConf_cube=bConf_cube) & self.monolithic_relevant_box_preds

                robot_transition_cube = turn_bit & self.kVal_cube & rConf_cube & bConf_cube & robot_act_cube & self.locs_empty_constraints[f'l{loc}']

                # update the valid robot moves
                self.monolithic_valid_state_robot_actions |= (rConf_cube & self.xVar_map_sym[curr_box_pred]).ite(robot_act_cube, self.manager.addZero())

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
            robot_act_cube = self.rAction_map_sym[f"transit b{b}"]

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
                    self.monolithic_valid_state_robot_actions |= (rConf_cube & self.xVar_map_sym[curr_box_pred]).ite(robot_act_cube, self.manager.addZero())

                    pred_clause_prime_string = self.xVar_map[f"in-transit l{from_loc} b{b}"]
                    
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    # create s a_s s' transitions
                    prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym[f"in-transit l{from_loc} b{b}"]
                    self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())


    def create_transfer_actions(self) -> None:
        """
        Here we create transfer actions for the robot. The precondition for the robot is box b is at location l0 (end effector location):
         Preconditions:
            1. The robot is at location l and is holding box b - (holding l) (b l0) predicates are true at current state
         Effects+:
            2. The robot is at location l': (holding l') (b l0) predicates are true at next state
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
                    robot_act_cube = self.rAction_map_sym[f'transfer l{to_loc}']

                    robot_transition_cube = turn_bit & self.kVal_cube & rConf_cube & robot_act_cube & bConf_cube

                    # update the valid robot moves
                    self.monolithic_valid_state_robot_actions |= (rConf_cube & self.xVar_map_sym[curr_box_pred]).ite(robot_act_cube, self.manager.addZero())

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

                    for human_box in range(self.boxes):
                        for human_to_loc in self.human_locs:
                            ##### VALID MOVE CASE #####
                            hmove_cube = turn_bit & kVal_cube & \
                            self.eAction_map_sym[f'hmove b{human_box} l{human_to_loc}'] & rConf_cube
                            
                            constraint_cube = self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube &= ~self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            hmove_cube &= constraint_cube & self.monolithic_relevant_box_preds
                            
                            ##### INVALID MOVE CASE #####
                            constraint_cube = ~self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube |= self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            invalid_hmove_cube |=  turn_bit & kVal_cube & rConf_cube & \
                                self.eAction_map_sym[f'hmove b{human_box} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds

                            if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                                pred_clause_prime_string = self.xVar_map[f'holding l{from_loc}']
                                for sidx, s in enumerate(pred_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                                
                                # update k var - increment k by 1
                                kVal_prime_str = self.kVar_map[f'k{k + 1}']
                                for sidx, s in enumerate(kVal_prime_str):
                                    if s == '1':
                                        self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |=  hmove_cube
                            else:
                                invalid_hmove_cube |= hmove_cube

                            
                    # add the human noop action here - hmove noop is always a valid human move
                    pred_clause_prime_string = self.xVar_map[f'holding l{to_loc}']
                    hmove_cube = turn_bit & kVal_cube & (self.monolithic_hnoop | invalid_hmove_cube) & rConf_cube
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    for sidx, s in enumerate(self.kVar_map['k0']):
                        if s == '1':
                            self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube

    

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

                    for human_box in range(self.boxes):
                        for human_to_loc in self.human_locs:
                            ##### VALID MOVE CASE #####
                            hmove_cube = turn_bit & kVal_cube & \
                                self.eAction_map_sym[f'hmove b{human_box} l{human_to_loc}'] & rConf_cube & self.ee_empty_cube
                            
                            constraint_cube = self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube &= ~self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            hmove_cube &= constraint_cube & self.monolithic_relevant_box_preds

                            ##### INVALID MOVE CASE (not because of reaching the max human intervention) #####
                            constraint_cube = ~self.locs_empty_constraints[f'l{human_to_loc}']
                            for restricted_loc in self.restricted_human_locs:
                                constraint_cube |= self.xVar_map_sym[f'b{human_box} l{restricted_loc}']
                            invalid_hmove_cube |=  turn_bit & kVal_cube & \
                                self.eAction_map_sym[f'hmove b{human_box} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds

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
                                        self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |=  hmove_cube
                            
                            # the human has reached the max number of interventions in this robot turn;
                            # all they can do is noop so we add all hmove to invalid move case
                            else:
                                invalid_hmove_cube |= hmove_cube
                        
                    # add the human noop action here - hmove noop is always a valid human move
                    hmove_cube = turn_bit & kVal_cube & (self.monolithic_hnoop | invalid_hmove_cube) & rConf_cube & self.ee_empty_cube
                    pred_clause_prime_string = self.xVar_map[f'to-obj b{b}']
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                    
                    for sidx, s in enumerate(self.kVar_map['k0']):
                        if s == '1':
                            self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube


    def create_human_move_actions(self) -> None:
        """
        Here we create human move actions for the environment. The precondition for the human to move box b to location l is as follows:
         Preconditions:
            1. The human can move box b to location l only if location l is empty - (b @ l') & ~(b @ l) predicates are true at current state 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for hb in range(self.boxes):
                # boxes can be "grounded" at any location (except for 0 and else := |locs| + 1 location)
                for from_loc in range(1, self.locs + 1):
                    invalid_hmove_cube = self.manager.addZero()
                    box_pred = f"b{hb} l{from_loc}"
                    bConf_cube = self.xVar_map_sym[box_pred]
                    for human_to_loc in self.human_locs:
                        ##### VALID MOVE CASE #####
                        hmove_cube = turn_bit & kVal_cube & \
                            self.eAction_map_sym[f'hmove b{hb} l{human_to_loc}']
                        
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
                            self.eAction_map_sym[f'hmove b{hb} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds
                        

                        if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                            # The moved box `hb` goes to its new location - TODO: do i need to add bconf_cube here?
                            moved_box_prime_str = self.xVar_map[f'b{hb} l{human_to_loc}'] 
                            for sidx, s in enumerate(moved_box_prime_str):
                                if s == '1':
                                    self.transition_relation[self.bVars[hb][sidx].bddPattern().__str__()] |= hmove_cube
                            
                            # update k var - increment k by 1
                            kVal_prime_str = self.kVar_map[f'k{k + 1}']
                            for sidx, s in enumerate(kVal_prime_str):
                                if s == '1':
                                    self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |=  hmove_cube
                        
                        # the human has reached the max number of interventions in this robot turn;
                        # all they can do is noop so we add all hmove to invalid move case
                        else:
                            invalid_hmove_cube |= hmove_cube
                    
                    
                    hmove_cube = turn_bit & kVal_cube & (self.monolithic_hnoop | invalid_hmove_cube) & bConf_cube
                    for sidx, s in enumerate(self.xVar_map[box_pred]):
                        if s == '1':
                            self.transition_relation[self.bVars[hb][sidx].bddPattern().__str__()] |= hmove_cube
                    
                    for sidx, s in enumerate(self.kVar_map['k0']):
                        if s == '1':
                            self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
    
    def create_human_move_action_grasp(self) -> None:
        """
        Here we create human move actions for the environment during robot grasp action.
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for hb in range(self.boxes):
                # boxes can be "grounded" at any location (except for 0 and else := |locs| + 1 location)
                for from_loc in range(1, self.locs + 1):
                    invalid_hmove_cube = self.manager.addZero()
                    rConf_cubes_list = [self.xVar_map_sym[f'holding l{from_loc}'], self.xVar_map_sym[f'ready l{from_loc}'] & self.ee_empty_cube]
                    pred_clause_prime_string_list = [self.xVar_map[f'holding l{from_loc}'], self.xVar_map[f'ready l{from_loc}']]
                    for rConf_cube, pred_clause_prime_string in zip(rConf_cubes_list, pred_clause_prime_string_list):
                        for human_to_loc in self.human_locs:
                            ##### VALID MOVE CASE #####
                            hmove_cube = turn_bit & kVal_cube & \
                                self.eAction_map_sym[f'hmove b{hb} l{human_to_loc}'] & rConf_cube
                            
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
                                self.eAction_map_sym[f'hmove b{hb} l{human_to_loc}'] & constraint_cube & self.monolithic_relevant_box_preds
                            
                            if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                                for sidx, s in enumerate(pred_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube 
                                
                                # update k var - increment k by 1
                                kVal_prime_str = self.kVar_map[f'k{k + 1}']
                                for sidx, s in enumerate(kVal_prime_str):
                                    if s == '1':
                                        self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
                            
                            # the human has reached the max number of interventions in this robot turn;
                            # all they can do is noop so we add all hmove to invalid move case
                            else:
                                invalid_hmove_cube |= hmove_cube
                        
                        
                        hmove_cube = turn_bit & kVal_cube & (self.monolithic_hnoop | invalid_hmove_cube) & rConf_cube
                        # pred_clause_prime_string = self.xVar_map[f'holding l{from_loc}']
                        for sidx, s in enumerate(pred_clause_prime_string):
                            if s == '1':
                                self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube 
                        
                        for sidx, s in enumerate(self.kVar_map['k0']):
                            if s == '1':
                                self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= hmove_cube
        


    def add_robot_frame_axioms(self):
        """
        A helper function to test the frame axioms. Frame axioms ensure that the boxes that
          are not being moved by the robot do not change their location. This is like a state invariance constraint.
        
        Here, we add frame axioms for all boxes that are currently "grounded" (i.e., not being moved by the robot). 
          For thw box that is at end-effector (ee) location (l0), we add the state invariance constraint during the action construction.
        """
        grasp_action_cube = self.rAction_map_sym['grasp']
        release_action_cube = self.rAction_map_sym['release']
        for b in range(self.boxes):
            # not_grasp_cube = ~(self.xVar_map_sym[f'to-obj b{b}'] & grasp_action_cube)
            for l in range(1, self.locs + 1):
                not_grasp_cube = ~((self.xVar_map_sym[f'to-obj b{b}'] | self.xVar_map_sym[f'ready l{l}']) & grasp_action_cube)
                box_pred = f"b{b} l{l}"
                constraint_cube = self.manager.addOne()
                not_release_cube = ~(self.xVar_map_sym[f'holding l{l}'] & release_action_cube)
                constraint_cube &= not_grasp_cube & not_release_cube \
                    & self.relevant_robot_actions & self.monolithic_relevant_box_preds 
                
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= self.tVar_map_sym['robot'] & self.kVal_cube \
                            & constraint_cube & self.xVar_map_sym[box_pred]
                
                # create s a_s s' transitions
                prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym[box_pred]
                self.monolithic_valid_state_robot_actions_prime_state |= self.tVar_map_sym['robot'] & self.kVal_cube \
                            & constraint_cube & self.xVar_map_sym[box_pred].ite(prime_state_cube, self.manager.addZero())
                

    def add_human_frame_axiom(self):
        """
        This helper method add frame axioms for the human move actions. 
         Boxes that the human does not move should remain in the same location in the next state.
        """
        for b in range(self.boxes):
            for l in range(0, self.locs + 1):
                box_pred = f"b{b} l{l}"
                if l in self.restricted_human_locs:
                    haction_cube = self.tVar_map_sym['human'] & self.xVar_map_sym[box_pred] & self.kVal_cube
                else:
                    haction_cube = self.tVar_map_sym['human'] \
                        & self.xVar_map_sym[box_pred] & (self.monolithic_hnoop | self.hmove_not_b[b]) & self.kVal_cube
                    constraint_cube = self.manager.addOne()
                    for restricted_loc in self.restricted_human_locs:
                        constraint_cube &= ~self.xVar_map_sym[f'b{b} l{restricted_loc}']
                    haction_cube &= constraint_cube & self.monolithic_relevant_box_preds

                # box remmains in the same location if human does not move it
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= haction_cube

    
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

    
    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, robot_action: bool = False,  human_action: bool = False) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] #+ self.tVar + self.kVars
        if state_flag:
            relevant_vars.extend(self.latches)
        if robot_action:
            relevant_vars.extend(self.oVars)
        if human_action:
            relevant_vars.extend(self.iVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])
        start_ivar_idx, end_ivar_idx = self.manager.addVariables().index(self.iVars[0]), self.manager.addVariables().index(self.iVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.oVars + self.iVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.oVars + self.iVars)
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.oVars + self.iVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.oVars + self.iVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.oVars + self.iVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        for cube, val in cubes:
            tConf_exist_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_exist_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                print(f"[({self.tVar_map.inv[tConf_exist_str]}, {self.kVar_map.inv[kConf_exist_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states}), {val}]")
                states_action_pairs.append([((self.tVar_map.inv[tConf_exist_str], self.kVar_map.inv[kConf_exist_str], self.pVar_map.inv[rConf_cube_str], box_states), val), None])
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


    def convert_full_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, robot_action: bool = False,  human_action: bool = False) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.

         Here the input dd is assumed to be a fully defined cube (latches as well prime latches).
        """
        relevant_vars = [] #+ self.tVar + self.kVars + self.prime_tVar + self.prime_kVars
        if state_flag:
            relevant_vars.extend(self.latches)
            relevant_vars.extend(self.prime_latches)
        if robot_action:
            relevant_vars.extend(self.oVars)
        # if human_action:
        #     relevant_vars.extend(self.iVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])
        # start_ivar_idx, end_ivar_idx = self.manager.addVariables().index(self.iVars[0]), self.manager.addVariables().index(self.iVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.oVars + self.iVars + self.prime_latches)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.oVars + self.iVars + self.prime_latches)
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.oVars + self.iVars + self.prime_latches)


        # create prime turn abstraction cube - can I just swap them? Yes, I can.
        prime_tConf_exist_cube = reduce(lambda a, b: a & b, self.prime_xVars + self.oVars + self.iVars + self.latches)
        prime_kConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_xVars[len(self.prime_kVars):] + self.oVars + self.iVars + self.latches)

        test = tConf_exist_cube.swapVariables(self.latches, self.prime_latches)
        assert test == prime_tConf_exist_cube, "Error in swapping tConf_exist_cube variables to get prime_tConf_exist_cube"
        test = kConf_exist_cube.swapVariables(self.latches, self.prime_latches)
        assert test == prime_kConf_exist_cube, "Error in swapping kConf_exist_cube variables to get prime_kConf_exist_cube"
        
        # create existential abstraction cubes
        prime_rConf_exist_cube = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_xVars[len(self.prime_kVars)+len(self.prime_pVars):] + self.oVars + self.iVars + self.latches)

        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        prime_bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.oVars + self.iVars + self.prime_latches)
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.oVars + self.iVars + self.latches)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.oVars + self.iVars + self.prime_latches) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
                prime_bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.prime_tVar + self.prime_kVars + self.prime_pVars + self.oVars + self.iVars + self.latches) & reduce(lambda x, y: x & y, self.prime_bVars_cubes[:bidx] + self.prime_bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        prime_states_action_pairs = []
        
        for cube, val in cubes:
            state = None
            prime_state = None
            rAction = None
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
                state = ({self.tVar_map.inv[tConf_cube_str]}, {self.kVar_map.inv[kConf_cube_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states})
                # print(f"({self.tVar_map.inv[tConf_cube_str]}, {self.kVar_map.inv[kConf_cube_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states})")
                states_action_pairs.append([(self.tVar_map.inv[tConf_cube_str], self.kVar_map.inv[kConf_cube_str], self.pVar_map.inv[rConf_cube_str], box_states), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if robot_action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    rAction_str = self.rAction_map_sym.inv[self.cube_to_add(oCube_str, self.oVars)]
                except KeyError:
                    continue
            # if human_action:
            #     iCube_str = cube.bddPattern().cubeString()[start_ivar_idx:end_ivar_idx + 1].replace('-', '')
            #     try:
            #         eAction_str = self.eAction_map_sym.inv[self.cube_to_add(iCube_str, self.iVars)]
            #     except KeyError:
            #         continue
            if robot_action:    
                # action = ", ".join(filter(None, [rAction_str if robot_action else None, eAction_str if human_action else None]))
                rAction = rAction_str
                # print(f"    -- Action: ({rAction_str})")
            
            prime_bCube_str = []
            for e in prime_bConf_exist_cube.values():
                prime_bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))

            try:
                prime_box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(prime_bCube_str))
            except KeyError:
                continue
            
            try:
                # print(f"({self.tVar_map.inv[prime_tConf_cube_str]}, {self.kVar_map.inv[prime_kConf_cube_str]}, {self.pVar_map.inv[prime_rConf_cube_str]}, {prime_box_states})")
                prime_states_action_pairs.append([(self.tVar_map.inv[prime_tConf_cube_str], self.kVar_map.inv[prime_kConf_cube_str], self.pVar_map.inv[prime_rConf_cube_str], prime_box_states), None])
                prime_state = ({self.tVar_map.inv[prime_tConf_cube_str]}, {self.kVar_map.inv[prime_kConf_cube_str]}, {self.pVar_map.inv[prime_rConf_cube_str]}, {prime_box_states})
            except KeyError:
                continue

            # if you made it till here then print stuff
            print(state, f'--({rAction})-->', prime_state, sep="      ")
        
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
            # assert curr_state[rConf_idx].startswith('to-obj'), "Make sure the robot is in to-obj status when grasping!!!"
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
        oVars_bdd: List[BDD] = [var.bddPattern() for var in self.oVars]
        iVars_bdd: List[BDD] = [var.bddPattern() for var in self.iVars]

        while (curr_state & self.goal_latch.existAbstract(self.tVar[0])).isZero():
            if verbose:
                print("Current State:")
                curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state, state_flag=True, robot_action=False)
                assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
                "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval: int = list((curr_state & self.comp_winning_states).generate_cubes())[0][1]
            except IndexError:
                opt_sval: int = 0

            # get the action to be taken at the current state
            if curr_state_exp[0][0][0][0] == 'robot':
                act_cube: BDD = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(oVars_bdd)
            elif curr_state_exp[0][0][0][0] == 'human':
                act_cube: BDD = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(iVars_bdd)
            else:
                print("Unknown turn variable value. Cannot proceed with rollout!!")
                return
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.rAction_map.inv[act_cube_string] if curr_state_exp[0][0][0][0] == 'robot' \
                    else self.eAction_map.inv[act_cube_string]
            except KeyError:
                print("No robot action found!!")
                return
           
            # get the next state
            if curr_state_exp[0][0][0][0] == 'robot':
                curr_state: ADD = self.get_next_state_robot(list(curr_state_exp[0][0][0]), act_name)
            elif curr_state_exp[0][0][0][0] == 'human':
                curr_state, act_name = self.get_next_state_human(list(curr_state_exp[0][0][0]), act_name)
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Robot Action: {act_name}")
            


    def solve(self, verbose: bool = False, cooperative_game: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.
        """
        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states =  self.manager.plusInfinity()
        curr_winning_states = curr_winning_states.min(goal)

        # print the initial winning states
        if verbose:
            print("Initial Winning States:")
            # by default generate cubes does not retuen cubes that point to 0 leaf. 
            # So, we manually convert the 0 leaf to a cube with leaf value 1 here for printing.
            self.convert_cube_to_state_ADD(curr_winning_states.bddInterval(0, 0).toADD(), robot_action=False)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")

            # prime the vars
            curr_winning_states_primed = curr_winning_states.swapVariables(self.latches, self.prime_latches)
            preimage = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))

            # add the action costs associated with the robot actions   
            preimage = preimage + self.weight
            # print("Current Preimage:")
            # self.convert_cube_to_state_ADD(preimage, state_flag=True, robot_action=False, human_action=False)
            # go over all the env actions and preserve the maximum one
            MaxUpre = []
            for env_tr_dd in self.env_action_cube_list:
                MaxUpre.append(preimage.restrict(env_tr_dd))
            
            if cooperative_game:
                Upre = reduce(lambda x, y: x.min(y), MaxUpre)
            else:
                Upre = reduce(lambda x, y: x.max(y), MaxUpre)

            # go over all the sys actions and preserve the manimum one
            Minpre = []
            for robot_tr_dd in self.robot_action_cube_list:
                Minpre.append(Upre.restrict(robot_tr_dd))
            
            next_winning_states = reduce(lambda x, y: x.min(y), Minpre)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, robot_action=False)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if self.init_latch & curr_winning_states != self.manager.plusInfinity():
                    init_val: int = list((self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    return preimage if init_val < math.inf else None
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states

    
    def test_pre_image_restricted_human_moves(self):
        goal_cube = self.tVar_map_sym['human'] & self.kVar_map_sym['k0'] & self.xVar_map_sym['holding l1'] & self.xVar_map_sym['b0 l0'] #& self.xVar_map_sym['b1 l3']
        print('Goal state:', goal_cube)
        # compute preimage 
        From = goal_cube.swapVariables(self.latches, self.prime_latches)
        preimage = From.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        print('Preimage: ', preimage)
        self.convert_cube_to_state_ADD(preimage, human_action=False, robot_action=False)



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
    
    print('****************rAction Map:****************')
    for k, v in fw_tb.rAction_map.items():
        print(f"{k} : {v}")

    print('****************eAction Map:****************')
    for k, v in fw_tb.eAction_map.items():
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
    # fw_tb.test_pre_image()
    tic = time.time()
    strategy = fw_tb.solve(verbose=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        fw_tb.roll_out_strategy(strategy=strategy, verbose=True)