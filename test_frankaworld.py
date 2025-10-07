import sys
import time
import math

from functools import reduce
from itertools import product
from collections import defaultdict
from typing import List, Tuple, Optional, Dict

from bidict import bidict
from cudd import Cudd, ADD, BDD

class FrankaWorld():

    def __init__(self, boxes: int, locs: int, init: str, goal: str):
        self.boxes: int = boxes
        self.locs: int = locs
        self.misc_preds = ['ready', 'to-obj', 'holding']
        self.robot_actions: List[str] = ['transit', 'transfer', 'grasp', 'release']
        self.init = init
        self.goal = goal
        self.manager: Cudd = Cudd()

        self.pVars, self.bVars = self.create_latches()
        self.xVars: List[ADD] = self.pVars + [var for box_adds in self.bVars for var in box_adds]
        self.prime_pVars, self.prime_bVars = self.create_prime_latches()
        self.oVars: List[ADD] = self.create_output_vars()
        
        self.latches: List[ADD] = self.xVars # in future we will primed version of these as well.
        self.prime_latches: List[ADD] = self.prime_pVars + self.prime_bVars
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]

        self.xVar_map = dict()
        self.rAction_map = bidict({})
        self.xVar_map_sym  = dict()
        
        # maps needs for lookup of the states corresponding to cubes
        self.pVar_map = bidict({})
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}

        self.create_xVar_map()
        self.create_rAction_map()

        # more bookeeping stuff
        self.rAction_map_sym = bidict({k: self.cube_to_add(v, self.oVars) for k, v in self.rAction_map.items()})
        self.create_symbolic_maps()

        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()

        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}

        # need these cubes for printing states from cubes
        self.pVars_cube: ADD = reduce(lambda a, b: a & b, self.pVars)
        self.bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.bVars]
        self.all_bVars_cube: ADD = reduce(lambda a, b: a & b, self.bVars_cubes)
        self.oVars_cube: ADD = reduce(lambda x, y: x & y, self.oVars)

        # precompute cubes of oVars - needed for synthesis
        self.robot_action_cube_list: List[ADD] = [self.cube_to_add(r, self.oVars) for r in self.rAction_map.values()]

        # state invariance constraint - end-effector empty cube - used in transit and grasp actions
        self.ee_empty_cube: ADD = self.create_ee_empty_cube()
        

    
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

    def create_ready_holding_to_obj_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        # num. of preds = ready x |locs| + to-obj x |boxes| + holding x |locs| + 1 (to account for l0 being end effector loc) + grasp + release
        num_of_preds = 2*self.locs + self.boxes + 2 + 1 # +1 for ready-else state
        vars_size: int = math.ceil(math.log2(num_of_preds))
        Vars: List[ADD] = [self.manager.addVar(k + varsize, 'p' + str(k)) for k in range(vars_size)]
        return Vars

    
    def create_prime_latches(self) -> Tuple[List[ADD], List[ADD]]:
        """
         Create a copy of prime variables for the latches.
        """
        varsize = self.manager.size()
        pVars_prime: List[ADD] = [self.manager.addVar(k + varsize, 'pp' + str(k)) for k in range(len(self.pVars))]
        varsize = self.manager.size()
        bVars_prime: List[ADD] = [self.manager.addVar(k + varsize, 'py' + str(k)) for k in range((len(self.xVars) - len(self.pVars)))]

        return pVars_prime, bVars_prime
    

    def create_xVar_map(self):
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

        # for each boxes we create |locs| boolean vars
        for b in range(self.boxes):
            for l in range(self.locs + 1):
                bit_str = f"{l + 1:0{len(self.bVars[b])}b}"
                self.xVar_map['b' + str(b) + ' l' + str(l)] = bit_str
                self.bVars_map[b]['b' + str(b) + ' l' + str(l)] = bit_str

    def create_output_vars(self) -> List[ADD]:
        """
         Num. of robot actions = transit x |boxes| +  transfer x |locs| + grasp + release
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes + self.locs + 2
        oVars_size = math.ceil(math.log2(num_of_rActions))
        oVars: List[ADD] =  [self.manager.addVar(r + varsize , 'o' + str(r)) for r in range(oVars_size)]
        return oVars

    
    def create_symbolic_maps(self):
        """
         Small function to create symbolic maps for the xVar_map, rAction_map and eAction_map
        """
        for k, v in self.pVar_map.items():
            self.xVar_map_sym[k] = self.cube_to_add(v, self.pVars)
        
        for bidx, d in self.bVars_map.items():
            for k, v in d.items():
                self.xVar_map_sym[k] = self.cube_to_add(v, self.bVars[bidx])

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
    
    
    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add


    def set_init_latch(self) -> ADD:
        init_cube = self.manager.addOne()
        for s in self.init:
            init_cube &= self.xVar_map_sym[s]
        return init_cube
    

    def set_goal_latch(self) -> ADD:
        goal_cube = self.manager.addOne()
        for s in self.goal:
            goal_cube &= self.xVar_map_sym[s]
        return goal_cube
    

    def create_only_b_at_ee_cube(self, curr_box: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.xVar_map_sym['b' + str(ob) + ' l0']
        return bConf_cube
    

    def create_only_b_at_l_cube(self, curr_box: int, curr_loc: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.xVar_map_sym['b' + str(ob) + f' {curr_loc}']
        return bConf_cube
    

    def create_ee_empty_cube(self) -> ADD:
        # cube that implies that end-effector location empty
        bConf_cube = self.manager.addOne()
        for b in range(self.boxes):
            bConf_cube &= ~self.xVar_map_sym['b' + str(b) + ' l0']
        return bConf_cube


    def add_frame_axioms(self):
        """
        A helper function to test the frame axioms. Frame axioms ensure that the boxes that
          are not being moved by the robot do not change their location. This is like a state invariance constraint.
        
        Here, we add frame axioms for all boxes that are currently "grounded" (i.e., not being moved by the robot). 
          For thw box that is at end-effector (ee) location (l0), we add the state invariance constraint during the action construction.
        """
        grasp_action_cube = self.rAction_map_sym['grasp']
        release_action_cube = self.rAction_map_sym['release']
        for b in range(self.boxes):
            not_grasp_cube = ~(self.xVar_map_sym[f'to-obj b{b}'] & grasp_action_cube)
            for l in range(1, self.locs + 1):
                box_pred = f"b{b} l{l}"
                constraint_cube = self.manager.addOne()
                not_release_cube = ~(self.xVar_map_sym[f'holding l{l}'] & release_action_cube)
                constraint_cube &= not_grasp_cube & not_release_cube
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= constraint_cube & self.xVar_map_sym[box_pred]



    def create_transition_relation(self):
        """
         A method to construct the transition relation for the Franka World domain. 
         This constructs the transition system that only consists of actions controlled by the robot (robot actions). 
         For game construction, see the FrankaWorldGame class.
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
        self.add_frame_axioms()
    

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
        state_constraint_cube = self.ee_empty_cube
        robot_act_cube = self.rAction_map_sym['grasp']
        for b in range(self.boxes):
            rConf_cube = self.xVar_map_sym[f'to-obj b{b}']

            # for a given box, it can be at any location, so we iterate over all locations
            for loc in range(1, self.locs + 1):
                curr_box_pred = f"b{b} l{loc}"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(loc), bConf_cube=bConf_cube)

                robot_transition_cube = rConf_cube & bConf_cube & state_constraint_cube & robot_act_cube

                # this is fixed and is independent of the human action
                pred_clause_prime_string = self.xVar_map['holding l' + str(loc)]
                box_clause_prime_string = self.xVar_map[f'b{b} l0']
                
                # now we add the transition where the human does all the valid move and the robot grasps the box
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube


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
        robot_act_cube = self.rAction_map_sym['release']
        for loc in range(1, self.locs + 1):
            rConf_cube = self.xVar_map_sym[f'holding l{loc}']

            # for a given location, it can any location, so we iterate over all locations
            for b in range(self.boxes):
                curr_box_pred = f"b{b} l0"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                bConf_cube = self.create_only_b_at_ee_cube(b, bConf_cube)

                robot_transition_cube = rConf_cube & bConf_cube & robot_act_cube

                # this is fixed and is independent of the human action
                next_box_pred = f"b{b} l{loc}"
                pred_clause_prime_string = self.xVar_map['ready l' + str(loc)]
                box_clause_prime_string = self.xVar_map[next_box_pred]
                
                # now we add the transition where the human does all the valid move and the robot grasps the box
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube


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
        for b in range(self.boxes):
            robot_act_cube = self.rAction_map_sym[f"transit b{b}"]

            for from_loc in range(1, self.locs + 2):
                rConf_cube = self.xVar_map_sym[f'ready l{from_loc}']
                
                for to_loc in range(1, self.locs + 1):
                    # we skip transiting to the same location
                    if from_loc == to_loc:
                        continue
                    curr_box_pred: str = f"b{b} l{to_loc}"
                    bConf_cube = self.xVar_map_sym[curr_box_pred]
                    
                    # need to enforce that only one box is at loc l
                    bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(to_loc), bConf_cube=bConf_cube)

                    robot_transition_cube = rConf_cube & bConf_cube & state_constraint_cube & robot_act_cube

                    box_clause_prime_string = self.xVar_map[curr_box_pred]
                    pred_clause_prime_string = self.xVar_map[f"to-obj b{b}"]
                    
                    # now we add the transition where the human does all the valid move and the robot grasps the box
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube

    
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
        for b in range(self.boxes):
            curr_box_pred = f'b{b} l0'
            for from_loc in range(1, self.locs + 1):
                rConf = f'holding l{from_loc}'
                rConf_cube = self.xVar_map_sym[rConf]
                for to_loc in range(1, self.locs + 1):
                    # skip transferring to the same location
                    if from_loc == to_loc:
                        continue
                    bConf_cube = self.xVar_map_sym[curr_box_pred]
                    bConf_cube = self.create_only_b_at_ee_cube(b, bConf_cube)
                    robot_act_cube = self.rAction_map_sym[f'transfer l{to_loc}']

                    robot_transition_cube = rConf_cube & bConf_cube & robot_act_cube

                    # next state clauses - (holding to_loc) ; box location does not change
                    pred_clause_prime_string = self.xVar_map[f'holding l{to_loc}']
                    box_clause_prime_string = self.xVar_map[curr_box_pred] 
                    
                    # now we add the transition where the human does all the valid move and the robot grasps the box
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube

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

    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, robot_action: bool = False) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches)
        if robot_action:
            relevant_vars.extend(self.oVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.xVars[len(self.pVars):] + self.oVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.pVars + self.oVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.pVars + self.oVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        for cube, val in cubes:
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                print(f"[({self.pVar_map.inv[rConf_cube_str]}, {box_states}), {val}]")
                states_action_pairs.append([((self.pVar_map.inv[rConf_cube_str], box_states), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if robot_action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    rAction_str = self.rAction_map_sym.inv[self.cube_to_add(oCube_str, self.oVars)]
                except KeyError:
                    continue
            if robot_action :    
                # action = ", ".join(filter(None, [rAction_str if robot_action else None]))
                print(f"    -- Actions: ({rAction_str})")
                states_action_pairs[-1][-1] = rAction_str
        
        return states_action_pairs
    
    
    def preimage(self, ts_action: List[BDD], From: BDD) -> BDD:
        return From.vectorCompose(self.latches_bdd, ts_action)
    

    def get_buckets_of_BDD(self, max_interval_val: int, winning_states: ADD) -> Dict[int, BDD]:
        _win_state_bucket: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval in range(max_interval_val + 1):
            # get the states with state value equal to sval and store them in their respective bukcets
            win_sval: BDD = winning_states.bddInterval(sval, sval)
            
            if not win_sval.isZero():
                _win_state_bucket[sval] |= win_sval
        
        return _win_state_bucket


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
    

    def get_next_state(self, curr_state: List[str], action: str) -> ADD:
        """
         A helper function to get the next state given the current state and action.
        """
        # if action is transit then, update the robot configuration
        if action.startswith('transit'):
            assert curr_state[0].startswith('ready'), "Make sure the robot is ready to transit!!!"
            b_idx = action.split(' ')[1]
            curr_state[0] = f'to-obj {b_idx}'
        
        # if action is grasp then, update the robot configuration and box configuration
        elif action.startswith('grasp'):
            assert curr_state[0].startswith('to-obj'), "Make sure the robot is in to-obj status when grasping!!!"
            box: str = curr_state[0].split(' ')[1]
            b_idx = int(box[-1])
            # the box str while will of th form b0 l1, b1 l3, etc..
            split_str = curr_state[1].split(', ')
            l_idx = split_str[b_idx].split(' ')[1] 
            split_str[b_idx] = f'{box} l0'
            curr_state[1] = ', '.join(split_str)
            # update the robot configuration
            curr_state[0] = f'holding {l_idx}'

        # if action is release then, update the robot configuration and box configuration
        elif action.startswith('release'):
            assert curr_state[0].startswith('holding'), "Make sure the robot is in holding status when releasing!!!"
            l_idx = curr_state[0].split(' ')[1]
            # change the box location from l0 to l_idx
            split_str = curr_state[1].split(', ')
            for bidx, b in enumerate(split_str):
                if b.endswith('l0'):
                    box = b.split(' ')[0]
                    split_str[bidx] = f'{box} {l_idx}'
                    break
            # update the robot configuration
            curr_state[1] = ', '.join(split_str)
            curr_state[0] = f'ready {l_idx}'

        
        # if action is transfer then, update the robot configuration 
        elif action.startswith('transfer'):
            assert curr_state[0].startswith('holding'), "Make sure the robot is holding when transfering to another loc!!!"
            l_idx = action.split(' ')[1]
            curr_state[0] = f'holding {l_idx}'

        else:
            print("Unknown action. Cannot compute next state!!")
            sys.exit(-1)
        
        # conver the string of boxes location to sperate state
        split_str = curr_state[1].split(', ')
        return self.xVar_map_sym[curr_state[0]] & reduce(lambda a, b: a & b, [self.xVar_map_sym[s] for s in split_str])


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a give strategy
        """
        curr_state = self.init_latch
        oVars_bdd: List[BDD] = [var.bddPattern() for var in self.oVars]

        while (curr_state & self.goal_latch).isZero():
            if verbose:
                print("Current State:")
                curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state, state_flag=True, robot_action=False)
                assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
                "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            opt_sval = list((curr_state & self.comp_winning_states).generate_cubes())[0][1]

            # get the action to be taken at the current state
            act_cube: BDD = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(oVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            if verbose:
                try:
                    ract_name = self.rAction_map.inv[act_cube_string]
                    print(f"Robot Action: {ract_name}")
                except KeyError:
                    print("No robot action found!!")
                    return
           
            # get the next state
            curr_state: ADD = self.get_next_state(list(curr_state_exp[0][0][0]), ract_name)



    def solve(self, verbose: bool = False) -> Optional[ADD]:
        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states =  self.manager.plusInfinity()
        curr_winning_states = curr_winning_states.min(goal)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")

            # prime the vars
            curr_winning_states_primed = curr_winning_states.swapVariables(self.latches, self.prime_latches)
            preimage = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
            
            # add the action costs    
            # preimage = preimage + self.weight
            preimage = preimage + self.manager.addOne()

            # go over all the sys actions and preserve the manimum one
            Minpre = []
            for robot_tr_dd in self.robot_action_cube_list:
                Minpre.append(preimage.restrict(robot_tr_dd))
            
            next_winning_states = reduce(lambda x, y: x.min(y), Minpre)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if self.init_latch & curr_winning_states != self.manager.plusInfinity():
                    if (self.init_latch & curr_winning_states).isZero():
                        print("Init state is a goal state. The state value is 0 and robot can take any action.")
                        return None
                    
                    init_val: int = list((self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A path to goal state exists!!. The Init state value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    return preimage if init_val < math.inf else None
                return None


            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def test_pre_image(self):
        # convert transition relation to latches bdd

        tr_bdd = [dd.bddPattern() for dd in self.transition_relation.values()]

        # goal state is b0 and l0 and ready l0
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars) 
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['b1 l0'], self.bVars[1]) & self.cube_to_add(self.xVar_map['holding l2'], self.pVars) 
        # goal_cube = self.cube_to_add(self.xVar_map['hol1ding l1'], self.pVars)
        goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['b1 l2'], self.bVars[1]) & self.cube_to_add(self.xVar_map['to-obj b1'], self.pVars)
        # goal_cube = self.cube_to_add(self.xVar_map['b1 l2'], self.bVars[1]) & self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars)
        print('Goal state:', goal_cube.bddPattern())
        From = goal_cube.bddPattern()

        pre = From.vectorCompose(self.latches_bdd, tr_bdd)

        print('Preimage: ', pre)
        # self.convert_cube_to_state(pre)


if __name__ == "__main__":
    boxes = 3
    locs = 8
    # init = ['ready l4', 'b0 l2', 'b1 l3']
    # goal = ['b0 l2']
    # goal = ['b0 l1', 'b1 l2']
    # init = ['ready l3', 'b0 l1']
    # goal = ['b0 l2']
    # goal = ['ready l1', 'b0 l1']
    init = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l4']
    goal = ['b0 l1', 'b1 l2', 'b2 l3']
    fw = FrankaWorld(boxes=boxes, locs=locs, init=init, goal=goal)

    print('****************xVars map:****************')
    for k, v in fw.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************rAction map:****************')
    for k, v in fw.rAction_map.items():
        print(f"{k} : {v}")    
    
    print("Total num of latches: ", len(fw.latches))
    print("Total num of prime latches: ", len(fw.prime_latches))
    print("Total boolean vars: ", len(fw.latches) + len(fw.prime_latches))

    tic = time.time()
    fw.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # print('Transition Relation:')
    # for k, v in fw.transition_relation.items():
    #     print(f"{k} : {v}")
    
    synth_start = time.time() 
    strategy = fw.solve(verbose=False)
    synth_stop = time.time()
    print(f"Time to synthesize strategy: {synth_stop - synth_start} seconds")
    # testing things out
    # t = fw.get_all_states_interval(upper=4, dd=strategy, lower=4)
    print("Rolling out Strategy")
    fw.roll_out_strategy(strategy=strategy, verbose=True)
    
    print("Done")