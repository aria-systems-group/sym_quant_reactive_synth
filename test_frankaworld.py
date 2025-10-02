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

        # book keeping
        # self.holding_preds = self.to_obj_preds = self.ready_preds = set({})

        self.xVar_map = dict()
        self.rAction_map = dict()
        
        # maps needs for lookup of the states corresponding to cubes
        self.pVar_map = bidict({})
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}

        self.create_xVar_map()
        self.create_rAction_map()

        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()

        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}

        # need these cubes for printing states from cubes
        self.pVars_cube: ADD = reduce(lambda a, b: a & b, self.pVars)
        self.bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.bVars]
        self.all_bVars_cube: ADD = reduce(lambda a, b: a & b, self.bVars_cubes)

        # precompute cubes for iVars and oVars - needed for synthesis
        self.robot_action_cube_list: List[ADD] = [self.cube_to_add(r, self.oVars) for r in self.rAction_map.values()]

    
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
                # self.ready_preds.add('ready l' + str(loc)) if pred == 'ready' else self.holding_preds.add('holding l' + str(loc))
                self.pVar_map[pred + ' l' + str(loc)] = bit_str
            #
            offset += self.locs
        
        offset = 2*(self.locs) + 2 # 1 for the offset from the 0-vector; another 1 for the ready-else state
        # for misc pred to-obj we create all boxes
        for b in range(self.boxes):
            bit_str = f"{b + offset:0{len(self.pVars)}b}"
            self.xVar_map['to-obj b' + str(b)] = bit_str
            # self.to_obj_preds.add('to-obj b' + str(b))
            self.pVar_map['to-obj b' + str(b)] = bit_str

        # for each boxes we create |locs| boolean vars
        for b in range(self.boxes):
            for l in range(self.locs + 1):
                bit_str = f"{l + 1:0{len(self.bVars[b])}b}"
                self.xVar_map['b' + str(b) + ' l' + str(l)] = bit_str
                self.bVars_map[b]['b' + str(b) + ' l' + str(l)] = bit_str

    def create_output_vars(self) -> List[ADD]:
        """
         Num of robot actions = transit x |boxes| +  transfer x |locs| + grasp + release
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes + self.locs + 2
        oVars_size = math.ceil(math.log2(num_of_rActions))
        oVars: List[ADD] =  [self.manager.addVar(r + varsize , 'o' + str(r)) for r in range(oVars_size)]
        return oVars


    def create_rAction_map(self):
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
        for idx, s in enumerate(self.init):
            if idx == 0:
                init_cube &= self.cube_to_add(self.xVar_map[s], self.pVars)
            else:
                init_cube &= self.cube_to_add(self.xVar_map[s], self.bVars[idx - 1])
        return init_cube
    

    def set_goal_latch(self) -> ADD:
        goal_cube = self.manager.addOne()
        for idx, s in enumerate(self.goal):
            if idx == 0:
                goal_cube &= self.cube_to_add(self.xVar_map[s], self.pVars)
            else:
                goal_cube &= self.cube_to_add(self.xVar_map[s], self.bVars[idx - 1])
        return goal_cube

    def create_only_b_at_ee_cube(self, curr_box: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.cube_to_add(self.xVar_map['b' + str(ob) + ' l0'], self.bVars[ob])
        return bConf_cube
    

    def create_only_b_at_l_cube(self, curr_box: int, curr_loc: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.cube_to_add(self.xVar_map['b' + str(ob) + f' {curr_loc}'], self.bVars[ob])
        return bConf_cube
    

    def create_ee_empty_cube(self) -> ADD:
        # cube that implies that end-effector location empty
        bConf_cube = self.manager.addOne()
        for b in range(self.boxes):
            bConf_cube &= ~self.cube_to_add(self.xVar_map['b' + str(b) + ' l0'], self.bVars[b])
        return bConf_cube
    

        
    
    def create_transition_relation(self) -> None:
        
        # transit actions evolvin to to-obj preds 
        ee_empty_cube: ADD = self.create_ee_empty_cube()
        for b in range(self.boxes):
            act_str = f'transit b{b}'

            for from_loc in range(1, self.locs + 2):
                rConf_cube = self.cube_to_add(self.xVar_map[f'ready l{from_loc}'], self.pVars)
                
                # as l0 is reserved for end-effector location
                for to_loc in range(1, self.locs + 1):
                    box_pred: str = 'b' + str(b) + ' l' + str(to_loc)
                    bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b])
                    act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                    # need to enforce that the end-effector is empty
                    state_constraint_cube = ee_empty_cube

                    # need to enforce that only one box is at loc l
                    bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(to_loc), bConf_cube=bConf_cube)

                    # next state clauses - (to-obj b0); box location does not change
                    box_clause_prime_string = self.xVar_map[box_pred]
                    pred_clause_prime_string = self.xVar_map['to-obj b' + str(b)]
                    
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube
                    
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube
            
                    # Frame Axioms: Other boxes do not change their location
                    for other_b in range(self.boxes):
                        if other_b == b:
                            continue
                        for frame_loc in range(1, self.locs + 1):
                            if frame_loc == to_loc:
                                continue
                            frame_box_pred: str = 'b' + str(other_b) + ' l' + str(frame_loc)
                            for sidx, s in enumerate(self.xVar_map[frame_box_pred]):
                                if s == '1':
                                    self.transition_relation[self.bVars[other_b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube & self.cube_to_add(self.xVar_map[frame_box_pred], self.bVars[other_b])
        
        # grasp action
        for b in range(self.boxes):
            act_str = 'grasp'
            rConf_cube = self.cube_to_add(self.xVar_map[f'to-obj b{b}'], self.pVars)
            
            # as l0 is reserved for end-effector location
            for l in range(1, self.locs + 1):
                box_pred = 'b' + str(b) + ' l' + str(l)
                bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b])
                act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                # need to enforce that the end-effector is empty
                state_constraint_cube = ee_empty_cube
                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(l), bConf_cube=bConf_cube)

                # next state clauses - (holding l) ; box location does not change
                box_pred_prime = 'b' + str(b) + ' l0' # box is now at end-effector location
                pred_clause_prime_string = self.xVar_map['holding l' + str(l)]
                box_clause_prime_string = self.xVar_map[box_pred_prime]
                
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & state_constraint_cube & bConf_cube & act_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & state_constraint_cube & bConf_cube & act_cube
                
                # Frame Axioms: Other boxes do not change their location
                for other_b in range(self.boxes):
                    if other_b == b:
                        continue
                    for frame_loc in range(1, self.locs + 1):
                        # box will be at l0 but I am keeping this if statement for consistency
                        if frame_loc == l:
                            continue
                        frame_box_pred: str = 'b' + str(other_b) + ' l' + str(frame_loc)
                        for sidx, s in enumerate(self.xVar_map[frame_box_pred]):
                            if s == '1':
                                self.transition_relation[self.bVars[other_b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube & self.cube_to_add(self.xVar_map[frame_box_pred], self.bVars[other_b])
        
        # release action
        act_str = 'release'
        for rConf, rCube_str in self.xVar_map.items():
            if not rConf.startswith('holding'):
                continue
            to_loc = rConf.split(' ')[1]
            
            rConf_cube = self.cube_to_add(rCube_str, self.pVars)
            
            for b in range(self.boxes):
                box_pred = 'b' + str(b) + ' l0'
                bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b]) # box is at end-effector location
                act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                bConf_cube = self.create_only_b_at_ee_cube(b, bConf_cube)

                # next state clauses - (ready l) ; box location does not change
                box_pred_prime = 'b' + str(b) + f' {to_loc}' # box is now at location loc
                pred_clause_prime_string = self.xVar_map['ready ' + to_loc]
                box_clause_prime_string = self.xVar_map[box_pred_prime]
                # need to enforce that the end-effector is empty - we enforce it from transit action, so maybe we don;t need it here. Check this!!!
                
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
                
                # Frame Axioms: Other boxes do not change their location
                for other_b in range(self.boxes):
                    if other_b == b:
                        continue
                    for frame_loc in range(1, self.locs + 1):
                        if frame_loc == to_loc:
                            continue
                        frame_box_pred: str = 'b' + str(other_b) + ' l' + str(frame_loc)
                        for sidx, s in enumerate(self.xVar_map[frame_box_pred]):
                            if s == '1':
                                self.transition_relation[self.bVars[other_b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube & self.cube_to_add(self.xVar_map[frame_box_pred], self.bVars[other_b])

        # transfer actions
        for b in range(self.boxes):
            act_str = 'transfer'
            # for all holding confs.
            for rConf_from, rCube_from_str in self.xVar_map.items():
                if not rConf_from.startswith('holding'):
                    continue
                from_loc = rConf_from.split(' ')[1]

                for rConf_to, rCube_to_str in self.xVar_map.items():
                    if not rConf_to.startswith('holding') or rConf_to == rConf_from:
                        continue
                    to_loc = rConf_to.split(' ')[1]

                    act_str = f'transfer {to_loc}'
                    box_pred = f'b{b} l0'
                    rConf_cube = self.cube_to_add(rCube_from_str, self.pVars)
                    bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b])  # box at current location from_loc
                    act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                    bConf_cube = self.create_only_b_at_ee_cube(b, bConf_cube)

                    # next state clauses - (holding to_loc) ; box location does not change
                    assert rConf_to == 'holding ' + to_loc, "Make sure the to_loc is correct. Fix this!!!"
                    pred_clause_prime_string = self.xVar_map['holding ' + to_loc]
                    box_clause_prime_string = self.xVar_map[box_pred] 

                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
                
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
                    

                    # Frame Axioms: Other boxes do not change their location
                    for other_b in range(self.boxes):
                        if other_b == b:
                            continue
                        for frame_loc in range(1, self.locs + 1):
                            if frame_loc == to_loc:
                                continue
                            frame_box_pred: str = 'b' + str(other_b) + ' l' + str(frame_loc)
                            for sidx, s in enumerate(self.xVar_map[frame_box_pred]):
                                if s == '1':
                                    self.transition_relation[self.bVars[other_b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube & self.cube_to_add(self.xVar_map[frame_box_pred], self.bVars[other_b])
        
    

    def convert_cube_to_state_ADD(self, dd: ADD) -> None:
        """
         Convert a cube to a state representation
        """
        cubes = []
        for cube_list, val in dd.generate_cubes():
            if val == math.inf:
                continue
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube_list):
                if self.manager.addVar(_idx) not in self.latches:
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
        

        # print the states
        for cube, val in cubes:
            rConf_cube_str = cube.existAbstract(self.all_bVars_cube).bddPattern().cubeString().replace('-', '')
            # for multiple boxes
            if self.boxes > 1:
                bCube_str = []
                for b in range(self.boxes):
                    all_but_b_cube = reduce(lambda a, b: a & b, self.bVars_cubes[:b] + self.bVars_cubes[b+1:])
                    # all_but_b_cube = self.bVars_cubes[0]
                    bCube_str.append(cube.existAbstract(all_but_b_cube & self.pVars_cube).bddPattern().cubeString().replace('-', ''))
            else:
                # for single box
                bCube_str = [cube.existAbstract(self.pVars_cube).bddPattern().cubeString().replace('-', '')]
            # you could have invalid states as well. We ksip over such cubes
            invalid_state = False
            for bidx, e in enumerate(bCube_str):
                if e not in self.bVars_map[bidx].inv:
                    invalid_state = True
                    break
            if invalid_state:
                continue
            box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            print(f"[({self.pVar_map.inv[rConf_cube_str]}, {box_states}), {val}]")
    
    
    
    def convert_cube_to_state(self, dd: BDD) -> None:
        """
         Convert a cube to a state representation
        """
        cubes = []
        for cube_list in dd.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(cube_list):
                if self.manager.addVar(_idx) not in self.latches:
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
                    cubes.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                cubes.append(reduce(lambda a, b: a & b, var_list))
        

        # print the states
        for cube in cubes:
            rConf_cube_str = cube.existAbstract(self.all_bVars_cube).bddPattern().cubeString().replace('-', '')
            # for multiple boxes
            if self.boxes > 1:
                bCube_str = []
                for b in range(self.boxes):
                    all_but_b_cube = reduce(lambda a, b: a & b, self.bVars_cubes[:b] + self.bVars_cubes[b+1:])
                    # all_but_b_cube = self.bVars_cubes[0]
                    bCube_str.append(cube.existAbstract(all_but_b_cube & self.pVars_cube).bddPattern().cubeString().replace('-', ''))
            else:
                # for single box
                bCube_str = [cube.existAbstract(self.pVars_cube).bddPattern().cubeString().replace('-', '')]
            
            # you could have invalid states as well. We ksip over such cubes
            invalid_state = False
            for bidx, e in enumerate(bCube_str):
                if e not in self.bVars_map[bidx].inv:
                    invalid_state = True
                    break
            if invalid_state:
                continue
            box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            print(f"({self.pVar_map.inv[rConf_cube_str]}, {box_states})")
    
    
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


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        raise NotImplementedError()


    def solve(self):
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

            # go over all the env actions and preserve the maximum one
            # MaxUpre = []
            # # for env_tr_dd in self.eAction_map.values():
            # for env_tr_dd in self.env_action_cube_list:
            #     MaxUpre.append(preimage.restrict(env_tr_dd))
            
            # Upre = reduce(lambda x, y: x.max(y), MaxUpre)

            # go over all the sys actions and preserve the manimum one
            Minpre = []
            for robot_tr_dd in self.robot_action_cube_list:
                Minpre.append(preimage.restrict(robot_tr_dd))
            
            next_winning_states = reduce(lambda x, y: x.min(y), Minpre)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            # self.convert_cube_to_state_ADD(next_winning_states)
            
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
        self.convert_cube_to_state(pre)


if __name__ == "__main__":
    boxes = 3
    locs = 4
    init = ['ready l3', 'b0 l2', 'b1 l3']
    goal = ['ready l1', 'b0 l1', 'b1 l3']
    # init = ['ready l3', 'b0 l2']
    # goal = ['ready l1', 'b0 l1']
    # init = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l5']
    # goal = ['ready l1', 'b0 l1', 'b1 l3', 'b2 l5']
    fw = FrankaWorld(boxes=boxes, locs=locs, init=init, goal=goal)

    print('****************xVars map:****************')
    for k, v in fw.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************rAction map:****************')
    for k, v in fw.rAction_map.items():
        print(f"{k} : {v}")
    print("Total num of latches: ", len(fw.latches))

    # simple_franka_world()
    tic = time.time()
    fw.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic} seconds")

    # fw.test_pre_image()

    # print('Transition Relation:')
    # for k, v in fw.transition_relation.items():
    #     print(f"{k} : {v}")
    
    synth_start = time.time() 
    strategy = fw.solve()
    synth_stop = time.time()
    print(f"Time to synthesize strategy: {synth_stop - synth_start} seconds")
    # testing things out
    t = fw.get_all_states_interval(upper=4, dd=strategy, lower=4)
    print("Done")