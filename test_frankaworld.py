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
        self.human_action: List[str] = ['hmove']
        self.init = init
        self.goal = goal
        self.manager: Cudd = Cudd()

        self.pVars, self.bVars = self.create_latches()
        self.xVars: List[ADD] = self.pVars + [var for box_adds in self.bVars for var in box_adds]
        self.prime_pVars, self.prime_bVars = self.create_prime_latches()
        self.oVars: List[ADD] = self.create_output_vars()
        self.iVars: List[ADD] = self.create_input_vars()
        self.latches: List[ADD] = self.xVars # in future we will primed version of these as well.
        self.prime_latches: List[ADD] = self.prime_pVars + self.prime_bVars
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]

        # book keeping
        # self.holding_preds = self.to_obj_preds = self.ready_preds = set({})

        self.xVar_map = dict()
        self.rAction_map = bidict({})
        self.eAction_map = bidict({})
        self.xVar_map_sym  = dict()
        
        # maps needs for lookup of the states corresponding to cubes
        self.pVar_map = bidict({})
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}

        self.create_xVar_map()
        self.create_rAction_map()
        self.create_eAction_map()

        # more bookeeping stuff
        self.rAction_map_sym = bidict({k: self.cube_to_add(v, self.oVars) for k, v in self.rAction_map.items()})
        self.eAction_map_sym = bidict({k: self.cube_to_add(v, self.iVars) for k, v in self.eAction_map.items()})
        self.create_symbolic_maps()

        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()

        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}

        # need these cubes for printing states from cubes
        self.pVars_cube: ADD = reduce(lambda a, b: a & b, self.pVars)
        self.bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.bVars]
        self.all_bVars_cube: ADD = reduce(lambda a, b: a & b, self.bVars_cubes)
        self.iVars_cube: ADD = reduce(lambda x, y: x & y, self.iVars)
        self.oVars_cube: ADD = reduce(lambda x, y: x & y, self.oVars)

        # precompute cubes for iVars and oVars - needed for synthesis
        self.robot_action_cube_list: List[ADD] = [self.cube_to_add(r, self.oVars) for r in self.rAction_map.values()]
        self.env_action_cube_list: List[ADD] = [self.cube_to_add(e, self.iVars) for e in self.eAction_map.values()]

    
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
         Num. of robot actions = transit x |boxes| +  transfer x |locs| + grasp + release
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes + self.locs + 2
        oVars_size = math.ceil(math.log2(num_of_rActions))
        oVars: List[ADD] =  [self.manager.addVar(r + varsize , 'o' + str(r)) for r in range(oVars_size)]
        return oVars

    
    def create_input_vars(self) -> List[ADD]:
        """
         Num. of human actions = |boxes| x |locs| + 1 (for no-op action)
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes * self.locs + 1
        iVars_size = math.ceil(math.log2(num_of_rActions))
        iVars: List[ADD] =  [self.manager.addVar(h + varsize , 'i' + str(h)) for h in range(iVars_size)]
        return iVars
    
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
    

    def create_eAction_map(self) -> None:
        # Add a no-op action for the human
        self.eAction_map[f'{self.human_action[0]} noop'] = f"{0:0{len(self.iVars)}b}"
        offset = 1
        for b in range(self.boxes):
            for l in range(1, self.locs +1):
                act_str = f'{self.human_action[0]} b{b} l{l}'
                hbit_str = f"{offset:0{len(self.iVars)}b}"
                self.eAction_map[act_str] = hbit_str
                offset += 1
    
    
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
    

    # def add_human_moves(self):
    #     """
    #      The human action has some precondition that needs to satisfied. These are as follows:

    #     1. Human can not move a box that is being currently grasped that is in end effector loc. 
    #     2. Human can not move a box that will be grasped in the next step. 
    #     3. Human can not move a box to a loc where the robot is about release a box
    #     4. If the robot is transit-ing to a box b and human moves box to some other loc, then the state under the transit evolves
    #         to 'ready' conf. so that the robot can again transit to another object.
    #     """
    #     ee_empty_cube: ADD = self.create_ee_empty_cube()
    #     robot_action_is_release_cube = self.cube_to_add(self.rAction_map['release'], self.oVars)
    #     # Human Moves: human can move any box to any location, with preconditions
    #     for b_idx in range(self.boxes):
    #         # Rule 1 & 2 Preconditions: Box is not held and robot is not about to grasp it.
    #         # These are independent of the destination location.
    #         # box_not_held_cube = ~self.cube_to_add(self.xVar_map[f'b{b_idx} l0'], self.bVars[b_idx])
    #         robot_not_to_obj_cube = ~self.cube_to_add(self.xVar_map[f'to-obj b{b_idx}'], self.pVars)
            
    #         for l_idx in range(1, self.locs + 1):
    #             h_act_str: str = f'{self.human_action[0]} b{b_idx} l{l_idx}'
    #             h_act_cube: ADD = self.cube_to_add(self.eAction_map[h_act_str], self.iVars)

    #             # Rule 3 Precondition: Robot is not holding a box at the destination location.
    #             forbidden_release_cond: ADD = self.cube_to_add(self.xVar_map[f'holding l{l_idx}'], self.pVars) & robot_action_is_release_cube
    #             robot_not_releasing_at_dest_cube: ADD = ~forbidden_release_cond

    #             # Combine all preconditions for this specific human move
    #             # pre_condition_cube = box_not_held_cube & robot_not_to_obj_cube & robot_not_releasing_at_dest_cube
    #             pre_condition_cube = robot_not_to_obj_cube & robot_not_releasing_at_dest_cube

    #             # The full condition for this transition to occur
    #             transition_cube = h_act_cube & pre_condition_cube

    #             # Define the next state for the box being moved
    #             box_next_state_str = self.xVar_map[f'b{b_idx} l{l_idx}']
    #             for sidx, s in enumerate(box_next_state_str):
    #                 if s == '1':
    #                     self.transition_relation[self.bVars[b_idx][sidx].bddPattern().__str__()] |= transition_cube

        
    
    def create_transition_relation(self) -> None:

        # The human does nothing, robot acts as normal
        h_noop_cube = self.cube_to_add(self.eAction_map['hmove noop'], self.iVars)
        
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
                    for turn in ['human-move', 'human-no-move']:
                        if turn == 'human-no-move':
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
                        
                        elif turn == 'human-move':
                            for human_to_loc in range(1, self.locs + 1):
                                if human_to_loc == to_loc:
                                    continue
                                h_act_str: str = f'{self.human_action[0]} b{b} l{human_to_loc}'
                                h_act_cube: ADD = self.cube_to_add(self.eAction_map[h_act_str], self.iVars)
                                box_clause_prime_string = self.xVar_map['b' + str(b) + ' l' + str(human_to_loc)]
                                pred_clause_prime_string = self.xVar_map['ready l' + str(to_loc)]
                                
                                for sidx, s in enumerate(pred_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube & h_act_cube
                                
                                for sidx, s in enumerate(box_clause_prime_string):
                                    if s == '1':
                                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube & h_act_cube
                        
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
                                                self.transition_relation[self.bVars[other_b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube & self.cube_to_add(self.xVar_map[frame_box_pred], self.bVars[other_b]) & h_act_cube

        
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
                # TODO: Need to enforce that the box can only be release at an empty location
                
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
        
        # add human moves to the transition relation
        self.add_human_moves()
    

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

    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, robot_action: bool = False, human_action: bool = False ) -> None:
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
        if human_action:
            relevant_vars.extend(self.iVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])
        start_ivar_idx, end_ivar_idx = self.manager.addVariables().index(self.iVars[0]), self.manager.addVariables().index(self.iVars[-1])
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.xVars[len(self.pVars):] + self.oVars + self.iVars)
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.pVars + self.oVars + self.iVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.pVars + self.oVars + self.iVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
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
            print("Current Preimage:")
            self.convert_cube_to_state_ADD(preimage, state_flag=True, robot_action=True, human_action=True)

            # go over all the env actions and preserve the maximum one
            MaxUpre = []
            # for env_tr_dd in self.eAction_map.values():
            for env_tr_dd in self.env_action_cube_list:
                MaxUpre.append(preimage.restrict(env_tr_dd))
            
            Upre = reduce(lambda x, y: x.min(y), MaxUpre)

            # go over all the sys actions and preserve the manimum one
            Minpre = []
            for robot_tr_dd in self.robot_action_cube_list:
                Minpre.append(Upre.restrict(robot_tr_dd))
            
            next_winning_states = reduce(lambda x, y: x.min(y), Minpre)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            print("Current Winning States:")
            self.convert_cube_to_state_ADD(next_winning_states)
            
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




class FrankaWorldDynamic(FrankaWorld):

    def __init__(self, boxes, locs, init, goal):
        super().__init__(boxes, locs, init, goal)
        self.one_box_per_loc_cube = defaultdict(lambda: self.manager.addZero())
        self.ee_empty_cube: ADD = self.create_ee_empty_cube()
    

    def create_one_box_per_loc(self):
        """
         Create a cube that enforces that only one box can be at a location at any time.
         Say x is cube that enforces that box b at loc l, then we need to add the following clauses:
            For all other boxes b', ~(b' at l). 
        """
        for l in range(1, self.locs + 1):
            for b in range(self.boxes):
                curr_box_pred = f"b{b} l{l}"
                self.one_box_per_loc_cube[curr_box_pred] |= self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(l),
                                                                                         bConf_cube=self.xVar_map_sym[curr_box_pred])
    


    def create_grasp_actions(self):
        """
         Create grasp actions for the robot. For each grasp action we create all possible human actions. 
            Human can move any box to any location, as long as the robot is not about to grasp that box.

         For a fixed robot action, we first construct all valid human moves and the corresponding transition cubes.
         Next, we add the transition where the human does no move as ~(valid_human_moves).

        """
        robot_act_cube = self.rAction_map_sym['grasp']
        for b in range(self.boxes):
            rConf_cube = self.xVar_map_sym[f'to-obj b{b}']

            # for a give box, it can any location, so we iterate over all locations
            for loc in range(1, self.locs + 1):
                curr_box_pred = f"b{b} l{loc}"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                # need to enforce that the end-effector is empty
                state_constraint_cube = self.ee_empty_cube
                
                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(loc), bConf_cube=bConf_cube)

                robot_transition_cube = rConf_cube & bConf_cube & state_constraint_cube & robot_act_cube

                # this is fixed and is independent of the human action
                pred_clause_prime_string = self.xVar_map['holding l' + str(loc)]
                box_clause_prime_string = self.xVar_map[f'b{b} l0']

                # we frist create all valid human moves
                valid_human_moves = self.manager.addZero()
                for other_b in range(self.boxes):
                    if other_b == b:
                        continue
                    for human_to_loc in range(1, self.locs + 1):
                        h_act_str: str = f'{self.human_action[0]} b{other_b} l{human_to_loc}'
                        h_act_cube: ADD = self.eAction_map_sym[h_act_str]
                        valid_human_moves |= h_act_cube

                        human_transition_cube = robot_transition_cube & h_act_cube
                        hbox_clause_prime_string = self.xVar_map[f'b{other_b} l{human_to_loc}']

                        # here we will only add the next state clauses for the box being moved by the human
                        for sidx, s in enumerate(hbox_clause_prime_string):
                            if s == '1':
                                self.transition_relation[self.bVars[other_b][sidx].bddPattern().__str__()] |= human_transition_cube
                
                # now we add the transition where the human does all the valid move and the robot grasps the box
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube & valid_human_moves
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube & ~valid_human_moves
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube & valid_human_moves
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube & ~valid_human_moves
    

    def create_release_actions(self):
        pass


    def create_transit_actions(self):
        pass


    def create_treansfer_actions(self):
        pass


    def preimage_test(From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
        From = From.swapVariables(latches, prime_latches)
        return From.vectorCompose(prime_latches, ts_action)



    def test_pre_image(self):
        ts_action = list(self.transition_relation.values())
        # restrict TR to be ove states where b1 and b2 are not l1
        # b1_l1 = self.xVar_map_sym['b1 l1']
        # b2_l1 = self.xVar_map_sym['b2 l1']
        # restric_cube = ~(b1_l1 | b2_l1)
        # ts_action = [tr.restrict(restric_cube) for tr in ts_action]


        # goal state is b0 and l0 and ready l0
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars) 
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['b1 l0'], self.bVars[1]) & self.cube_to_add(self.xVar_map['holding l2'], self.pVars) 
        goal_cube = self.xVar_map_sym['holding l2'] & self.xVar_map_sym['b0 l0'] & self.xVar_map_sym['b1 l1']
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['b1 l2'], self.bVars[1]) & self.cube_to_add(self.xVar_map['to-obj b1'], self.pVars)
        # goal_cube = self.cube_to_add(self.xVar_map['b1 l2'], self.bVars[1]) & self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars)
        print('Goal state:', goal_cube)
        
        preimage = preimage_test(From=goal_cube,
                                 latches=self.latches,
                                 prime_latches=self.prime_latches,
                                 ts_action=ts_action)

        print('Preimage: \n', preimage)
        self.convert_cube_to_state_ADD(preimage, state_flag=True, robot_action=False, human_action=False)
    
    
    def create_transition_relation(self):
        """
         Overriding the transition relation creation function to account for dynamic human actions
        """
        # first we create grasp actions
        self.create_grasp_actions()

        # next we create the release actions
        self.create_release_actions()

        # next we create the transit actions
        self.create_transit_actions()

        # finally we create the transfer actions
        self.create_treansfer_actions()



def preimage_test(From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
    From = From.swapVariables(latches, prime_latches)
    return From.vectorCompose(prime_latches, ts_action)


def test_dynamic_franka_world():
    manager = Cudd()
    # ready l4, ready l1; ready l2; ready l3, to-obj b0
    p0, p1, p2 = manager.addVar(0, 'p0'), manager.addVar(1, 'p1'), manager.addVar(2, 'p2')
    # b0 l1, b0 l2, b0 l0, b0 l3
    b0, b1, b2 = manager.addVar(3, 'b00'), manager.addVar(4, 'b01'), manager.addVar(5, 'b02')
    # bookkeeping
    pVars = [p0, p1, p2]
    bVars = [b0, b1, b2]

    # create prime vars
    offset = len(pVars) + len(bVars)
    p0_p, p1_p, p2_p = manager.addVar(offset, "pp0"), manager.addVar(offset + 1, "pp1"), manager.addVar(offset + 2, "pp2")
    b0_p, b1_p, b2_p = manager.addVar(offset + 3, "pb00"), manager.addVar(offset + 4, "pb01"), manager.addVar(offset + 5, "pb02")

    # bookkeeping
    prime_pVars = [p0_p, p1_p, p2_p]
    prime_bVars = [b0_p, b1_p, b2_p]

    # create robot action vars - transit b0
    offset = len(pVars) + len(bVars) + len(prime_pVars) + len(prime_bVars)
    # transit b0 + dummy var
    o0 = manager.addVar(offset, 'o0')
    # hmove b0 l1, hmove b0 l2 + dummy var
    i0, i1 = manager.addVar(offset + 1, 'i0'), manager.addVar(offset + 2, 'i1')  

    transition_relation = {var.bddPattern().__str__(): manager.addZero() for var in [p0, p1, p2, b0, b1, b2]}

    # create cubes
    ready_l4 = ~p0 & ~p1 & p2
    ready_l1 = ~p0 & p1 & ~p2
    ready_l2 = ~p0 & p1 & p2
    to_obj_b0 = p0 & ~p1 & ~p2
    ready_l3 = p0 & ~p1 & p2

    # box cubes
    b0_l0 = ~b0 & ~b1 & b2
    b0_l1 = ~b0 & b1 & ~b2
    b0_l2 = ~b0 & b1 & b2
    b0_l3 = b0 & ~b1 & ~b2

    transit_b0 = o0
    hmove_b0_l3 = ~i0 & i1
    hmove_b0_l2 = i0 & ~i1

    # (ready l4) (b0 l1) --- (transit b0) ---> (to-obj b0) (b0 l1)
    tr1 = ready_l4 & b0_l1 & transit_b0

    rConf_prime_str = '100'
    bConf_prime_str = '010'
    for sidx, s in enumerate(rConf_prime_str):
        if s == '1':
            transition_relation[pVars[sidx].bddPattern().__str__()] |= tr1 & ~(hmove_b0_l2 | hmove_b0_l3)
    
    for sidx, s in enumerate(bConf_prime_str):
        if s == '1':
            transition_relation[bVars[sidx].bddPattern().__str__()] |= tr1 & ~(hmove_b0_l2 | hmove_b0_l3)
    
    # (ready l4) (b0 l1) --- (transit b0) (hmove b0 l2) ---> (ready l1) (b0 l2)
    rConf_prime_str = '010'
    bConf_prime_str = '011'
    for sidx, s in enumerate(rConf_prime_str):
        if s == '1':
            transition_relation[pVars[sidx].bddPattern().__str__()] |= tr1 & hmove_b0_l2
    
    for sidx, s in enumerate(bConf_prime_str):
        if s == '1':
            transition_relation[bVars[sidx].bddPattern().__str__()] |= tr1 & hmove_b0_l2
    

    # (ready l4) (b0 l1) --- (transit b0) (hmove b0 l3) ---> (ready l1) (b0 l3)
    rConf_prime_str = '010'
    bConf_prime_str = '100'
    for sidx, s in enumerate(rConf_prime_str):
        if s == '1':
            transition_relation[pVars[sidx].bddPattern().__str__()] |= tr1 & hmove_b0_l3
    
    for sidx, s in enumerate(bConf_prime_str):
        if s == '1':
            transition_relation[bVars[sidx].bddPattern().__str__()] |= tr1 & hmove_b0_l3
    
    # goal_latch = to_obj_b0 & b0_l1
    # goal_latch = ready_l1 & b0_l3
    # lets test the restict functionality - lets restrict human moves to be anything but hmove_b0_l2
    ts_action = list(transition_relation.values())
    print("Testing Restrict Functionality")
    tr_restricted = [e.restrict(~hmove_b0_l2) for e in transition_relation.values()]
    # ts_action = tr_restricted
    
    goal_latch = b0_l2 
    preimage = preimage_test(From=goal_latch,
                            latches=pVars + bVars,
                            prime_latches=prime_pVars + prime_bVars,
                            ts_action=ts_action)


    
    
    print("Preimage: ", preimage)
    



    



if __name__ == "__main__":
    boxes = 2
    locs = 3
    init = ['ready l3', 'b0 l2', 'b1 l3']
    goal = ['ready l1', 'b0 l1', 'b1 l3']
    # init = ['ready l3', 'b0 l2']
    # goal = ['holding l2', 'b0 l0']
    # goal = ['ready l1', 'b0 l1']
    # init = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l4']
    # goal = ['ready l1', 'b0 l1', 'b1 l3', 'b2 l4']
    # fw = FrankaWorld(boxes=boxes, locs=locs, init=init, goal=goal)
    fw = FrankaWorldDynamic(boxes=boxes, locs=locs, init=init, goal=goal)

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

    fw.test_pre_image()

    # print('Transition Relation:')
    # for k, v in fw.transition_relation.items():
    #     print(f"{k} : {v}")
    
    # synth_start = time.time() 
    # strategy = fw.solve()
    # synth_stop = time.time()
    # print(f"Time to synthesize strategy: {synth_stop - synth_start} seconds")
    # testing things out
    # t = fw.get_all_states_interval(upper=4, dd=strategy, lower=4)
    print("Done")

    # test_dynamic_franka_world()