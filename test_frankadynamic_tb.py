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

from typing import List, Dict, Tuple
from functools import reduce
from itertools import product
from collections import defaultdict


from bidict import bidict
from cudd import Cudd, ADD, BDD


# from test_frankaworld import FrankaWorld


class FrankaWorldDyanmicTurnBased():

    def __init__(self, boxes: int, locs: int, init: tuple, goal: tuple, human_locs: List[int]):
        self.boxes: int = boxes
        self.locs: int = locs
        self.human_locs: List[int] = human_locs
        self.misc_preds = ['ready', 'in-transit', 'in-transfer' 'to-obj', 'holding']
        self.robot_actions: List[str] = ['transit', 'transfer', 'grasp', 'release']
        self.init = init
        self.goal = goal
        self.manager: Cudd = Cudd()

        # create turn variable
        # self.tVar: ADD = [self.manager.addVar(0, 't0'), self.manager.addVar(1, 't1')]
        self.tVar: List[ADD] = [self.manager.addVar(0, 't0')]
        self.pVars, self.bVars = self.create_latches()
        self.xVars: List[ADD] = self.pVars + [var for box_adds in self.bVars for var in box_adds]
        
        # create prime turn latch
        offset = self.manager.size()
        # self.prime_tVar: ADD = [self.manager.addVar(offset, "pt0"), self.manager.addVar(offset + 1, "pt1")]
        self.prime_tVar: List[ADD] = [self.manager.addVar(offset, "pt0")]
        self.prime_pVars, self.prime_bVars = self.create_prime_latches()
        self.oVars: List[ADD] = self.create_output_vars()

        self.latches: List[ADD] = self.tVar + self.xVars
        self.prime_latches: List[ADD] = self.prime_tVar + self.prime_pVars + self.prime_bVars
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]
        self.prime_latches_bdd: List[BDD] = [var.bddPattern() for var in self.prime_latches]

        self.weight_dict: Dict[str, int] = {'transit': 1, 'transfer': 1, 'grasp': 1, 'release': 1}
        self.symbolic_weight_dict: Dict[str, ADD] = defaultdict(lambda: self.manager.addOne())

        self.xVar_map = dict()
        self.rAction_map = bidict({})
        self.xVar_map_sym  = dict()
        
        # maps needs for lookup of the states corresponding to cubes
        self.pVar_map = bidict({})
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}
        
        # different from frankaworld, we need a turn variable map
        self.tVar_map = bidict({'robot': '1', 'human': '0'})
        self.tVar_map_sym = bidict({'robot': self.cube_to_add(self.tVar_map['robot'], self.tVar),
                                    'human': self.cube_to_add(self.tVar_map['human'], self.tVar)})
        self.create_xVar_map()
        self.create_rAction_map()

        # more bookeeping stuff
        self.rAction_map_sym = bidict({k: self.cube_to_add(v, self.oVars) for k, v in self.rAction_map.items()})
        self.create_symbolic_maps()
        self.create_sym_weight_dict()

        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()

        # monolithic transition relation for the robot actions
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
        self.locs_empty_constraints = defaultdict(lambda: self.manager.addZero())
        self.create_loc_empty_constraint()

        # create env move related vars and maps
        self.human_action: List[str] = ['hmove']
        self.iVars: List[ADD] = self.create_input_vars()
        self.eAction_map = bidict({})
        self.create_eAction_map()
        self.eAction_map_sym = bidict({k: self.cube_to_add(v, self.iVars) for k, v in self.eAction_map.items()})
        self.iVars_cube: ADD = reduce(lambda x, y: x & y, self.iVars)

        # precompute cubes for iVars and oVars - needed for synthesis
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
        # additional preds: in-transit x |locs| x |boxes| + in-transfer x |locs| x |locs|
        num_of_preds = 2*self.locs + self.boxes + 2 + 1 # +1 for ready-else state
        num_of_preds += self.locs * self.boxes # in-transit preds
        num_of_preds += self.locs * self.locs # in-transfer preds
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
                bit_str = f"{offset + loc:0{len(self.pVars)}b}"
                self.xVar_map[f'in-transit l{loc} b{b}'] = bit_str
                self.pVar_map[f'in-transit l{loc} b{b}'] = bit_str
            offset += self.locs + 1
        
        # create preds for in-transfer
        for from_loc in range(1, self.locs + 1):
            for to_loc in range(1, self.locs + 1):
                bit_str = f"{offset + to_loc:0{len(self.pVars)}b}"
                self.xVar_map[f'in-transfer l{from_loc} l{to_loc}'] = bit_str
                self.pVar_map[f'in-transfer l{from_loc} l{to_loc}'] = bit_str
            offset += self.locs

        # for each boxes we create |locs| boolean vars
        for b in range(self.boxes):
            for l in range(self.locs + 1):
                bit_str = f"{l + 1:0{len(self.bVars[b])}b}"
                self.xVar_map['b' + str(b) + ' l' + str(l)] = bit_str
                self.bVars_map[b]['b' + str(b) + ' l' + str(l)] = bit_str

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
        num_of_rActions = self.boxes * self.locs + 1 + 1 # to offset the 0-vector
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
        self.eAction_map[f'{self.human_action[0]} noop'] = f"{1:0{len(self.iVars)}b}"
        offset = 2
        for b in range(self.boxes):
            # TODO: Update this in future to restrict human moves to certain locations only
            for l in range(1, self.locs +1):
                act_str = f'{self.human_action[0]} b{b} l{l}'
                hbit_str = f"{offset:0{len(self.iVars)}b}"
                self.eAction_map[act_str] = hbit_str
                offset += 1

    def create_sym_weight_dict(self) -> None:
        # for ract, w in self.weight_dict.items():
        for ract, dd in self.rAction_map_sym.items():
            # extract the name
            act_name: str = ract.split(' ')[0]
            w = self.weight_dict[act_name]
            self.symbolic_weight_dict[ract] = dd.ite(self.manager.addConst(w), self.manager.addOne())
        
        self.weight = reduce(lambda x, y: x & y, self.symbolic_weight_dict.values())
    
    
    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add


    def set_init_latch(self) -> ADD:
        init_cube = self.tVar_map_sym['robot']
        for s in self.init:
            init_cube &= self.xVar_map_sym[s]
        return init_cube
    

    def set_goal_latch(self) -> ADD:
        goal_cube = self.tVar_map_sym['robot']
        for s in self.goal:
            goal_cube &= self.xVar_map_sym[s]
        return goal_cube
    

    def create_ee_empty_cube(self) -> ADD:
        # cube that implies that end-effector location empty
        bConf_cube = self.manager.addOne()
        for b in range(self.boxes):
            bConf_cube &= ~self.xVar_map_sym['b' + str(b) + ' l0']
        return bConf_cube
    
    def create_only_b_at_l_cube(self, curr_box: int, curr_loc: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.xVar_map_sym['b' + str(ob) + f' {curr_loc}']
        return bConf_cube
    
    

    def create_transition_relation(self):
        """
         A method to construct the transition relation for the Franka World Dynamic domain. 
         This constructs the transition system that consists of actions controlled by the robot (robot actions) and human (env actions).

         After every turn, the turn variable is flipped and the game evolves to the other player.
        """
        # first we create grasp actions
        self.create_grasp_actions()

        # next we create the release actions
        # self.create_release_actions()

        # next we create the transit actions
        self.create_transit_actions()

        # next we create the transfer actions
        # self.create_transfer_actions()

        # add robot frame axioms
        self.add_robot_frame_axioms()

        # finally, we add frame axioms for all boxes that enforce state invariance constraint
        # self.add_frame_axioms()
        self.create_human_move_transit()
        self.create_human_move_actions()

        self.add_human_frame_axiom()
    

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
                curr_box_pred = f"b{b} l{loc}"
                bConf_cube = self.xVar_map_sym[curr_box_pred]
                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(loc), bConf_cube=bConf_cube)

                robot_transition_cube = turn_bit & rConf_cube & bConf_cube & state_constraint_cube & robot_act_cube

                # this is fixed and is independent of the human action
                turn_prime_string = self.tVar_map['human']
                pred_clause_prime_string = self.xVar_map['holding l' + str(loc)]
                box_clause_prime_string = self.xVar_map[f'b{b} l0']

                for sidx, s in enumerate(turn_prime_string):
                    if s == '1':
                        self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= robot_transition_cube
                
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
        raise NotImplementedError("Release action not implemented yet")
    

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
        turn_prime_string = self.tVar_map['human']
        turn_bit = self.tVar_map_sym['robot']
        for b in range(self.boxes):
            robot_act_cube = self.rAction_map_sym[f"transit b{b}"]

            for from_loc in range(1, self.locs + 2):
                rConf_cube = self.xVar_map_sym[f'ready l{from_loc}']
                
                # for to_loc in range(1, self.locs + 1):
                # we skip transiting to the same location
                # if from_loc == to_loc:
                #     continue
                # curr_box_pred: str = f"b{b} l{to_loc}"
                # bConf_cube = self.xVar_map_sym[curr_box_pred]
                
                # need to enforce that only one box is at loc l
                # bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(to_loc), bConf_cube=bConf_cube)

                robot_transition_cube = turn_bit & rConf_cube & state_constraint_cube & robot_act_cube

                # box_clause_prime_string = self.xVar_map[curr_box_pred]
                pred_clause_prime_string = self.xVar_map[f"in-transit l{from_loc} b{b}"]
                
                # now we add the transition where the human does all the valid move and the robot grasps the box
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                
                for sidx, s in enumerate(turn_prime_string):
                    if s == '1':
                        self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= robot_transition_cube



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
        raise NotImplementedError("Transfer action not implemented yet")
    

    def create_human_move_transit(self) -> None:
        """
         From the in-transit states, if human moves a box, the robot conf and the box conf. change. 
          If the human does not move a box, the robot conf. evolves to to-obj boc and box conf. remain the same for all the boxes. 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        turn_prime_string = self.tVar_map['robot']
        for b in range(self.boxes):
            for from_loc in range(1, self.locs + 2):
                # for all in-transit preds
                rConf_cube = self.xVar_map_sym[f'in-transit l{from_loc} b{b}']

                for human_box in range(self.boxes):
                    for human_to_loc in self.human_locs:
                        hmove_cube = turn_bit & \
                            self.eAction_map_sym[f'hmove b{human_box} l{human_to_loc}'] & self.locs_empty_constraints[f'l{human_to_loc}'] \
                            & ~self.xVar_map_sym[f'b{human_box} l0']
                        
                        for sidx, s in enumerate(turn_prime_string):
                            if s == '1':
                                self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= hmove_cube & rConf_cube & self.ee_empty_cube
                        
                        pred_clause_prime_string = self.xVar_map[f'ready l{from_loc}']
                        for sidx, s in enumerate(pred_clause_prime_string):
                            if s == '1':
                                self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube & rConf_cube & self.ee_empty_cube #& ~self.xVar_map_sym[f'b{b} l{from_loc}']
                
                    
            # add the human noop action here
            hmove_cube = turn_bit & self.eAction_map_sym['hmove noop']

            for sidx, s in enumerate(turn_prime_string):
                if s == '1':
                    self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= hmove_cube & rConf_cube & self.ee_empty_cube
                    
            pred_clause_prime_string = self.xVar_map[f'to-obj b{b}']
            for sidx, s in enumerate(pred_clause_prime_string):
                if s == '1':
                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube & rConf_cube & self.ee_empty_cube


    def create_human_move_actions(self) -> None:
        """
        Here we create human move actions for the environment. The precondition for the human to move box b to location l is as follows:
         Preconditions:
            1. The human can move box b to location l only if location l is empty - (b @ l') & ~(b @ l) predicates are true at current state 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        turn_prime_string = self.tVar_map['robot']
        for hb in range(self.boxes):
            for human_to_loc in self.human_locs:
                hmove_cube = turn_bit & \
                    self.eAction_map_sym[f'hmove b{hb} l{human_to_loc}'] & self.locs_empty_constraints[f'l{human_to_loc}'] \
                        & ~self.xVar_map_sym[f'b{hb} l0']
                
                for sidx, s in enumerate(turn_prime_string):
                    if s == '1':
                        self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= hmove_cube
                
                # The moved box `hb` goes to its new location
                moved_box_prime_str = self.xVar_map[f'b{hb} l{human_to_loc}']
                for sidx, s in enumerate(moved_box_prime_str):
                    if s == '1':
                        self.transition_relation[self.bVars[hb][sidx].bddPattern().__str__()] |= hmove_cube
    

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
            not_grasp_cube = ~(self.xVar_map_sym[f'to-obj b{b}'] & grasp_action_cube)
            for l in range(1, self.locs + 1):
                box_pred = f"b{b} l{l}"
                constraint_cube = self.manager.addOne()
                not_release_cube = ~(self.xVar_map_sym[f'holding l{l}'] & release_action_cube)
                constraint_cube &= not_grasp_cube & not_release_cube
                
                for sidx, s in enumerate(self.tVar_map['human']):
                    if s == '1':
                        self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= self.tVar_map_sym['robot'] & constraint_cube & self.xVar_map_sym[box_pred]
                
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= self.tVar_map_sym['robot'] & constraint_cube & self.xVar_map_sym[box_pred]

    def add_human_frame_axiom(self):
        """
        This helper method add frame axioms for the human move actions. 
         Boxes that the human does not move should remain in the same location in the next state.
        """
        noop_cube = self.tVar_map_sym['human'] & self.eAction_map_sym['hmove noop']
        for b in range(self.boxes):
            for l in range(0, self.locs + 1):
                box_pred = f"b{b} l{l}"
                haction_cube = noop_cube
                # b# l0 stay constant irrespective of human action as l0 is end-effector loc;
                # b# l (l > 0) stay constant if human does not move it.
                if l == 0:
                    haction_cube = self.tVar_map_sym['human']

                # turn to robot state after human move
                for sidx, s in enumerate(self.tVar_map['robot']):
                    if s == '1':
                        self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= haction_cube & self.xVar_map_sym[box_pred]
                # box remmains in the same location if human does not move it
                for sidx, s in enumerate(self.xVar_map[box_pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= haction_cube & self.xVar_map_sym[box_pred]

        # the robot's holding conf. remains the same after human move
        for l in range(1, self.locs + 1):
            rConf_cube = self.xVar_map_sym[f'holding l{l}']
            for sidx, s in enumerate(self.tVar_map['robot']):
                if s == '1':
                    self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= self.tVar_map_sym['human'] & rConf_cube
            
            pred_clause_prime_string = self.xVar_map[f'holding l{l}']
            for sidx, s in enumerate(pred_clause_prime_string):
                if s == '1':
                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= self.tVar_map_sym['human'] & rConf_cube

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

    
    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, robot_action: bool = False,  human_action: bool = False) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] + self.tVar
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
        
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.pVars):] + self.oVars + self.iVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.pVars + self.oVars + self.iVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.pVars + self.oVars + self.iVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        for cube, val in cubes:
            tConf_exist_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                print(f"[({self.tVar_map.inv[tConf_exist_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states}), {val}]")
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



    def test_pre_image(self):
        # convert transition relation to latches bdd
        # tr_bdd = [dd.bddPattern() for dd in self.transition_relation.values()]

        # ready to transit testing
        # ready l3 -> transit b0 -> in-transit l3 b0
        # goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['in-transit l3 b0']

        # in-transit l3 b0 -> hmove b0 l2 -> to-obj b0
        # goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['to-obj b0'] & self.xVar_map_sym['b0 l1']
        # goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['ready l1'] & self.xVar_map_sym['b0 l2']

        # goal state is b0 and l0 and ready l0
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars) 
        goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['holding l2'] & self.xVar_map_sym['b0 l0'] & self.xVar_map_sym['b1 l1'] & self.xVar_map_sym['b2 l2']  
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['b1 l0'], self.bVars[1]) & self.cube_to_add(self.xVar_map['holding l2'], self.pVars) 
        # goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['holding l2'] & self.xVar_map_sym['b0 l0'] & self.xVar_map_sym['b1 l3']
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['b1 l2'], self.bVars[1]) & self.cube_to_add(self.xVar_map['to-obj b1'], self.pVars)
        # goal_cube = self.cube_to_add(self.xVar_map['b1 l2'], self.bVars[1]) & self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars)
        print('Goal state:', goal_cube)
        From = goal_cube

        From = goal_cube.swapVariables(self.latches, self.prime_latches)

        preimage = From.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        print('Preimage: ', preimage)
        self.convert_cube_to_state_ADD(preimage, human_action=True, robot_action=False)


def preimage_test(From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
    From = From.swapVariables(latches, prime_latches)
    return From.vectorCompose(prime_latches, ts_action)



def test_dynamic_franka_world():
    def tconf_cube_to_tr(transition_relation: Dict[str, ADD], tConf_prime_str: str, tr_cube: ADD, tVars: List[ADD]):
        for sidx, s in enumerate(tConf_prime_str):
            if s == '1':
                transition_relation[tVars[sidx].bddPattern().__str__()] |= tr_cube
        return transition_relation
    
    def rconf_cube_to_tr(transition_relation: Dict[str, ADD], rConf_prime_str: str, tr_cube: ADD, pVars: List[ADD]):
        for sidx, s in enumerate(rConf_prime_str):
            if s == '1':
                transition_relation[pVars[sidx].bddPattern().__str__()] |= tr_cube
        return transition_relation
    

    def bconf_cube_to_tr(transition_relation: Dict[str, ADD], bConf_prime_str: str, bidx:int, tr_cube: ADD, bVars: List[ADD]):
        for sidx, s in enumerate(bConf_prime_str):
            if s == '1':
                transition_relation[bVars[bidx][sidx].bddPattern().__str__()] |= tr_cube
        return transition_relation
    
    manager = Cudd()
    # turn variable 
    t = manager.addVar(0, 't')
    offset = 1
    # ready l4, ready l1; ready l2; ready l3, to-obj b0; to_obj b1
    p0, p1, p2 = manager.addVar(offset + 0, 'p0'), manager.addVar(offset + 1, 'p1'), manager.addVar(offset + 2, 'p2')
    # b0 l1, b0 l2, b0 l0, b0 l3
    b00, b01, b02 = manager.addVar(offset + 3, 'b00'), manager.addVar(offset + 4, 'b01'), manager.addVar(offset + 5, 'b02')
    # b1 l1, b1 l2, b1 l0, b1 l3
    b10, b11, b12 = manager.addVar(offset + 6, 'b10'), manager.addVar(offset + 7, 'b11'), manager.addVar(offset + 8, 'b12')
    # bookkeeping
    pVars = [p0, p1, p2]
    bVars = [b00, b01, b02, b10, b11, b12]

    # create prime vars
    offset = len([t]) + len(pVars) + len(bVars)
    prime_t = manager.addVar(offset, 'pt')
    offset += 1
    p0_p, p1_p, p2_p = manager.addVar(offset + 0, "pp0"), manager.addVar(offset + 1, "pp1"), manager.addVar(offset + 2, "pp2")
    b00_p, b01_p, b02_p = manager.addVar(offset + 3, "pb00"), manager.addVar(offset + 4, "pb01"), manager.addVar(offset + 5, "pb02")
    b10_p, b11_p, b12_p = manager.addVar(offset + 6, "pb10"), manager.addVar(offset + 7, "pb11"), manager.addVar(offset + 8, "pb12")

    # bookkeeping
    prime_pVars = [p0_p, p1_p, p2_p]
    prime_bVars = [b00_p, b01_p, b02_p, b10_p, b11_p, b12_p]

    # create robot action vars - transit b0
    offset = len([t]) + len(pVars) + len(bVars) + len([prime_t]) + len(prime_pVars) + len(prime_bVars)
    # transit b0; transit b1 + dummy var
    o0, o1 = manager.addVar(offset, 'o0'), manager.addVar(offset + 1, 'o1')
    # hmove b0 l1, hmove b0 l2; hmove b0 l3; hmove b1 l1, hmove b1 l2, hmove b1 l3 + dummy var
    i0, i1, i2 = manager.addVar(offset + 2, 'i0'), manager.addVar(offset + 3, 'i1'), manager.addVar(offset + 4, 'i2')  

    transition_relation: Dict[str, ADD] = {var.bddPattern().__str__(): manager.addZero() for var in [t, p0, p1, p2, b00, b01, b02, b10, b11, b12]}

    # turn cubes
    robot_turn = t
    human_turn = ~t

    # create cubes
    to_obj_b0 = ~p0 & ~p1 & p2
    to_obj_b1 = ~p0 & p1 & ~p2
    holding_l1 = ~p0 & p1 & p2
    holding_l2 = p0 & ~p1 & ~p2
    holding_l3 = p0 & ~p1 & p2

    # box cubes
    b0_l0 = ~b00 & ~b01 & b02
    b0_l1 = ~b00 & b01 & ~b02
    b0_l2 = ~b00 & b01 & b02
    b0_l3 = b00 & ~b01 & ~b02

    b1_l0 = ~b10 & ~b11 & b12
    b1_l1 = ~b10 & b11 & ~b12
    b1_l2 = ~b10 & b11 & b12
    b1_l3 = b10 & ~b11 & ~b12
    
    # 0-vector is skipped
    # transit_b0 = ~o0 & o1
    # transit_b1 = o0 & ~o1
    grasp = ~o0 & o1
    
    # human move cubes
    hmove_noop = ~i0 & ~i1 & ~i2
    hmove_b0_l3 = ~i0 & ~i1 & i2
    hmove_b0_l2 = ~i0 & i1 & ~i2
    hmove_b0_l1 = ~i0 & i1 & i2
    hmove_b1_l3 = i0 & ~i1 & ~i2
    hmove_b1_l2 = i0 & ~i1 & i2
    hmove_b1_l1 = i0 & i1 & ~i2

    # locs empty constraints
    l1_empty = ~(b0_l1 | b1_l1)
    l2_empty = ~(b0_l2 | b1_l2)
    l3_empty = ~(b0_l3 | b1_l3)

    # (Sys) (to-obj b0) (b0 l1)  ---- (grasp) ----> (Env)(holding l1) (b0 l0)
    # robot_action = to_obj_b0 & b0_l1 & b1_l2 & robot_turn & grasp
    robot_action = to_obj_b0 & b0_l1 & robot_turn & grasp
    prime_turn_str = '0' # human turn
    prime_rConf_str = '011' # holding l1
    prime_bConf_str_b0 = '001' # b0 l0
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=robot_action,
    #                                        tVars=[t])
    transition_relation = rconf_cube_to_tr(transition_relation=transition_relation,
                                           rConf_prime_str=prime_rConf_str,
                                           tr_cube=robot_action,
                                           pVars=pVars)
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=robot_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
                                           
    
    # (Env) (holding l1)  ---- (hmove noop) ----> (Sys) (holding l1)
    # human_action = holding_l1 & b0_l0 & b1_l2 & human_turn & hmove_noop
    human_action = holding_l1 & human_turn #& hmove_noop
    prime_turn_str = '1' # robot turn
    prime_rConf_str = '011' # holding l1
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = rconf_cube_to_tr(transition_relation=transition_relation,
                                           rConf_prime_str=prime_rConf_str,
                                           tr_cube=human_action,
                                           pVars=pVars)

    ############################## HUMAN MOVES ##############################    
    # Human move b1 l3 
    # (Env) (b1 l2) ---- (hmove b1 l3) ----> (Sys) (b1 l3)
    human_action = human_turn & hmove_b1_l3 & l3_empty & (b1_l1 | b1_l2 | b1_l3)
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b1 = '100' # b1 l3
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    

    # using (b1_l1 | b1_l2 | b1_l3) reduces the size of the TR as ~b1_l0 includes other invalid states as well
    human_action = human_turn & hmove_b1_l2 & l2_empty & (b1_l1 | b1_l2 | b1_l3)
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b1 = '011' # b1 l2
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    human_action = human_turn & hmove_b1_l1 & l1_empty & (b1_l1 | b1_l2 | b1_l3)
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b1 = '010' # b1 l1
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    
    human_action = human_turn & hmove_b0_l3 & l3_empty & ~b0_l0
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b0 = '100' # b0 l3
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    

    human_action = human_turn & hmove_b0_l2 & l2_empty & ~b0_l0
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b0 = '011' # b0 l2
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    human_action = human_turn & hmove_b0_l1 & l1_empty & ~b0_l0
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b0 = '010' # b0 l1
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])

    
    ############################## HUMAN FRAME AXIOMS ##############################
    # add human noop for testng compositional construction
    # (b0 l0) ---- (hmove noop) ---> (b0 l0)
    human_action = b0_l0 & human_turn #& hmove_noop
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b0 = '001' # b0 l0
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    human_action = b1_l0 & human_turn #& hmove_noop
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b1 = '001' # b1 l0
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    #  (b1 l2) ---- (hmove noop) ---> (b1 l2)
    human_action = b1_l2 & human_turn & hmove_noop
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b1 = '011' # b1 l2
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    

    #  (b1 l3) ---- (hmove noop) ---> (b1 l3)
    human_action = b1_l3 & human_turn & hmove_noop
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b1 = '100' # b1 l3
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    

    #  (b0 l2) ---- (hmove noop) ---> (b0 l2)
    human_action = b0_l2 & human_turn & hmove_noop
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b0 = '011' # b0 l2
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    

    # (b0 l3) ---- (hmove noop) ---> (b0 l3)
    human_action = b0_l3 & human_turn & hmove_noop
    prime_turn_str = '1' # robot turn
    prime_bConf_str_b0 = '100' # b0 l3
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=human_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b0,
                                           bidx=0, tr_cube=human_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    

    ############################## ROBOT FRAME AXIOMS ##############################
    # let try adding frame axiom for the robot move
    # robot_turn & ~to_obj b1 & ~grasp & b1 l3 ----> (b1 l3)
    robot_action = robot_turn & to_obj_b0 & b1_l3
    prime_turn_str = '0' # human turn
    prime_bConf_str_b1 = '100' # b1 l3
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=robot_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=robot_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])

    robot_action = robot_turn & to_obj_b0 & b1_l2
    prime_turn_str = '0' # human turn
    prime_bConf_str_b1 = '011' # b1 l2
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=robot_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=robot_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    robot_action = robot_turn & to_obj_b0 & b1_l1
    prime_turn_str = '0' # human turn
    prime_bConf_str_b1 = '010' # b1 l1
    # transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
    #                                        tConf_prime_str=prime_turn_str,
    #                                        tr_cube=robot_action,
    #                                        tVars=[t])
    transition_relation = bconf_cube_to_tr(transition_relation=transition_relation,
                                           bConf_prime_str=prime_bConf_str_b1,
                                           bidx=1, tr_cube=robot_action,
                                           bVars=[[b00, b01, b02], [b10, b11, b12]])
    
    robot_action = robot_turn
    prime_turn_str = '0' # human turn
    transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
                                           tConf_prime_str=prime_turn_str,
                                           tr_cube=robot_action,
                                           tVars=[t])
    
    robot_action = human_turn
    prime_turn_str = '1' # robot turn
    transition_relation = tconf_cube_to_tr(transition_relation=transition_relation,
                                           tConf_prime_str=prime_turn_str,
                                           tr_cube=robot_action,
                                           tVars=[t])


    # test pre-image computation
    goal_cube = human_turn & holding_l1 & b0_l0 & b1_l3

    # creae box conf
    # care_set = b0_l0 | b0_l1 | b0_l2 | b0_l3 | b1_l0 | b1_l1 | b1_l2 | b1_l3
    # dont_care_set = ~(care_set)
    # tr_restricted = [e.constrain(care_set) for e in transition_relation.values()]
    # ts_action = tr_restricted
    ts_action = list(transition_relation.values())

    print("Goal cube: ", goal_cube)
    preimage = preimage_test(From=goal_cube,
                             latches=[t] + pVars + bVars,
                             prime_latches=[prime_t] + prime_pVars + prime_bVars,
                             ts_action=ts_action)
    print("Preimage: ", preimage)




if __name__ == "__main__":
    test_dynamic_franka_world()
    sys.exit(0)
    
    # setting things up
    boxes = 3
    locs = 4
    init = ['ready l3', 'b0 l2']
    goal = ['b0 l1']
    fw_tb = FrankaWorldDyanmicTurnBased(boxes=boxes, locs=locs, init=init, goal=goal, human_locs=range(1, locs + 1))

    print('****************xVars map:****************')
    for k, v in fw_tb.xVar_map.items():
        print(f"{k} : {v}")
    
    print('****************rAction map:****************')
    for k, v in fw_tb.rAction_map.items():
        print(f"{k} : {v}")

    print('****************eAction map:****************')
    for k, v in fw_tb.eAction_map.items():
        print(f"{k} : {v}")


    print("Total num of latches: ", len(fw_tb.latches))
    print("Total num of prime latches: ", len(fw_tb.prime_latches))
    print("Total boolean vars: ", len(fw_tb.latches) + len(fw_tb.prime_latches))

    fw_tb.create_transition_relation()

    fw_tb.test_pre_image()