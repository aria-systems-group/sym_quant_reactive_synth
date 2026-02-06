import sys
import math
import time

from functools import reduce
from typing import List, Tuple, Union

from src.compositional_graphs.test_frankadynamic_ratio_tb import FrankaWorldDynamicRatioTurnBased

from cudd import Cudd, ADD, BDD


class FrankaWorldDynamicRatioTurnBasedElse(FrankaWorldDynamicRatioTurnBased):
    """
     This class inherits from FrankaWorldDyanmicRatioTurnBased aand makes the following change:
     In TR, when the human moves a box, the robot transit to an "else" state rather than going to original state.
    """
    def __init__(self, boxes: int, locs: int, ratio: int, init: tuple, goal: tuple, restricted_human_locs: List[int], restricted_human_boxes: List[int], enable_reordering: bool = False, only_reachable_states: bool = False):
        super().__init__(boxes, locs, ratio, init, goal, restricted_human_locs, restricted_human_boxes, enable_reordering, only_reachable_states)
    

    def create_ready_holding_to_obj_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        # num. of preds = ready x |locs + 1| + to-obj x |boxes| + holding x |locs + 1| + 1 (to account for l0 being end effector loc)
        # additional preds: in-transit x |boxes| (in-transit box) + in-transfer x |locs| (in-transfer to_loc)
        num_of_preds = 2*self.locs + self.boxes + 2 + 1 # +2 for reasy else and holding else
        num_of_preds += self.boxes # in-transit preds
        num_of_preds += self.locs # in-transfer preds
        vars_size: int = math.ceil(math.log2(num_of_preds))
        vars_size = vars_size + 1 if pow(2, vars_size) == num_of_preds else vars_size
        Vars: List[ADD] = [self.manager.addVar(k + varsize, 'p' + str(k)) for k in range(vars_size)]
        return Vars
    

    def create_xVar_map(self):
        # for misc preds ready and holding we create all locs.
        offset = 0 # 
        for pred in ['ready', 'holding']:
            # start from 1 as l0 is reserved for end-effector location and locs +  1 is reserved for else location
            for loc in range(1, self.locs + 2):
                bit_str = f"{offset + loc:0{len(self.pVars)}b}"
                self.xVar_map[pred + ' l' + str(loc)] = bit_str
                self.pVar_map[pred + ' l' + str(loc)] = bit_str
            offset += self.locs + 1
        
        offset = 2*(self.locs + 1) + 1 # +2 (for ready else and holdinbg else) +1 for the offset from the 0-vector
        # for misc pred to-obj we create all boxes
        for b in range(self.boxes):
            bit_str = f"{b + offset:0{len(self.pVars)}b}"
            self.xVar_map['to-obj b' + str(b)] = bit_str
            self.pVar_map['to-obj b' + str(b)] = bit_str
        
        # create preds for in-transit
        offset += self.boxes
        for b in range(self.boxes):
            bit_str = f"{offset + b:0{len(self.pVars)}b}"
            self.xVar_map[f'in-transit b{b}'] = bit_str
            self.pVar_map[f'in-transit b{b}'] = bit_str
        
        offset += self.boxes
        # create preds for in-transfer
        for to_loc in range(1, self.locs + 1):
            bit_str = f"{offset + to_loc:0{len(self.pVars)}b}"
            self.xVar_map[f'in-transfer l{to_loc}'] = bit_str
            self.pVar_map[f'in-transfer l{to_loc}'] = bit_str

        # for each boxes we create |locs| boolean vars
        for b in range(self.boxes):
            for l in range(self.locs + 1):
                bit_str = f"{l + 1:0{len(self.bVars[b])}b}"
                self.xVar_map['b' + str(b) + ' l' + str(l)] = bit_str
                self.bVars_map[b]['b' + str(b) + ' l' + str(l)] = bit_str
    
    def get_number_of_states(self, verbose: bool = True) -> Tuple[int, int]:
        """
         A method to to compute the |Sys States| and |Env states| in the game.
         Sys States = Robot Configurations (ready, holding, to-obj) x Box Configurations x |turn variables|
         Env States = Robot Configurations (in-transit, in-transfer) x Box Configurations x |turn variables|

         Box conf. = (|locs + 1|)! / (|locs + 1| - |boxes|)! (locs = locations; +1 for end-effector loc)
         |ready| = |locs|; |holding| = |locs|; |to-obj| = |boxes| + 1 (for the 0-offset)
         |in-transit| = |boxes|; |in-transfer| = |locs|;
        """
        sys_states = (self.ratio + 1)*(2*self.locs + self.boxes + 1)*(math.factorial(self.locs + 1) // math.factorial(self.locs + 1 - self.boxes))
        env_states = (self.ratio + 1)*((self.boxes + self.locs))*(math.factorial(self.locs + 1) // math.factorial(self.locs + 1 - self.boxes))
        if verbose:
            print(f'Number of States in Game: \n Sys States: {sys_states:,} \n Env States: {env_states:,} \n Total States: {sys_states + env_states:,}')
        return sys_states, env_states
    

    def create_valid_state_constraints(self):
        """
         A method to create valid state constraints that capture the relationship between robot configuration and box configurations. 
         As the robot configuration update rules are changed to go to an "else" location rather than going to original location, 
         we need to change the valid state constraints accordingly.

         Specifically, I need to update constraints related to robot configuration. 
        """
        # now lets add constraints that is rConf is ready then no box is at ee-location
        for b in range(self.boxes):
            for at_loc in range(1, self.locs + 2):
                valid_rConf_for_grasp_cube: ADD = self.xVar_map_sym[f'ready l{at_loc}'] | self.xVar_map_sym[f'in-transit b{b}'] | self.xVar_map_sym[f'to-obj b{b}'] 
                self.monolithic_relevant_box_preds &= valid_rConf_for_grasp_cube.ite(self.ee_empty_cube, self.manager.addOne())
        
        # now lets add constraints that is rConf is holding then some box is at ee-location
        some_box_at_ee: ADD = reduce(lambda x, y: x | y, [self.xVar_map_sym[f'b{b} l0'] for b in range(self.boxes)])
        for to_loc in range(1, self.locs + 1):
            self.monolithic_relevant_box_preds &= (self.xVar_map_sym[f'in-transfer l{to_loc}'] | self.xVar_map_sym[f'holding l{to_loc}']).ite(some_box_at_ee, self.manager.addOne())
        
        # map all invalid Rconf to zero
        valid_rconf: ADD = reduce(lambda x, y: x | y, [self.cube_to_add(i, self.pVars) for i in self.pVar_map.values()])
        self.monolithic_relevant_box_preds = valid_rconf.ite(self.monolithic_relevant_box_preds, self.manager.addZero())

    
    def preprocess_monolithic_valid_state_robot_actions(self):
        # for in-transit and in-transfer preds, just addOne()
        for b in range(self.boxes):
            self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'in-transit b{b}']).ite(self.manager.addOne(), self.manager.addZero())
        
        for to_loc in range(1, self.locs + 1):
            self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'in-transfer l{to_loc}']).ite(self.manager.addOne(), self.manager.addZero())
        
        for loc in range(1, self.locs + 2):
            self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'holding l{loc}']).ite(self.manager.addOne(), self.manager.addZero())
            self.monolithic_valid_state_robot_actions |= (self.tVar_map_sym['human'] & self.xVar_map_sym[f'ready l{loc}']).ite(self.manager.addOne(), self.manager.addZero())
    

    def create_sym_weight_dict(self):
        """
         Ovverride base class method. Here the weights are associated with states rather than actions. We assign all states
           where the robot is not at `else` location (ready else; holding else) a weight of 1, and states where the robot
           is at `else` location a weight of 0. For human states, the weight is always 0.
        """
        # initialize a weight ADD that assigns cost to each robot state
        self.weight = self.manager.addZero()
        for rConf in self.pVar_map.keys():
            if rConf != f'ready l{self.locs + 1}' and rConf != f'holding l{self.locs + 1}':
                # if rConf in self.init:
                #     continue
                self.weight |= self.tVar_map_sym['robot'] & self.xVar_map_sym[rConf]
    
    def get_states_per_cost(self):
        """
         A helper function that takes in the ADD weight abd return a vector of 0-1 BDD per cost.
        """
        min_val: int = 0
        max_val: int = 1
        relevant_box_preds_bdd: BDD = self.monolithic_relevant_box_preds.bddPattern()
        
        for val in range(min_val, max_val + 1, 1):
            self.states_per_cost[val] |= self.weight.bddInterval(val, val) & relevant_box_preds_bdd & ~self.goal_latch.bddPattern()
        
        self.states_per_cost[0] |= self.goal_latch.bddPattern()
    
    def create_transit_actions(self):
        """
         The update rule for transit action is changed here. When the robot transits from ready to in-transit,
         the robot goes to an "else" location rather than going to the original location. The in-transit predicate is now indepedent of the from_loc.
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

                    pred_clause_prime_string = self.xVar_map[f"in-transit b{b}"]
                    
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    # create s a_s s' transitions
                    prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym[f"in-transit b{b}"] & self.prime_xVar_map_sym[curr_box_pred]
                    self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())
    
    def create_transfer_actions(self):
        """
         The update rule for transfer action is changed here. When the robot transfers from holding to in-transfer,
         the robot goes to an "else" location rather than going to the original location. The in-transfer predicate is now indepedent of the from_loc.
        """
        turn_bit = self.tVar_map_sym['robot'] 
        for b in range(self.boxes):
            curr_box_pred = f'b{b} l0'
            bConf_cube = self.xVar_map_sym[curr_box_pred]
            bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l0', bConf_cube=bConf_cube) & self.monolithic_relevant_box_preds
        
            for from_loc in range(1, self.locs + 2):
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
                    pred_clause_prime_string = self.xVar_map[f'in-transfer l{to_loc}']
                    
                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    box_clause_prime_string = self.xVar_map[curr_box_pred]
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= robot_transition_cube
                    
                    # create s a_s s' transitions
                    prime_state_cube: ADD = self.prime_tVar_map_sym['human'] & self.prime_xVar_map_sym[f'in-transfer l{to_loc}'] & self.prime_xVar_map_sym[curr_box_pred]
                    self.monolithic_valid_state_robot_actions_prime_state |= robot_transition_cube.ite(prime_state_cube, self.manager.addZero())
    

    def create_human_move_transit(self) -> None:
        """
         From the in-transit states, if human moves a box, the robot conf and the box conf. change. 
          If the human does not move a box, the robot conf. evolves to to-obj box and box conf. remain the same for all the boxes. 
          If the human does move a box, then robot conf. evolves to ready at else state
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for b in range(self.boxes):
                # for all in-transit preds
                invalid_hmove_cube = self.manager.addZero()
                rConf_cube = self.xVar_map_sym[f'in-transit b{b}']

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

                        # If the human can still intervene then add it set of valid moves and increment k by 1
                        if (k == 0 or k % self.ratio != 0) and self.ratio != 0:
                            pred_clause_prime_string = self.xVar_map[f'ready l{self.locs + 1}'] # ready else location
                            for sidx, s in enumerate(pred_clause_prime_string):
                                if s == '1':
                                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube 

                            # update k var - increment k by 1
                            kVal_prime_str = self.kVar_map[f'k{k + 1}']
                            for sidx, s in enumerate(kVal_prime_str):
                                if s == '1':
                                    self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |=  hmove_cube
                            
                            # create s a_s s' transitions - human
                            prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'ready l{self.locs + 1}'] & self.prime_kVar_map_sym[f'k{k + 1}']
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
    
    def create_human_move_transfer(self) -> None:
        """
         From the in-transfer states, if human moves a box, the robot conf and the box conf. change. 
          If the human does not move a box, the robot conf. evolves to holding org-destination and box conf. remain the same for all the boxes.
          If the human does move a box, the robot conf. goes back to holding else loc and the box conf. of the moved box changes. 
        """
        turn_bit: ADD = self.tVar_map_sym['human']
        for k in range(self.ratio + 1):
            kVal_cube: ADD = self.kVar_map_sym[f'k{k}']
            for to_loc in range(1, self.locs + 1):
                # for all in-transfer preds
                invalid_hmove_cube = self.manager.addZero()
                rConf_cube = self.xVar_map_sym[f'in-transfer l{to_loc}']

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
                            pred_clause_prime_string = self.xVar_map[f'holding l{self.locs + 1}'] # holding else location
                            for sidx, s in enumerate(pred_clause_prime_string):
                                if s == '1':
                                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= hmove_cube
                            
                            # update k var - increment k by 1
                            kVal_prime_str = self.kVar_map[f'k{k + 1}']
                            for sidx, s in enumerate(kVal_prime_str):
                                if s == '1':
                                    self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |=  hmove_cube
                            
                            # create s a_s s' transitions - human
                            prime_state_cube: ADD = self.prime_tVar_map_sym['robot'] & self.prime_xVar_map_sym[f'holding l{self.locs + 1}'] & self.prime_kVar_map_sym[f'k{k + 1}']
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
    
    
    def get_next_state_human(self, curr_state: List[str], action: str) -> Tuple[ADD, str] :
        """
         A helper function to get the next state under human action given the current state and human action.
         As we changed the robot conf. update rules in human move actions, we need to override this function here. 
         Also, when the human doe not intervene the update rule parse the robot confg. to update it correctly. As the robot conf. 
         (in-transfer and in-transit) were updated, we update the hmove noop case accordingly as well.  
        """
        turn_var_idx = 0
        human_move_idx = 1
        rConf_idx = 2
        box_idx = 3
        action: str = self.check_valid_human_move(curr_state=curr_state, action=action)
        if action.startswith('hmove noop'):
            # boxes do not change location but we need to update robot confguration
            if curr_state[rConf_idx].startswith('in-transit'):
                # the rConf state is of the form in-transit b_idx
                to_box = curr_state[rConf_idx].split(' ')[1]
                curr_state[rConf_idx] = f'to-obj {to_box}'
            elif curr_state[rConf_idx].startswith('in-transfer'):
                # the rConf state is of the form in-transfer to_loc
                to_loc = curr_state[rConf_idx].split(' ')[1]
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
                # the rConf state is of the form in-transit b_idx
                curr_state[rConf_idx] = f'ready l{self.locs + 1}' # ready else location
            elif curr_state[rConf_idx].startswith('in-transfer'):
                # the rConf state is of the form in-transfer to_loc
                curr_state[rConf_idx] = f'holding l{self.locs + 1}' # holding else location
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
          As we changed the robot conf. update rules in transit and transfer actions,
          we need to override this function here.
        """
        turn_var_idx = 0
        human_move_idx = 1
        rConf_idx = 2
        box_idx = 3
        # if action is transit then, update the robot configuration
        if action.startswith('transit'):
            assert curr_state[rConf_idx].startswith('ready'), "Make sure the robot is ready to transit!!!"
            b_idx = action.split(' ')[1]
            # from ready you evolve to in-transit
            curr_state[rConf_idx] = f'in-transit {b_idx}'
        
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
                    print("[INVALID ROBOT ACTION]: No box found at the location where robot is trying to grasp. Cannot proceed!!")
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
            to_loc = action.split(' ')[1]
            curr_state[rConf_idx] = f'in-transfer {to_loc}'

        else:
            print("Unknown action. Cannot compute next state!!")
            sys.exit(-1)

        # update state turn
        curr_state[turn_var_idx] = 'human' if curr_state[turn_var_idx] == 'robot' else 'robot'
        # convert the string of boxes location to sperate state
        split_str = curr_state[box_idx].split(', ')
        return self.tVar_map_sym[curr_state[turn_var_idx]] & self.kVar_map_sym[curr_state[human_move_idx]] &  \
              self.xVar_map_sym[curr_state[rConf_idx]] & reduce(lambda a, b: a & b, [self.xVar_map_sym[s] for s in split_str])


    def preimage_test(self, From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
        From = From.swapVariables(latches, prime_latches)
        return From.vectorCompose(prime_latches, ts_action)


    def test_pre_image(self):
        # convert transition relation to latches bdd
        # goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['ready l4'] & self.xVar_map_sym['b0 l1'] & self.xVar_map_sym['b1 l3']  #& \
        #(self.xVar_map_sym['b1 l4'] | self.xVar_map_sym['b1 l3'])
        goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['ready l3'] & self.xVar_map_sym['b0 l1'] & self.kVar_map_sym['k1']
        # goal state is b0 and l0 and ready l0
        print('Goal state:', goal_cube)
        # From = goal_cube
        preimage = self.preimage_test(From=goal_cube, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('Preimage: ', preimage)
        self.convert_cube_to_state_ADD(preimage, action=False, verbose=True)


if __name__ == "__main__":
    # setting things up
    boxes = 1
    locs = 3
    ratio = 1
    # init = ['ready l2', 'b0 l2', 'b1 l3', 'b2 l4', 'b3 l5']
    # goal = [['b0 l1']]
    # init = ['ready l2', 'b0 l2', 'b1 l3']
    # goal = [['b0 l1', 'b1 l3'], ['b0 l1', 'b1 l4']]
    init = ['ready l2', 'b0 l2']
    goal = [['b0 l1']]
    # goal = ['holding l1', 'b0 l0']
    # goal = [['b0 l1', 'b1 l3'], ['b0 l1', 'b1 l4']]
    human_locs = range(1, locs + 1)
    # human_locs =  [3, 4, 5, 6, 7, 8, 9, 10] #range(1, locs + 1)
    # human_locs = []
    fw_tb = FrankaWorldDynamicRatioTurnBasedElse(boxes=boxes, locs=locs, ratio=ratio, init=init, goal=goal, restricted_human_locs=human_locs)

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

    # fw_tb.test_pre_image()
    # sys.exit(0)

    tic = time.time()
    strategy = fw_tb.solve(verbose=False)
    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic} seconds")

    if strategy is not None:
        fw_tb.roll_out_strategy(strategy=strategy, verbose=True)