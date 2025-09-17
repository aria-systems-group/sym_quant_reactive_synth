import math
import warnings

from typing import List, Tuple
from collections import defaultdict

from bidict import bidict
from cudd import Cudd, ADD, BDD

class AddGridWorld:

    def __init__(self, rows: int, columns: int, init: tuple, goal: tuple):
        self.rows = rows
        self.columns = columns
        self.robot_actions: set = frozenset(['east', 'west', 'north', 'south'])
        self.env_actions: set = frozenset(['north-east', 'north-west', 'south-east', 'south-west'])
        self.init = init if (0 < init[0] < self.rows and 0 < init[1] < self.columns) else warnings.warn("Make sure init state in within the bounds of the gridworld.")
        self.goal = goal if (0 < goal[0] < self.rows and 0 < goal[1] < self.columns) else warnings.warn("Make sure goal state in within the bounds of the gridworld.")
        self.manager: Cudd = Cudd()
        self.iVars: List[ADD] = self.create_input_cube()
        self.oVars: List[ADD] = self.create_output_cube()
        self.xVars, self.yVars = self.create_latches()
        self.xVars_bdd: List[BDD] = [var.bddPattern() for var in self.xVars]
        self.yVars_bdd: List[BDD] = [var.bddPattern() for var in self.yVars]
        self.latches  = self.xVars + self.yVars

        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())

        # creat var maps for rows and column vars
        self.xVar_map = defaultdict(lambda: None)
        self.yVar_map = defaultdict(lambda: None)
        self.rAction_map = defaultdict(lambda: None)
        self.eAction_map = defaultdict(lambda: None)

        self.create_xVar_map()
        self.create_yVar_map()

        self.init_latch: ADD = self.cube_to_add(self.xVar_map[init[0]], self.xVars) & self.cube_to_add(self.yVar_map[init[1]], self.yVars) 
        self.goal_latch: ADD = self.cube_to_add(self.xVar_map[goal[0]], self.xVars) & self.cube_to_add(self.yVar_map[goal[1]], self.yVars)

        # create mapping for robot and env actions
        self.create_action_map()

        # create a list that will hold the funcitonal ADDs.
        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
    
    def create_input_cube(self) -> List[ADD]:
        varsize = self.manager.size()
        iVars: List[ADD] =  [self.manager.addVar(k + varsize , 'i' + str(k)) for k in range(2)]
        return iVars
    
    def create_output_cube(self) -> List[ADD]:
        varsize = self.manager.size()
        oVars: List[ADD] =  [self.manager.addVar(k + varsize, 'o' + str(k)) for k in range(2)]
        return oVars
    
    def create_latches(self):
        """
         Given a gridworld of size n x m create log(n) x vars and log(m) y vars that represent the x and y position respectively. 
        """
        x_size, y_size = math.ceil(math.log2(self.rows)), math.ceil(math.log2(self.columns))
        varsize = self.manager.size()
        xVars: List[ADD] = [self.manager.addVar(k + varsize, 'x' + str(k)) for k in range(x_size)]
        varsize = self.manager.size()
        yVars: List[ADD] = [self.manager.addVar(k + varsize, 'y' + str(k)) for k in range(y_size)]

        return xVars, yVars


    def get_valid_robot_transitions(self, rPos: int, cPos: int) -> List[str]:
        """
         Returns a list valid agent actions you can take given  row and column position
        """
        valid_actions = set()
        if rPos + 1 < self.rows:
            valid_actions.add('south')
        if rPos - 1 >= 0:
            valid_actions.add('north')
        if cPos + 1 < self.columns:
            valid_actions.add('east')
        if cPos - 1 >= 0:
            valid_actions.add('west')
        
        return valid_actions
    
    def get_valid_env_transitions(self, rPos: int, cPos: int, robot_act: str) -> List[str]:
        """
         Returns a list valid env actions you can take given row and column position
        """
        valid_actions = set()
        if robot_act == 'north' and cPos + 1 < self.columns:
            valid_actions.add('north-east')
        if robot_act == 'south'and cPos - 1 >= 0:
            valid_actions.add('south-west')
        if robot_act == 'east' and rPos + 1 < self.rows:
            valid_actions.add('south-east')
        if robot_act == 'west' and rPos - 1 >= 0:
            valid_actions.add('north-west')
        
        return valid_actions

    def get_next_state(self, rPos: int, cPos: int, act: str) -> Tuple[int, int]:
        if act == 'north-east':
            return rPos - 1, cPos + 1
        elif act == "north-west":
            return rPos - 1, cPos - 1
        elif act == "south-east":
            return rPos + 1, cPos + 1
        elif act == "south-west":
            return rPos + 1, cPos - 1
    
    def create_xVar_map(self) -> None:
        for r in range(self.rows):
            bit_str = f"{r:0{len(self.xVars)}b}"
            self.xVar_map[r] = bit_str
    

    def create_yVar_map(self) -> None:
        for c in range(self.columns):
            bit_str = f"{c:0{len(self.yVars)}b}"
            self.yVar_map[c] = bit_str
    
    def create_action_map(self) -> None:
        for ridx, ract in enumerate(self.robot_actions):
            rbit_str = f"{ridx:0{len(self.oVars)}b}" 
            self.rAction_map[ract] = rbit_str
        
        for eidx, eact in enumerate(self.env_actions):
            ebit_str = f"{eidx:0{len(self.iVars)}b}"
            self.eAction_map[eact] = ebit_str

    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add
    

    def create_transition_relation(self):
        """
         Create the transition Relation. TR: l x i x o x l' -> 1 for valid transitions.
        """
        for r in range(self.rows):
            rVar_add: ADD = self.cube_to_add(self.xVar_map[r], self.xVars)

            for c in range(self.columns):
                cVar_add: ADD = self.cube_to_add(self.yVar_map[c], self.yVars)
                
                # get valid robot acts for grid position (r, c)
                valid_robot_actions = self.get_valid_robot_transitions(rPos=r, cPos=c)
                for rAct in valid_robot_actions:
                    # get valied env actions for every robot act and gridworld position (r,c)
                    valid_env_actions = self.get_valid_env_transitions(rPos=r, cPos=c, robot_act=rAct)
                    rAct_cube: str = self.rAction_map[rAct]
                    for eAct in valid_env_actions:
                        # create the transition relation
                        next_rPos, next_cPos = self.get_next_state(rPos=r, cPos=c, act=eAct)
                        
                        eAct_cube: str = self.eAction_map[eAct]
                        # if next state var is positive 
                        for idx, prime_rVar in enumerate(self.xVar_map[next_rPos]):
                            if prime_rVar == '1':
                                self.transition_relation[self.xVars_bdd[idx].__str__()] |= rVar_add & self.cube_to_add(rAct_cube, self.oVars) & self.cube_to_add(eAct_cube, self.iVars)
                        
                        for idx, prime_cVar in enumerate(self.yVar_map[next_cPos]):
                            if prime_cVar == '1':
                                self.transition_relation[self.yVars_bdd[idx].__str__()] |= cVar_add & self.cube_to_add(rAct_cube, self.oVars) & self.cube_to_add(eAct_cube, self.iVars)
    
    
    def preimage(self, From: BDD) -> BDD:
        return From.vectorCompose(self.latches, self.transition_relation)
    
    def solve(self):
        """
        Given a goal state, compute the optimal winning strategy that ensures reaching goal for all possible non-determinism.

        # initialize winning state = self.goals
        # while True:
            compute preimage using vectorCompose()
            # take max over env actions
            # take min over sys actions
            # add the state to winning region
            # if fixpoint then break
        """
        goal = self.goal.ite(self.manager.addZero(), self.manager.plusInfinity())
        self.winning_states[0] |= self.winning_states[0].min(goal)
        layer = 0
        while True:
            if self.winning_states[layer].compare(self.winning_states[layer - 1]):
                break

            # if print_layers:
            print(f"**************************Layer: {layer}**************************")
            # compute preimage
            pre: BDD = self.preimage(From=self.winning_states[layer])


            # take max over env actions
            # take min over sys actions
            
            # update the counter
            layer += 1

if __name__ == "__main__":
    game = AddGridWorld(rows=2, columns=2, init = (0, 0), goal=(1, 1))
    game.create_transition_relation()

    for var, f in game.transition_relation.items():
        print(f"f_{var}: \n {f}")






