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
        self.init = init if (0 < init[0] < self.rows and 0 < init[1] < self.columns) else warnings.warn("Make sure init state in within the bounds of the gridworld.")
        self.goal = goal if (0 < goal[0] < self.rows and 0 < goal[1] < self.columns) else warnings.warn("Make sure goal state in within the bounds of the gridworld.")
        self.manager: Cudd = Cudd()
        self.iVars: List[ADD] = self.create_input_cube()
        self.oVars: List[ADD] = self.create_output_cube()
        self.xVars, self.yVars = self.create_latches()
        self.latches  = self.xVars + self.yVars

        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())

        # creat var maps for rows and column vars
        self.xVar_map = defaultdict(lambda: None)
        self.yVar_map = defaultdict(lambda: None)

        self.init_latch = None
        self.goal_latch = None

        # create mapping for robot and env actions
        self.raction_map = bidict({act: self.oVars[idx // 2] if idx % 2 == 0 else ~self.oVars[idx // 2] for idx, act in enumerate(['east', 'west', 'north', 'south'])})
        self.eaction_map = bidict({act: self.iVars[idx // 2] if idx % 2 == 0 else ~self.iVars[idx // 2] for idx, act in enumerate(['north-east', 'north-west', 'south-east', 'south-west'])})

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


    def create_transition_relation(self):
        """
         Create the transition Relation. TR: l x i x o x l' -> 1 for valid transitions.
        """
        for r in range(self.rows):
            # get the booleans variable 
            xVar_idx, x_sign = divmod(r, 2)
            rVar: ADD = self.xVars[xVar_idx] if x_sign == 0 else ~self.xVars[xVar_idx]
            for c in range(self.columns):
                yVar_idx, y_sign = divmod(c, 2)
                cVar: ADD = self.yVars[yVar_idx] if y_sign == 0 else ~self.yVars[yVar_idx]
                
                # grid position (r, c)
                valid_robot_actions = self.get_valid_robot_transitions(rPos=r, cPos=c)
                for rAct in valid_robot_actions:
                    # for every robot act and gridworld position
                    valid_env_actions = self.get_valid_env_transitions(rPos=r, cPos=c, robot_act=rAct)
                    for eAct in valid_env_actions:
                        # create the transition relation
                        next_rPos, next_cPos = self.get_next_state(rPos=r, cPos=c, act=eAct)
                        xVar_idx, x_sign = divmod(next_rPos, 2)
                        yVar_idx, y_sign = divmod(next_cPos, 2)
                        prime_rVar: ADD = self.xVars[xVar_idx] if x_sign == 0 else ~self.xVars[xVar_idx]
                        prime_cVar: ADD = self.yVars[yVar_idx] if y_sign == 0 else ~self.yVars[yVar_idx]
                        
                        # if next state var is positive 
                        if x_sign == 0:
                            self.transition_relation[prime_rVar.bddPattern().__str__()] |= rVar & self.raction_map[rAct] & self.eaction_map[eAct]
                        
                        if y_sign == 0:
                            self.transition_relation[prime_cVar.bddPattern().__str__()] |= cVar & self.raction_map[rAct] & self.eaction_map[eAct]
    
    
    def preimage(self, From: BDD) -> BDD:
        return From.vectorComposr(self.latches, self.transition_relation)
    
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
    game = AddGridWorld(rows=5, columns=10)
    game.create_transition_relation()

    for var, f in game.transition_relation.items():
        print(f"Var: {var} and f: {f}")






