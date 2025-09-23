import sys
import math
import warnings

from enum import Enum
from functools import reduce
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

from bidict import bidict
from cudd import Cudd, ADD, BDD

class Moves(Enum):
    NORTH = (-1, 0)
    SOUTH = (1, 0)
    WEST = (0, -1)
    EAST = (0, 1)
    STAY = (0, 0)


class AddGridWorld:

    def __init__(self, rows: int, columns: int, init: tuple, goal: tuple):
        self.rows = rows
        self.columns = columns
        # self.robot_actions: set = frozenset(['EAST', 'WEST', 'NORTH', 'SOUTH', 'STAY'])
        self.robot_actions: List[str] = ['WEST', 'EAST', 'SOUTH', 'NORTH']  # making this a List to keep the order consistent across runs
        self.env_actions: List[str] = ['north-east', 'north-west', 'south-east', 'south-west', 'no-int']
        self.manager: Cudd = Cudd()
        self.iVars: List[ADD] = self.create_input_vars()
        self.oVars: List[ADD] = self.create_output_vars()
        self.xVars, self.yVars = self.create_latches()
        self.xVars_bdd: List[BDD] = [var.bddPattern() for var in self.xVars]
        self.yVars_bdd: List[BDD] = [var.bddPattern() for var in self.yVars]
        self.latches: List[ADD]  = self.xVars + self.yVars
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]

        self.winning_states: ADD = defaultdict(lambda: self.manager.plusInfinity())

        # creat var maps for rows and column vars
        self.xVar_map = bidict({})
        self.yVar_map = bidict({})
        self.rAction_map = bidict({})
        self.eAction_map = bidict({})

        self.create_xVar_map()
        self.create_yVar_map()
        self.set_init_state(init)
        self.set_goal_state(goal)
        self.init_latch: ADD = self.cube_to_add(self.xVar_map[init[0]], self.xVars) & self.cube_to_add(self.yVar_map[init[1]], self.yVars) 
        self.goal_latch: ADD = self.cube_to_add(self.xVar_map[goal[0]], self.xVars) & self.cube_to_add(self.yVar_map[goal[1]], self.yVars)

        # create mapping for robot and env actions
        self.create_action_map()

        self.iVars_cube: ADD = reduce(lambda x, y: x & y, self.iVars)
        self.oVars_cube: ADD = reduce(lambda x, y: x & y, self.oVars)

        # create a list that will hold the funcitonal ADDs.
        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
    

    def set_init_state(self, init: tuple):
        if (0 <= init[0] < self.rows and 0 <= init[1] < self.columns):
            self.init = init
        else:
            warnings.warn("Make sure init state in within the bounds of the gridworld.")
            sys.exit()
    
    def set_goal_state(self, goal: tuple):
        if (0 <= goal[0] < self.rows and 0 <= goal[1] < self.columns):
            self.goal = goal
        else:
            warnings.warn("Make sure Goal state in within the bounds of the gridworld.")
            sys.exit()

    
    def create_input_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        # We not create the no-int action
        iVars_size = math.ceil(math.log2(len(self.env_actions))) - 1
        iVars: List[ADD] =  [self.manager.addVar(k + varsize , 'i' + str(k)) for k in range(iVars_size)]
        # manually create a add var for no-int
        # iVars += [self.manager.addVar(self.manager.size(), 'i'+ str(iVars_size + 1))]
        return iVars
    
    def create_output_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        oVars_size = math.ceil(math.log2(len(self.robot_actions)))
        oVars: List[ADD] =  [self.manager.addVar(k + varsize, 'o' + str(k)) for k in range(oVars_size)]
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
        # sys can always choose to stay
        # valid_actions = set({'STAY'})
        valid_actions = set({})
        if rPos + 1 < self.rows:
            valid_actions.add('SOUTH')
        if rPos - 1 >= 0:
            valid_actions.add('NORTH')
        if cPos + 1 < self.columns:
            valid_actions.add('EAST')
        if cPos - 1 >= 0:
            valid_actions.add('WEST')
        
        return valid_actions
    
    def get_valid_env_transitions(self, rPos: int, cPos: int, robot_act: str) -> List[str]:
        """
         Returns a list valid env actions you can take given row and column position
        """
        # env can always choose to not intervene
        # valid_actions = set({'no-int'})
        valid_actions = set({})
        if robot_act == 'NORTH' and cPos + 1 < self.columns:
            valid_actions.add('north-east')
        if robot_act == 'SOUTH'and cPos - 1 >= 0:
            valid_actions.add('south-west')
        if robot_act == 'EAST' and rPos + 1 < self.rows:
            valid_actions.add('south-east')
        if robot_act == 'WEST' and rPos - 1 >= 0:
            valid_actions.add('north-west')
        
        return list(valid_actions)

    def get_next_state(self, rPos: int, cPos: int, eAct: str, rAct: str) -> Tuple[int, int]:
        assert eAct in self.env_actions, "Make sure the action is valid"
        if eAct == 'north-east':
            return rPos - 1, cPos + 1
        elif eAct == "north-west":
            return rPos - 1, cPos - 1
        elif eAct == "south-east":
            return rPos + 1, cPos + 1
        elif eAct == "south-west":
            return rPos + 1, cPos - 1
        elif eAct == "no-int":
            return rPos + Moves[rAct].value[0], cPos + Moves[rAct].value[1]
    
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
            # skip the no-int action
            if eact != "no-int": 
                # ebit_str = f"{eidx:0{len(self.iVars) - 1}b}"
                # the last bit is always 0 to represent no-int
                # self.eAction_map[eact] = ebit_str + '0'
                ebit_str = f"{eidx:0{len(self.iVars)}b}"
                self.eAction_map[eact] = ebit_str

    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add

    def get_no_int_cube(self, valid_env_actions: List[str]) -> ADD:
        """
         Given a set of valid human moves, return the a cube that represents no intervention.

         If valid env move = {south-west} then no intervention = ~(south-west)
         If valid env move = {north-east, south-west} then no intervention = ~(north-east | south-west)
         If valid env move = {} then no intervention = 1
        """
        valid_env_cube_strings = [self.eAction_map[i] for i in valid_env_actions if i != "no-int"]
        no_human_int = self.manager.addOne()
        if len(valid_env_cube_strings) > 0:
            valid_env_cubes = [self.cube_to_add(i, self.iVars) for i in valid_env_cube_strings]
            no_human_int = ~reduce(lambda x, y: x | y, valid_env_cubes)
        return no_human_int

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
                    valid_env_actions: List[str] = self.get_valid_env_transitions(rPos=r, cPos=c, robot_act=rAct)
                    rAct_cube: str = self.rAction_map[rAct]
                    # no_human_int: ADD = self.get_no_int_cube(valid_env_actions=valid_env_actions)
                    new_tmp_list = valid_env_actions + ['no-int']
                    for eAct in new_tmp_list:
                        assert len(valid_env_actions) <= 1 , "Make sure the valid env actions are either no-int or one intervention"
                        if eAct == 'no-int':
                            # eAct_cube: ADD = self.get_no_int_cube(valid_env_actions=[eAct])
                            if len(valid_env_actions) == 1:
                                eAct_cube_string: str = self.eAction_map.get(valid_env_actions[0], None)
                                eAct_cube: ADD = ~self.cube_to_add(eAct_cube_string, self.iVars)
                            else:
                                eAct_cube = self.manager.addOne()
                        else:
                            eAct_cube_string: str = self.eAction_map.get(eAct, None)
                            eAct_cube: ADD = self.cube_to_add(eAct_cube_string, self.iVars)
                        
                        # create the transition relation
                        next_rPos, next_cPos = self.get_next_state(rPos=r, cPos=c, eAct=eAct, rAct=rAct)
                        
                        # if next state var is positive 
                        for idx, prime_rVar in enumerate(self.xVar_map[next_rPos]):
                            if prime_rVar == '1':
                                self.transition_relation[self.xVars_bdd[idx].__str__()] |= rVar_add & cVar_add & self.cube_to_add(rAct_cube, self.oVars) & eAct_cube
                        
                        for idx, prime_cVar in enumerate(self.yVar_map[next_cPos]):
                            if prime_cVar == '1':
                                self.transition_relation[self.yVars_bdd[idx].__str__()] |= cVar_add & rVar_add & self.cube_to_add(rAct_cube, self.oVars) & eAct_cube
    

    def get_buckets_of_BDD(self, max_interval_val: int, layer: int) -> Dict[int, BDD]:
        _win_state_bucket: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval in range(max_interval_val + 1):
            # get the states with state value equal to sval and store them in their respective bukcets
            win_sval: BDD = self.winning_states[layer].bddInterval(sval, sval)
            
            if not win_sval.isZero():
                _win_state_bucket[sval] |= win_sval
        
        return _win_state_bucket

    
    def preimage(self, ts_action: List[BDD], From: BDD) -> BDD:
        return From.vectorCompose(self.latches_bdd, ts_action)
    

    def roll_out(self, strategy: ADD):
        """
         Give a strategy ADD, roll it out from the initial state until you reach the goal state.
        """
        curr_state: ADD = self.init_latch
        max_steps: int = max(self.winning_states.keys())
        
        while (self.goal_latch & curr_state).isZero():
            # get the current position from the ADD
            curr_state_cube_str = curr_state.bddInterval(1, 1).cubeString()
            curr_state_cube_str_support: str = [a for a in curr_state_cube_str if a != '-']
            rpos = self.xVar_map.inv[''.join(curr_state_cube_str_support[:len(self.xVars)])]
            cpos = self.yVar_map.inv[''.join(curr_state_cube_str_support[len(self.xVars):])]
            print(f"Current Position: ({rpos}, {cpos})")
            
            # get the act
            opt_sval =  list((curr_state & self.winning_states[max_steps]).generate_cubes())[0][1]
            act_cube_string: List[int] = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneCube()[2:4]
            
            # extract the relevant cube string and then look up the actual name
            act_cube_string_support = [str(a) for a in act_cube_string if a != '-']
            ract_name = self.rAction_map.inv[''.join(act_cube_string_support)]

            # ask human for Env move input
            # valid_env_moves = self.get_valid_env_transitions(rPos=rpos, cPos=cpos, robot_act=ract_name)
            # valid_env_moves.append('no-int')
            # for idx, hm in enumerate(valid_env_moves):
            #     print(f"Env Move: {idx} : {hm}")
            # hmove = int(input("Enter move: "))

            # next based on action, get the next state
            next_state: tuple = self.get_next_state(rPos=rpos, cPos=cpos, eAct='no-int', rAct=ract_name)
            
            # update current state to be next state and repeat
            curr_state: ADD = self.cube_to_add(self.xVar_map[next_state[0]], self.xVars) & self.cube_to_add(self.yVar_map[next_state[1]], self.yVars)

    
    def solve(self) -> Optional[ADD]:
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
        # strategy - optimal (state & robot-action) pair stored in the ADD
        strategy: ADD  = self.manager.plusInfinity()
        
        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        self.winning_states[0] |= self.winning_states[0].min(goal)
        strategy = strategy.min(goal)
        
        # intialize the iteration counter and hardcode the action cost
        layer = 0
        c_max = 1 # hardcoded for now. Make it general in future
        act_val = 1 # hardcoded action cost for now. Make it general in future

        # preprocess the transition relation to be list of functional BDDs as CUDD's vectorCompsoe only accepts list of functional BDDs
        partitioned_tr_bdd = [tr.bddPattern() for tr in self.transition_relation.values()]
        while True:
            # if oyu reach a fix point then break
            if self.winning_states[layer].compare(self.winning_states[layer - 1], 2):
                print("**************************Reached fixpoint**************************")
                if self.init_latch & self.winning_states[layer] != self.manager.plusInfinity():
                    init_val: int = list((self.init_latch & self.winning_states[layer]).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    return strategy if init_val < math.inf else None
                return None

            # if print_layers:
            print(f"**************************Layer: {layer}**************************")

            # convert the winning states into buckets of BDD
            max_interval_val = layer * c_max
            win_state_bucket: Dict[int, BDD] = self.get_buckets_of_BDD(max_interval_val, layer)

            pre_buckets: Dict[ADD] = defaultdict(lambda: self.manager.addZero())
            for sval, succ_states in win_state_bucket.items():
                pre_states: BDD = self.preimage(ts_action=partitioned_tr_bdd, From=succ_states)

                if not pre_states.isZero():
                    pre_buckets[sval + act_val] |= pre_states.toADD()       

            # unions of all predecessors
            pre_states: ADD = reduce(lambda x, y: x | y, pre_buckets.values())

            # now take univ abstraction to remove edges to states with infinity value
            upre_states: ADD = pre_states.univAbstract(self.iVars_cube)
            tmp_strategy: ADD = upre_states.ite(self.manager.addZero(), self.manager.plusInfinity())

            for sval, apre_s in pre_buckets.items():
                # we skip the zero states
                if sval != 0:
                    tmp_strategy = tmp_strategy.max(apre_s.ite(self.manager.addConst(int(sval)), self.manager.addZero()))
            
            new_tmp_strategy: ADD = tmp_strategy

            # go over all the env actions and preserve the maximum one
            for env_tr_dd in self.eAction_map.values():
                new_tmp_strategy = new_tmp_strategy.max(tmp_strategy.restrict(self.cube_to_add(env_tr_dd, self.iVars)))

            # compute the minimum of state action pairs
            strategy = strategy.min(new_tmp_strategy)

            self.winning_states[layer + 1] |= self.winning_states[layer]

            for robot_tr_dd in self.rAction_map.values():
                # remove the dependency for that action and preserve the minimum value for every state
                self.winning_states[layer + 1] = self.winning_states[layer  + 1].min(strategy.restrict(self.cube_to_add(robot_tr_dd, self.oVars)))

            # update the counter
            layer += 1


def test_things_add():
    m = Cudd()
    # i0 = m.addVar(0, 'i0')
    i = [m.addVar(0 + n, 'i' + str(n)) for  n in range(2)]
    o = [m.addVar(2 + k , 'o' + str(k)) for k in range(2)]
    # x0 = m.addVar(4, 'x0')
    x0 = m.addVar(4, 'x0')
    x1 = m.addVar(5, 'x1')
    y0 = m.addVar(6, 'y0')

    # care region
    # care_region: ADD = (x0 & y0) | (x0 & ~y0) | (~x0 & y0) | (~x0 & ~y0)
    # dont_care_region: ADD = ~care_region
    # print(dont_care_region)

    # creat eempty TR List. 
    tr = [m.addZero(), m.addZero(), m.addZero()]
    robot_north: ADD = o[0] & o[1]
    robot_east: ADD = ~o[0] & o[1]
    robot_west: ADD = ~o[0] & ~o[1]
    robot_south: ADD = o[0] & ~o[1]
    # env_move: ADD = i0
    
    # now lets make it a little more complicated by adding explicit boolean function for various env moves
    env_ne: ADD = ~i[0] & ~i[1]
    env_nw: ADD = ~i[0] & i[1]
    env_se: ADD = i[0] & ~i[1]
    env_sw: ADD = i[0] & i[1]

    # row 0 = ~x0 & x1
    # row 1 = x0 & ~x1 

    # (1, 0) -> N & No Env move (0, 0)
    tr[1] |=  x0 & ~x1 & ~y0 & robot_north & ~env_ne
    
    
    # (1, 0) -> N & Env move (0, 1)
    # tr[1] |= x0 & ~y0 & robot_north & env_move
    # tr[1] |= x0 & ~y0 & robot_north & env_ne
    
    ###########################################
    tr[1] |= x0 & ~x1 & ~y0 & robot_north & env_ne
    tr[2] |= x0 & ~x1 & ~y0 & robot_north & env_ne

    #(1, 0) -> E & !Env move (1, 1)
    # tr[0] |= x0 & ~y0 & robot_east & ~env_move
    # tr[0] |= x0 & ~y0 & robot_east & env_move
    # tr[0] |= x0 & ~y0 & robot_east & ~env_se
    # tr[0] |= x0 & ~y0 & robot_east & env_se
    # tr[0] |= x0 & ~y0 & robot_east

    ###########################################
    tr[0] |= x0 & ~x1 & ~y0 & robot_east

    # copy of the above for the y var
    # tr[1] |= x0 & ~y0 & robot_east & ~env_move
    # tr[1] |= x0 & ~y0 & robot_east & env_move
    
    # tr[1] |= x0 & ~y0 & robot_east & ~env_se
    # tr[1] |= x0 & ~y0 & robot_east & env_se
    # tr[1] |= x0 & ~y0 & robot_east

    ###########################################
    tr[2] |= x0 & ~x1 & ~y0 & robot_east

    # now lets add (1, 1) to (0, 1) 
    # tr[1] |= x0 & y0 & robot_north & ~env_move
    # tr[1] |= x0 & y0 & robot_north & env_move

    # tr[1] |= x0 & y0 & robot_north & ~env_nw
    # tr[1] |= x0 & y0 & robot_north & env_nw

    ###########################################
    tr[1] |= x0 & ~x1 & y0 & robot_north & ~env_nw
    tr[1] |= x0 & ~x1 & y0 & robot_north & env_nw
    tr[2] |= x0 & ~x1 & y0 & robot_north & ~env_nw
    tr[2] |= x0 & ~x1 & y0 & robot_north & env_nw

    # now lets add (1, 1) to (1, 0) - no env move
    # tr[0] |= x0 & y0 & robot_west & ~env_move

    # tr[0] |= x0 & y0 & robot_west & ~env_nw

    ###########################################
    tr[0] |= x0 & ~x1 & y0 & robot_west & ~env_nw

    # now lets add (1, 1) to (0, 0) - if env moves; dont need to add as successort states are 0
    # tr[0] |= x0 & y0 & robot_west & env_move
    tr[1] |= x0 & ~x1 & y0 & robot_west & env_nw

    # now lets add (0, 0) to (0, 1) - no env move
    # tr[1] |= ~x0 & ~y0 & robot_east & ~env_move
    # tr[1] |= ~x0 & ~y0 & robot_east & ~env_se

    ###########################################
    tr[1] |= ~x0 & x1 & ~y0 & robot_east & ~env_se
    tr[2] |= ~x0 & x1 & ~y0 & robot_east & ~env_se

    # now lets add (0, 0) to (1, 1)  env move
    # tr[0] |= ~x0 & ~y0 & robot_east & env_move
    # tr[1] |= ~x0 & ~y0 & robot_east & env_move
    # tr[0] |= ~x0 & ~y0 & robot_east & env_se
    # tr[1] |= ~x0 & ~y0 & robot_east & env_se

    ###########################################
    tr[0] |= ~x0 & x1 & ~y0 & robot_east & env_se
    tr[2] |= ~x0 & x1 & ~y0 & robot_east & env_se

    # finally lets add (0, 0) to (1, 0) - no env move
    # tr[0] |= ~x0 & ~y0 & robot_south
    # tr[0] |= ~x0 & ~y0 & robot_south & env_move

    ###########################################
    tr[0] |= ~x0 & x1 & ~y0 & robot_south

    # lets add (0, 1) to (0, 0) - env and no env move
    # as the succ values are zero, I don't need to add it.
    tr[1] |= ~x0 & x1 & y0 & robot_west

    # lets add (0, 1) to (1, 1) - no env move
    # tr[0] |= ~x0 & y0 & robot_south & ~env_move
    # tr[1] |= ~x0 & y0 & robot_south & ~env_move

    # tr[0] |= ~x0 & y0 & robot_south & ~env_sw
    # tr[1] |= ~x0 & y0 & robot_south & ~env_sw

    ###########################################
    tr[0] |= ~x0 & x1 & y0 & robot_south & ~env_sw
    tr[2] |= ~x0 & x1 & y0 & robot_south & ~env_sw

    # add (0, 1) to (1, 0) - env move
    # tr[0] |= ~x0 & y0 & robot_south & env_move
    # tr[0] |= ~x0 & y0 & robot_south & env_sw

    ###########################################
    tr[0] |= ~x0 & x1 & y0 & robot_south & env_sw


    tr_bdd: List[BDD] = [e.bddPattern() for e in tr]
    
    # restrict youself to states you care about
    # tr_bdd = [e.restrict(care_region.bddPattern()) for e in tr_bdd]

    print('function ADD for x0: ', tr[0])
    print('function ADD for x1: ', tr[1])
    print('function ADD for y0: ', tr[2])

    # compute pre image of (x, y)
    # iter 1: (~x0 & y0)
    # iter 2: (~x0 & y0) | (x0 & y0) ## equiv y0
    # iter 3: (x0 | ~y0) |  (~x0 & y0) | (x0 & y0)
    # From = ( (x0 | ~y0) | (~x0 & y0) ).bddPattern()
    From = (x0 & ~x1 | ~x0 & x1 & y0).bddPattern()
    # From = (x0 | ~y0).bddPattern()
    # From = (y0).bddPattern()
    pre = From.vectorCompose([x0.bddPattern(), x1.bddPattern(), y0.bddPattern()], tr_bdd)
    upre = pre.existAbstract((i[0]& i[1]).bddPattern() & (o[0] & o[1]).bddPattern())
    cpre = pre.univAbstract((i[0]& i[1]).bddPattern())

    # print(f"Pre image of {From.cubeString()[-2:]} : {pre}")
    try: 
        string_form = From.cubeString()[-3:]
    except:
        string_form = From
    print(f"Pre image of {string_form} : {pre}")
    print(f"Existential Pre image of {string_form} : {upre}")
    print(f"Universal Pre image of {string_form} : {cpre.existAbstract((o[0] & o[1]).bddPattern())}")
    print(f"Strategy : {cpre}")



if __name__ == "__main__":
    # test_things_add()
    
    game = AddGridWorld(rows=3, columns=3, init=(0, 0), goal=(1, 2))
    game.create_transition_relation()

    # for var, f in game.transition_relation.items():
    #     print(f"f_{var}: \n {f}")
    
    strategy = game.solve()
    if strategy:
        game.roll_out(strategy=strategy)
    





