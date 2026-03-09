import sys
import math

from enum import Enum
from functools import reduce
from itertools import product

from bidict import bidict
from tabulate import tabulate

from collections import defaultdict
from typing import List, Tuple, Dict, Union

from cudd import Cudd, ADD, BDD, REORDER_GROUP_SIFT_CONV


class Moves(Enum):
    NORTH = (1, 0)
    SOUTH = (-1, 0)
    WEST = (0, -1)
    EAST = (0, 1)
    STAY = (0, 0)


class GridWorldDynamic():
    def __init__(self, rows: int, columns: int, init: List[tuple], goal: tuple, enable_reordering: bool = False):
        self.rows = rows
        self.columns = columns
        self.sys_actions: List[str] = ['STAY', 'NORTH', 'SOUTH', 'EAST', 'WEST']
        self.env_actions: List[str] = ['STAY', 'NORTH', 'SOUTH', 'EAST', 'WEST']
        self.init = init
        self.goal = goal
        self.manager: Cudd = Cudd(maxMem=16000000000)
        # TODO: hard coding should update to be the same the number of agents
        self.xVar_map = {p: bidict({}) for p in range(2)}
        self.yVar_map = {p: bidict({}) for p in range(2)}

        self.tVar_map = bidict({'sys': '1', 'env': '0'})  # fixed turn variable map

        # Predicate to Cube maps - needed for symbolic operations; also avoid multiple calls to cube_to_add()
        self.xVar_map_sym = {p: bidict({}) for p in range(2)}
        self.yVar_map_sym = {p: bidict({}) for p in range(2)}
        self.prime_xVar_map_sym = {p: bidict({}) for p in range(2)}
        self.prime_yVar_map_sym = {p: bidict({}) for p in range(2)}

        # create latches - tVars + kVars + pVars + bVars
        self.create_all_boolean_state_vars_and_maps()
        self.set_latches()

         # create prime latches - prime tVars + prime kVars + prime pVars + prime bVars
        self.create_all_prime_boolean_state_vars_and_maps()
        self.set_prime_latches()

        self.tVar_map_sym = bidict({'sys': self.cube_to_add(self.tVar_map['sys'], self.tVar),
                                    'env': self.cube_to_add(self.tVar_map['env'], self.tVar)})
        
        self.prime_tVar_map_sym = bidict({'sys': self.cube_to_add(self.tVar_map['sys'], self.prime_tVar),
                                          'env': self.cube_to_add(self.tVar_map['env'], self.prime_tVar)})
        

        # now that the maps are initialized we create init and goal states
        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()

        # monolithic transition relation
        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
        self.ts_bdd_transition_fun_list: List[BDD] = []

        # create sys and env action vars and maps
        self.rVars: List[ADD] = self.create_action_vars()
        self.action_map = bidict({})
        self.create_action_map()
        self.action_map_sym = bidict({k: self.cube_to_add(v, self.rVars) for k, v in self.action_map.items()})

        self.rVars_cube = reduce(lambda x, y: x & y, self.rVars)
        self.action_cube_list: List[ADD] = list(self.action_map_sym.values())

        # non-zero weight only for the sys player. Env player does not expend any energy.
        self.weight_dict: Dict[str, int] = {'STAY': 2, 'NORTH': 1, 'SOUTH': 1, 'EAST': 1, 'WEST': 1}
        self.symbolic_weight_dict: Dict[str, ADD] = defaultdict(lambda: self.manager.addOne())
        self.weight = self.manager.addZero()
        self.create_sym_weight_dict(debug=False)

        # for printing 
        self.xVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.xVars]
        self.prime_xVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.prime_xVars]
        self.yVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.yVars]
        self.prime_yVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.prime_yVars]

        # for solver 
        self.env_action_cube_list = []
        self.sys_action_cube_list = []
        for act_str, act_dd in self.action_map_sym.items():
            if act_str.startswith('env'):
                self.env_action_cube_list.append(act_dd)
            else:
                self.sys_action_cube_list.append(act_dd)
        
        self.env_action_cube_list_bdd: List[BDD] = [act_dd.bddPattern() for act_dd in self.env_action_cube_list]
        self.sys_action_cube_list_bdd: List[BDD] = [act_dd.bddPattern() for act_dd in self.sys_action_cube_list]

        if enable_reordering:
            self.manager.autodynEnable()
    

    def create_all_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the FrankaDynamic Turn-Based Game.
          1. turn variables - tVars
          2. column variables - xVars
          3. row variables - yVars
        """
        offset = self.manager.size()
        self.tVar: List[ADD] = [self.manager.addVar(offset, 't0')]
        self.xVars, self.yVars = self.create_latches()
        self.create_xVar_map()
        self.create_yVar_map()
        self.create_symbolic_maps(prime=False)
    

    def create_all_prime_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the FrankaDynamic Turn-Based Game.
          1. turn variables - tVars
          2. row variables - xVars
          3. column variables - yVars
        """
        offset = self.manager.size()
        self.prime_tVar: List[ADD] = [self.manager.addVar(offset, 'pt0')]
        self.prime_xVars, self.prime_yVars = self.create_prime_latches()
        self.create_symbolic_maps(prime=True)
    

    def create_latches(self) -> Tuple[List[ADD], List[ADD]]:
        """
         Given a gridworld of size n x m create log(n) x vars and log(m) y vars that represent the x and y position respectively. 
        """
        x_size, y_size = math.ceil(math.log2(self.rows)), math.ceil(math.log2(self.columns))
        # the x_size and y_size match then, we need to an extra boolean vairables to offset the all 0-vector latch
        if pow(2, x_size) == self.rows and pow(2, y_size) == self.columns:
            # we need creat an additional boolean variable. I will create one addiiotnal variable for x
            x_size += 1
            y_size += 1
        xVars: List[List[ADD]] = []
        for player in range(2):
            varsize = self.manager.size()
            xVars.append([self.manager.addVar(k + varsize, f'x{player}{k}') for k in range(x_size)])
        
        yVars: List[List[ADD]] = []
        for player in range(2):
            varsize = self.manager.size()
            yVars.append([self.manager.addVar(k + varsize, f'y{player}{k}') for k in range(y_size)])

        return xVars, yVars
    

    def create_prime_latches(self) -> Tuple[List[ADD], List[ADD]]:
        """
         Create a copy of prime variables for the latches.
        """
        prime_xVars: List[List[ADD]] = []
        prime_yVars: List[List[ADD]] = []
        for p_idx, p_xVar in enumerate(self.xVars):
            varsize = self.manager.size()
            prime_xVars.append([self.manager.addVar(k + varsize, f'px{p_idx}{k}') for k in range(len(p_xVar))])
        
        for p_idx, p_yVar in enumerate(self.yVars):
            varsize = self.manager.size()
            prime_yVars.append([self.manager.addVar(k + varsize, f'py{p_idx}{k}') for k in range(len(p_yVar))])

        return prime_xVars, prime_yVars


    def create_action_vars(self) -> List[ADD]:
        """
         Create a single method wehre we create robot action and env actions using the same of variables.
         
         This will lead to savings in the # of boolean vars needed. 
         This approach will require log(num_robot_actions + num_env_actions) boolean vars.
        """
        varsize = self.manager.size()
        num_of_rActions =  len(self.sys_actions)
        num_of_eActions = len(self.env_actions)
        rVars_size = math.ceil(math.log2(num_of_rActions + num_of_eActions))
        rVars: List[ADD] =  [self.manager.addVar(r + varsize , 'r' + str(r)) for r in range(rVars_size)]
        return rVars


    def set_latches(self):
        self.latches: List[ADD] = self.tVar + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds]
        self.latches_bdd: List[BDD] = [latch.bddPattern() for latch in self.latches]
    

    def set_prime_latches(self):
        self.prime_latches: List[ADD] = self.prime_tVar + [var for prime_xVar_adds in self.prime_xVars for var in prime_xVar_adds] + [var for prime_yVar_adds in self.prime_yVars for var in prime_yVar_adds]
        self.prime_latches_bdd: List[BDD] = [latch.bddPattern() for latch in self.prime_latches]
    

    def set_init_latch(self) -> ADD:
        # TODO: hard coding; should update to be the same the number of agents
        assert len(self.init) == 2, "[Error]: The init state should be fully defined for the Synthesis code else the Synthesis code will not work correctly."
        init_cube = self.tVar_map_sym['sys']
        for i, (x, y) in enumerate(self.init):
            init_cube &= self.xVar_map_sym[i][x] & self.yVar_map_sym[i][y]
        return init_cube
    

    def set_goal_latch(self) -> ADD:
        goal_cube = self.manager.addOne()
        for x, y in self.goal:
            # TODO: hard coding; should update such that the goal is associated with the Sys player state
            goal_cube &= self.xVar_map_sym[0][x] & self.yVar_map_sym[0][y]
        return goal_cube


    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add
    

    def create_xVar_map(self) -> None:
        for p in range(2):
            for r in range(self.rows):
                # offset is to avoid the 0-vector
                bit_str = f"{r + 1:0{len(self.xVars[p])}b}"
                self.xVar_map[p][r] = bit_str

    def create_yVar_map(self) -> None:
        for p in range(2):
            for c in range(self.columns):
                # offset is to avoid the 0-vector
                bit_str = f"{c + 1:0{len(self.yVars[p])}b}"
                self.yVar_map[p][c] = bit_str

    
    def create_symbolic_maps(self, prime: bool = False):
        """
         Small function to create symbolic maps for the xVar_map, rAction_map and eAction_map
        """
        for player in range(2):
            for k, v in self.xVar_map[player].items():
                if prime:
                    self.prime_xVar_map_sym[player][k] = self.cube_to_add(v, self.prime_xVars[player])
                else:
                    self.xVar_map_sym[player][k] = self.cube_to_add(v, self.xVars[player])
        
        for player in range(2):
            for k, v in self.yVar_map[player].items():
                if prime:
                    self.prime_yVar_map_sym[player][k] = self.cube_to_add(v, self.prime_yVars[player])
                else:
                    self.yVar_map_sym[player][k] = self.cube_to_add(v, self.yVars[player])
    
    
    def create_action_map(self) -> None:
        for ridx, ract in enumerate(self.sys_actions):
            rbit_str = f"{ridx:0{len(self.rVars)}b}" 
            act_str = f"sys_{ract}"
            self.action_map[act_str] = rbit_str
        
        offset = len(self.sys_actions)
        for eidx, eact in enumerate(self.env_actions):
            ebit_str = f"{eidx + offset:0{len(self.rVars)}b}"
            act_str = f"env_{eact}"
            self.action_map[act_str] = ebit_str
    

    def create_sym_weight_dict(self, debug: bool = False) -> None:
        for ract_full, rdd in self.action_map_sym.items():
            # env actions have 0 weight and sys actions have the weight defined in self.weight_dict
            if ract_full.startswith('sys'):
                ract = ract_full.split('_')[1]
                assert ract in self.sys_actions, f"Action {ract} not found in sys_actions list."
                self.symbolic_weight_dict[ract_full] = rdd.ite(self.manager.addConst(self.weight_dict[ract]), self.manager.addZero()) & self.tVar_map_sym['sys']
        
        self.weight = reduce(lambda x, y: x | y, self.symbolic_weight_dict.values())

        if debug:
            print("Debug: Dumping computed weights (state-action pairs):")
            self.convert_cube_to_state_ADD(self.weight, state_flag=True, action=True, verbose=True)
    

    def get_valid_row_transitions(self, rPos: int) -> List[str]:
        # sys can always choose to stay
        valid_actions = set({'STAY', 'EAST', 'WEST'})
        if rPos + 1 < self.rows:
            valid_actions.add('NORTH')
        if rPos - 1 >= 0:
            valid_actions.add('SOUTH')
        
        return valid_actions

    def get_valid_column_transitions(self, cPos: int) -> List[str]:
        # sys can always choose to stay
        valid_actions = set({'STAY', 'NORTH', 'SOUTH'})
        if cPos + 1 < self.columns:
            valid_actions.add('EAST')
        if cPos - 1 >= 0:
            valid_actions.add('WEST')
        
        return valid_actions
    

    def add_turn_var_update_rule(self):
        """
         A method to add turn variable update rule. Irrespective of the action taken, after every turn, the turn variable is flipped.
        """
        curr_pred = [self.tVar_map_sym['sys'], self.tVar_map_sym['env']]
        next_pred_str = [self.tVar_map['env'], self.tVar_map['sys']]
        for turn_bit, turn_prime_string in zip(curr_pred, next_pred_str):
            for sidx, s in enumerate(turn_prime_string):
                if s == '1':
                    self.transition_relation[self.tVar[sidx].bddPattern().__str__()] |= turn_bit
    

    def create_actions(self, player: str):
        turn_bit: ADD = self.tVar_map_sym[player]
        p_idx = 0 if player == 'sys' else 1

        # for row variables
        for r in range(self.rows):
            rVar_add: ADD = self.cube_to_add(self.xVar_map[p_idx][r], self.xVars[p_idx])
            valid_actions = self.get_valid_row_transitions(rPos=r)

            for act in valid_actions:
                rAct_cube: str = self.action_map_sym[f'{player}_{act}']
                nxt_rPos = r + Moves[act].value[0]

                for idx, prime_rVar in enumerate(self.xVar_map[p_idx][nxt_rPos]):
                    if prime_rVar == '1':
                        self.transition_relation[self.xVars[p_idx][idx].bddPattern().__str__()] |= turn_bit & rVar_add & rAct_cube
        

        # for column variables
        for c in range(self.columns):
            cVar_add: ADD = self.cube_to_add(self.yVar_map[p_idx][c], self.yVars[p_idx])
            valid_actions = self.get_valid_column_transitions(cPos=c)

            for act in valid_actions:
                rAct_cube: str = self.action_map_sym[f'{player}_{act}']
                nxt_cPos = c + Moves[act].value[1]

                for idx, prime_rVar in enumerate(self.yVar_map[p_idx][nxt_cPos]):
                    if prime_rVar == '1':
                        self.transition_relation[self.yVars[p_idx][idx].bddPattern().__str__()] |= turn_bit & cVar_add & rAct_cube
    

    def add_sys_frame_axioms(self):
        for rPos in range(self.rows):
            for cPos in range(self.columns):
                rVar_add = self.cube_to_add(self.xVar_map[0][rPos], self.xVars[0])
                cVar_add = self.cube_to_add(self.yVar_map[0][cPos], self.yVars[0])
                for act in self.action_map:
                    if not act.startswith('env'):
                        continue
                    rAct_cube: str = self.action_map_sym[act]
                    for idx, prime_rVar in enumerate(self.xVar_map[0][rPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.xVars[0][idx].bddPattern().__str__()] |= rVar_add & rAct_cube
                    
                    for idx, prime_rVar in enumerate(self.yVar_map[0][cPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.yVars[0][idx].bddPattern().__str__()] |= cVar_add & rAct_cube
    
    def add_env_frame_axioms(self):
        for rPos in range(self.rows):
            for cPos in range(self.columns):
                rVar_add = self.cube_to_add(self.xVar_map[1][rPos], self.xVars[1])
                cVar_add = self.cube_to_add(self.yVar_map[1][cPos], self.yVars[1])
                for act in self.action_map:
                    if not act.startswith('sys'):
                        continue
                    rAct_cube: str = self.action_map_sym[act]
                    for idx, prime_rVar in enumerate(self.xVar_map[1][rPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.xVars[1][idx].bddPattern().__str__()] |= rVar_add & rAct_cube
                    
                    for idx, prime_rVar in enumerate(self.yVar_map[1][cPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.yVars[1][idx].bddPattern().__str__()] |= cVar_add & rAct_cube
    

    def create_transition_relation(self):
        """
         Create the transition relation for the gridworld. We will create a transition relation for each action and then combine them together at the end. 
        """
        for player in ['sys', 'env']:
            self.create_actions(player=player)
        
        # need to add frame axioms, i.e., when it is env move Sys variables remain the same and vice versa.
        self.add_sys_frame_axioms()
        self.add_env_frame_axioms()

        self.add_turn_var_update_rule()
    

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
    

    def compute_min_max_preimage(self, preimage: ADD, valid_env_action_mask: ADD) -> ADD:
        robot_states = preimage.cofactor(self.tVar_map_sym['sys'])
        next_winning_states_robot = self.symbolic_min_abstract(robot_states, self.rVars)
        
        # take max over Env player states; but first map the invalid env actions and robot action from these states to -inf
        env_states = preimage.cofactor(self.tVar_map_sym['env'])
        preimage_for_max = valid_env_action_mask.ite(env_states, self.manager.minusInfinity()) 
        next_winning_states_env = self.symbolic_max_abstract(preimage_for_max, self.rVars)

        next_winning_states = self.tVar[0].ite(next_winning_states_robot, next_winning_states_env)
        return next_winning_states


    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.latches, self.prime_latches)
        preimage = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))

        return preimage
    

    def solve(self, verbose: bool = False, cooperative_game: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.
        """
        # initialize goal state with 0 state value and add it to the winning region
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states = self.manager.plusInfinity().min(goal)

        if verbose:
            print("Initial Winning States:")
            # by default generate cubes does not retuen cubes that point to 0 leaf. 
            # So, we manually convert the 0 leaf to a cube with leaf value 1 here for printing.
            self.convert_cube_to_state_ADD(curr_winning_states.bddInterval(0, 0).toADD(), action=False, verbose=True)
        
        # intialize the iteration counter
        layer = 0
        valid_env_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        while True:
            print(f"**************************Layer: {layer}**************************")
            preimage: ADD = self.compute_preimage(curr_winning_states)
            preimage = preimage + self.weight
            
            # take min over Sys player states; as invalid actions and human action are mapped to inf, they will not affect the min operation
            if cooperative_game:
                next_winning_states = self.symbolic_min_abstract(preimage, self.rVars)
            else:
                next_winning_states = self.compute_min_max_preimage(preimage, valid_env_action_mask=valid_env_action_mask)
            
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
    
    def convert_exlpicit_state_to_cube(self, state) -> ADD:
        """
         A smaller helper function to convert an explicit state representation to a cube. This is useful for debugging and printing purposes.
        """
        state_cube = self.manager.addOne()
        # TODO: hard coding for now, need to update so the idx macthes the respective player
        p_idx = 0
        for e in state:
            if 'sys' in e or 'env' in e:
                state_cube &= self.tVar_map_sym[e]
            elif isinstance(e, list):
                x, y = e[0], e[1]
                state_cube &= self.xVar_map_sym[p_idx][x] & self.yVar_map_sym[p_idx][y]
                p_idx += 1
        
        return state_cube
    

    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a given strategy
        """
        curr_state = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        # NOt sure if this is the best way to do it
        sys_action_mask = reduce(lambda x, y: x | y, self.sys_action_cube_list)
        env_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)

        # we always stars with Sys player turn
        turn: str ='sys'

        while (curr_state & self.goal_latch).isZero():
            curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state, state_flag=True, action=False, table_header=False, verbose=False)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval: int = list((curr_state & self.comp_winning_states).generate_cubes())[0][1]
            except IndexError:
                opt_sval: int = 0
            
            if verbose:
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)]))

            # get the action to be taken at the current state
            if turn == 'sys':
                strategy_mask = sys_action_mask.ite(strategy, self.manager.plusInfinity())
            else:
                strategy_mask = env_action_mask.ite(strategy, self.manager.plusInfinity())
            act_cube: BDD = (strategy_mask.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string]
            except KeyError:
                print("No robot action found!!")
                return
        
            turn = 'sys' if curr_state_exp[0][0][0][0] == 'sys' else'env'
            act = act_name.split('_')[1]
           
            # get the next state
            if turn == 'sys':
                curr_state_exp[0][0][0][1][0] = curr_state_exp[0][0][0][1][0] + Moves[act].value[0]
                curr_state_exp[0][0][0][1][1] = curr_state_exp[0][0][0][1][1] + Moves[act].value[1]
            elif turn == 'env':
                curr_state_exp[0][0][0][2][0] = curr_state_exp[0][0][0][2][0] + Moves[act].value[0]
                curr_state_exp[0][0][0][2][1] = curr_state_exp[0][0][0][2][1] + Moves[act].value[1]
            
            curr_state = self.convert_exlpicit_state_to_cube(curr_state_exp[0][0][0])
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Sys Action: {act_name}") if turn == 'sys' else print(f"Env Action: {act_name}")

    

    def test_preimage(self):
        sys_pos = (1, 1)
        goal_cube_sys = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (1, 0)
        goal_cube_env = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        goal_cube = self.tVar_map_sym['sys'] & goal_cube_env & goal_cube_sys
        print('Goal state:', goal_cube)
        # compute preimage 
        From = goal_cube.swapVariables(self.latches, self.prime_latches)
        preimage = From.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        print('Preimage of goal state:', preimage)
        self.convert_cube_to_state_ADD(preimage, state_flag=True, action=True, verbose=True)

        # now let takes min and max
        new_preimage = self.compute_min_max_preimage(preimage, valid_env_action_mask=reduce(lambda x, y: x | y, self.env_action_cube_list))
        print('Preimage after min max abstraction:', new_preimage)
        self.convert_cube_to_state_ADD(new_preimage, state_flag=True, action=True, verbose=True)
    

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


    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, action: bool = False, verbose: bool = False, table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the action as well, set action to True. 
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
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        xConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])
        
        yConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        

         # print the states
        states_action_pairs = []
        states_bookkeeping = [] 
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            xCube_str = []
            for e in xConf_exist_cube.values():
                xCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            yCube_str = []
            for e in yConf_exist_cube.values():
                yCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                row_states = [self.xVar_map[pidx].inv[e] for pidx, e in enumerate(xCube_str)]
            except KeyError:
                continue

            try:
                column_states = [self.yVar_map[pidx].inv[e] for pidx, e in enumerate(yCube_str)]
            except KeyError:
                continue
            
            try:
                pos = []
                for r, c in zip(row_states, column_states):
                    pos.append([r, c])
                state = [self.tVar_map.inv[tConf_cube_str]] + pos
                states_action_pairs.append([((self.tVar_map.inv[tConf_cube_str], *pos), val), None])
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
