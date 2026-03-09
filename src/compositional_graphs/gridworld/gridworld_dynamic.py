import math

from functools import reduce

from bidict import bidict
from tabulate import tabulate

from collections import defaultdict
from typing import List, Tuple, Dict

from cudd import Cudd, ADD, BDD, REORDER_GROUP_SIFT_CONV


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
          2. column variables - xVars
          3. row variables - yVars
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

    
    def create_transition_relation(self):
        """
         Create the transition relation for the gridworld. We will create a transition relation for each action and then combine them together at the end. 
        """
        pass



    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, action: bool = False, verbose: bool = False, table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
         Convert a cube to a state representation. Set the flag to True if you want to print the state only. 
         If you want to print the action as well, set action to True. 
        """
        raise NotImplementedError()