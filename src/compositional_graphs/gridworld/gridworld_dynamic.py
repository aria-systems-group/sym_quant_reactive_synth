import sys
import math
import warnings

from enum import Enum
from functools import reduce
from itertools import product

from bidict import bidict
from tabulate import tabulate

from collections import defaultdict
from typing import List, Tuple, Dict, Union, Optional

from cudd import Cudd, ADD, BDD

# Custom Types
CELL = Tuple[int, int]

class Moves(Enum):
    NORTH = (1, 0)
    SOUTH = (-1, 0)
    WEST = (0, -1)
    EAST = (0, 1)
    STAY = (0, 0)


class GridWorldDynamicGame():
    def __init__(self,
                 rows: int, columns: int,
                 init: List[CELL], goal: List[CELL],
                 players: Dict[str, int] = {'sys': 1, 'env': 1},
                 grid: Optional[Dict[str, List[CELL]]] = dict({}),
                 restricted_env_locs: Optional[List[CELL]] = [],
                 cooperative_game: bool = False,
                 enable_reordering: bool = False, **kwargs):
        """
        Initializes the GridWorldDynamicGame with the given parameters. Give, n x m gridworld, we create a turn-based game where the robot and the environment take turns to move. 
        The robot can choose to move in one of the four cardinal directions or stay in place, and the environment can do the same. 
        
        The objective of the robot is to reach the goal state while avoiding obstacles in the gridworld. 

        Args:
            rows (int): Number of rows in the grid.
            columns (int): Number of columns in the grid.
            init (tuple): Initial state configuration.
            goal (tuple): Goal state configuration.
            players (dict): Dictionary specifying the number of players for the system and environment. Default is {'sys': 1, 'env': 1}.
            grid: Optinoal Dictonary that containts information about the Atomic Propositions in the game. 
            The key is the name of the proposition and the value is a list of states where the proposition is true.
            This is used for labeling states with propositions for LTL synthesis.
        """
        self.rows = rows
        self.columns = columns
        self.sys_actions: List[str] = ['STAY', 'NORTH', 'SOUTH', 'EAST', 'WEST']
        self.env_actions: List[str] = ['NORTH', 'SOUTH', 'EAST', 'WEST']
        self.obstacles = set({'wall', 'lava'})
        self._players = players
        self._vi_layers = 0
        self.total_players = sum(self.players.values())
        self.init = init
        self.goal = goal
        self.grid = grid
        self.cooperative_game: bool = cooperative_game
        self.restricted_env_locs = restricted_env_locs
        self.manager: Cudd = Cudd(maxMem=16000000000)
        self.xVar_map = {p: bidict({}) for p in range(self.total_players)}
        self.yVar_map = {p: bidict({}) for p in range(self.total_players)}
        self.tVar_map = bidict({})
        self.eVar_map = dict({})
        self.pidx_to_pstr: Dict[int, str] = bidict({i: (f'sys{i}' if i < self.players['sys'] else f'env{i - self.players["sys"]}') for i in range(self.total_players)})

        # Predicate to Cube maps - needed for symbolic operations; also avoid multiple calls to cube_to_add()
        self.xVar_map_sym = {p: bidict({}) for p in range(self.total_players)}
        self.yVar_map_sym = {p: bidict({}) for p in range(self.total_players)}
        self.tVar_map_sym = bidict({})
        self.eVar_map_sym = bidict({})
        self.prime_xVar_map_sym = {p: bidict({}) for p in range(self.total_players)}
        self.prime_yVar_map_sym = {p: bidict({}) for p in range(self.total_players)}
        self.prime_tVar_map_sym = bidict({})

        # main method to create boolean variables for the game and the maps for both prime and non-prime variables
        self.parent_boolean_state_vars_and_maps()

        # now that the maps are initialized we create init and goal states
        self.init_latch: ADD = self.set_init_latch() 
        self.goal_latch: ADD = self.set_goal_latch()
        self.states_per_cost: Dict[int, ADD] = defaultdict(lambda: self.manager.bddZero())

        # monolithic transition relation
        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}
        self.ts_bdd_transition_fun_list: List[BDD] = []

        # create sys and env action vars and maps
        self.rVars = self.create_action_vars()
        self.sys_action_map = bidict({})
        self.env_action_map = bidict({})
        self.action_map_sym = dict({})
        self.create_action_map()

        self.rVars_cube = reduce(lambda x, y: x & y, self.rVars)

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
        self.env_action_cube_list = [[] for _ in range(self.players['env'])]
        self.sys_action_cube_list = [[] for _ in range(self.players['sys'])]
        for act_str, act_dd in self.action_map_sym.items():
            pidx = int(act_str.split('_')[0][-1])
            if act_str.startswith('env'):
                self.env_action_cube_list[pidx].append(act_dd)
            else:
                self.sys_action_cube_list[pidx].append(act_dd)
        
        self.env_action_cube_list_bdd: List[BDD] = [act.bddPattern() for act_dd in self.env_action_cube_list for act in act_dd]
        self.sys_action_cube_list_bdd: List[BDD] = [act.bddPattern() for act_dd in self.sys_action_cube_list for act in act_dd]

        self.miscellanoues_helper_stuff()

        if enable_reordering:
            self.manager.autodynEnable()
    
    @property
    def players(self):
        return self._players

    @property
    def vi_layers(self):
        return self._vi_layers

    def parent_boolean_state_vars_and_maps(self):
        # create latches - tVars + xVars + yVars
        self.create_all_boolean_state_vars_and_maps()
        self.set_latches()

        # create prime latches - prime tVars + prime xVars + prime yVars + prime eVar
        self.create_all_prime_boolean_state_vars_and_maps()
        self.set_prime_latches()
    

    def create_all_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the GridWorld Turn-Based Game.
          1. turn variables - tVars
          2. column variables - xVars
          3. row variables - yVars
          4. error variable - eVar - used to map invalid Env actions to this error state
        """
        self.tVars = self.create_player_latches()
        self.xVars, self.yVars = self.create_latches()
        self.eVars = self.create_error_latches()
        self.create_xVar_map()
        self.create_yVar_map()
        self.create_tVar_map()
        self.create_eVar_map()
        self.create_symbolic_maps(prime=False)    

    def create_all_prime_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the GridWorld Turn-Based Game.
          1. turn variables - tVars
          2. row variables - xVars
          3. column variables - yVars
          4. error variable - eVar - used to map invalid Env actions to this error state
        """
        self.prime_tVars = self.create_prime_player_latches()
        self.prime_xVars, self.prime_yVars = self.create_prime_latches()
        self.prime_eVars = self.create_prime_error_vars()
        self.create_symbolic_maps(prime=True)

    def create_player_latches(self) -> List[ADD]:
        """
        A method to create player latches. For turn-based games, we need a turn variable to keep track of which player's turn it is. 
         For the gridworld game, we have two players - the system player (robot) and the environment player.
        """
        offset = self.manager.size()
        # Do I need to the 0-bit offset?
        num_of_vars = math.ceil(math.log2(self.total_players))
        tVar: List[ADD] = [self.manager.addVar(p + offset, f't{p}') for p in range(num_of_vars)]
        return tVar
    

    def create_latches(self) -> Tuple[List[ADD], List[ADD]]:
        """
         Given a gridworld of size n x m create log(n) x vars and log(m) y vars that represent the x and y position respectively. 
        """
        x_size, y_size = math.ceil(math.log2(self.rows)), math.ceil(math.log2(self.columns))
        # the x_size and y_size match then, we need to an extra boolean vairables to offset the all 0-vector latch
        if pow(2, x_size) == self.rows:
            # we need creat an additional boolean variable. I will create one addiiotnal variable for x
            x_size += 1
        
        if pow(2, y_size) == self.columns:
            y_size += 1
        
        xVars: List[List[ADD]] = []
        for player in range(self.total_players):
            varsize = self.manager.size()
            xVars.append([self.manager.addVar(k + varsize, f'x{player}{k}') for k in range(x_size)])
        
        yVars: List[List[ADD]] = []
        for player in range(self.total_players):
            varsize = self.manager.size()
            yVars.append([self.manager.addVar(k + varsize, f'y{player}{k}') for k in range(y_size)])

        return xVars, yVars
    
    def create_error_latches(self) -> List[ADD]:
        """
         Create a two error variables to which we can map all invalid actions to error state. 

         sys-err - Sys error state - all invalid sys edges map to this error state.
         env-err - Env error state - all invalid env edges map to this error state.
        """
        varsize = self.manager.size()
        eVar = [self.manager.addVar(varsize, 'e0'), self.manager.addVar(varsize + 1, 'e1')]
        self.sys_error_cube = eVar[0]
        self.env_error_cube = eVar[1]
        # make constraint that state is valid state
        self.not_error_state_cube: ADD = ~self.sys_error_cube & ~self.env_error_cube
        return eVar
    

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
    

    def create_prime_player_latches(self) -> List[ADD]:
        varsize = self.manager.size()
        prime_tVars = [self.manager.addVar(p + varsize, f'pt{p}') for p in range(len(self.tVars))]
        return prime_tVars


    def create_prime_error_vars(self) -> List[ADD]:
        """
         A method to create prime version of the error state variables.
        """
        varsize = self.manager.size()
        prime_eVars: List[ADD] =  [self.manager.addVar(e + varsize , f"pe{e}") for e in range(len(self.eVars))]
        return prime_eVars

    def create_action_vars(self) -> Tuple[List[ADD], List[ADD]]:
        """
        Create a single method wehre we create robot action and env actions using the same of variables.
         
        This will lead to savings in the # of boolean vars needed. This approach will require
         log(num_robot_actions + num_env_actions) boolean vars.
        """
        varsize = self.manager.size()
        rVars_size = math.ceil(math.log2((len(self.sys_actions) * self.players['sys']) + (len(self.env_actions) * self.players['env'])))
        rVars: List[ADD] =  [self.manager.addVar(r + varsize , 'r' + str(r)) for r in range(rVars_size)]
        return rVars
    

    def miscellanoues_helper_stuff(self):
        self.rVars_cube_bdd: BDD = self.rVars_cube.bddPattern()
        # create cubes of valid env and sys turn var
        self.sys_tVar_cube: ADD = reduce(lambda x, y: x | y, [self.tVar_map_sym[player] for player in self.tVar_map.keys() if player.startswith('sys')])
        self.env_tVar_cube: ADD = reduce(lambda x, y: x | y, [self.tVar_map_sym[player] for player in self.tVar_map.keys() if player.startswith('env')])
        self.env_tVar_cube_bdd: BDD = self.env_tVar_cube.bddPattern()
        # create cubes of valid env and sys player action.
        self.sys_action_cube: Dict[str, ADD] = {f'sys{player}': reduce (lambda x, y: x | y, sact_list) for player, sact_list in enumerate(self.sys_action_cube_list)}
        self.env_action_cube: Dict[str, ADD] = {f'env{player}': reduce (lambda x, y: x | y, eact_list) for player, eact_list in enumerate(self.env_action_cube_list)}
        # ADD for set of valid state
        self.obsatcle_constraint_cube = self.manager.addZero()
        # used during rollout to check of the action is valid or not
        self.invalid_env_state_action_cube = self.manager.addZero()
        self.invalid_sys_state_action_cube = self.manager.addZero()
        self.create_obstacle_constraint()
    
    def get_number_of_states(self, verbose: bool = True) -> Tuple[int, int]:
        """
         A method to to compute the |Sys States| and |Env states| in the game.
         Sys States = rows x columns x turn variables
         Env States = rows x columns x turn variables
        """
        sys_states = (self.rows * self.columns) ** self.total_players
        env_states = (self.rows * self.columns) ** self.total_players
        if verbose:
            print(f'Number of States in Game: \n Sys States: {sys_states:,} \n Env States: {env_states:,} \n Total States: {sys_states + env_states:,}')
        return sys_states, env_states
        

    def create_obstacle_constraint(self):    
        for obst in self.obstacles:
            if obst not in self.grid.keys():
                continue
            # get the positions of the obstacle from the grid dict
            for pidx, player_str in enumerate(self.tVar_map.keys()):
                player_obst_const = self.manager.addZero()
                for pos in self.grid[obst]:
                    player_obst_const |= self.tVar_map_sym[player_str] & self.xVar_map_sym[pidx][pos[0]] & self.yVar_map_sym[pidx][pos[1]]
                self.obsatcle_constraint_cube |= player_obst_const
        
        # add restricted env location to obstacle constraint
        for pos in self.restricted_env_locs:
            for pidx, player_str in enumerate(self.tVar_map.keys()):
                if player_str.startswith('env'):
                    self.obsatcle_constraint_cube |= self.tVar_map_sym[player_str] & self.xVar_map_sym[pidx][pos[0]] & self.yVar_map_sym[pidx][pos[1]]


    def set_latches(self):
        self.latches: List[ADD] = self.tVars + self.eVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds]
        self.latches_bdd: List[BDD] = [latch.bddPattern() for latch in self.latches]
    

    def set_prime_latches(self):
        self.prime_latches: List[ADD] = self.prime_tVars + self.prime_eVars + [var for prime_xVar_adds in self.prime_xVars for var in prime_xVar_adds] + [var for prime_yVar_adds in self.prime_yVars for var in prime_yVar_adds]
        self.prime_latches_bdd: List[BDD] = [latch.bddPattern() for latch in self.prime_latches]
    

    def set_init_latch(self) -> ADD:
        # TODO: add check that the agent's init state must be valid position, i.e., they must not start in wall cell or lava cell from which this can not move.
        assert len(self.init) == self.total_players, "[Error]: The init state should be fully defined for the Synthesis code else the Synthesis code will not work correctly."
        init_cube = self.tVar_map_sym[self.pidx_to_pstr[0]]
        for i, (x, y) in enumerate(self.init):
            init_cube &= self.xVar_map_sym[i][x] & self.yVar_map_sym[i][y]
        return init_cube & self.not_error_state_cube
    

    def set_goal_latch(self) -> ADD:
        # TODO: Make sure the goal state must be in a valid position, i.e., goal must not be a wall cell or lava cell which can never be reached.
        goal_cube = self.manager.addZero()
        # The goal is state for the Sys coalition, so if a Sys player reaches the goal, then the goal proposition is satisfied.
        for p in range(self.players['sys']):
            for x, y in self.goal:
                goal_cube |= self.xVar_map_sym[p][x] & self.yVar_map_sym[p][y]
        # if the env takes an illegal action, then we want transition to an error state where the goal is also satisfied
        # because the environment has made an illegal move and the game terminates.
        if self.cooperative_game:
            return (goal_cube & self.not_error_state_cube)
        return (goal_cube & self.not_error_state_cube) | self.env_error_cube


    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add
    

    def create_xVar_map(self) -> None:
        for p in range(self.total_players):
            for r in range(self.rows):
                # offset is to avoid the 0-vector
                bit_str = f"{r + 1:0{len(self.xVars[p])}b}"
                self.xVar_map[p][r] = bit_str

    def create_yVar_map(self) -> None:
        for p in range(self.total_players):
            for c in range(self.columns):
                # offset is to avoid the 0-vector
                bit_str = f"{c + 1:0{len(self.yVars[p])}b}"
                self.yVar_map[p][c] = bit_str

    def create_tVar_map(self) -> None:
        for pidx, pstr in self.pidx_to_pstr.items():
            bit_str = f"{pidx:0{len(self.tVars)}b}"
            self.tVar_map[pstr] = bit_str
    
    
    def create_eVar_map(self) -> None:
        for ap in self.eVars:
            self.eVar_map[ap.bddPattern().__str__()] = '1'
            self.eVar_map_sym[ap.bddPattern().__str__()] = ap
    
    def create_symbolic_maps(self, prime: bool = False):
        """
         Small function to create symbolic maps for the xVar_map, rAction_map and eAction_map
        """
        for player in range(self.total_players):
            for k, v in self.xVar_map[player].items():
                if prime:
                    self.prime_xVar_map_sym[player][k] = self.cube_to_add(v, self.prime_xVars[player])
                else:
                    self.xVar_map_sym[player][k] = self.cube_to_add(v, self.xVars[player])
        
        for player in range(self.total_players):
            for k, v in self.yVar_map[player].items():
                if prime:
                    self.prime_yVar_map_sym[player][k] = self.cube_to_add(v, self.prime_yVars[player])
                else:
                    self.yVar_map_sym[player][k] = self.cube_to_add(v, self.yVars[player])
        
        # create turn variable symbolic maps
        for player, bit_str in self.tVar_map.items():
            if prime:
                self.prime_tVar_map_sym[player] = self.cube_to_add(bit_str, self.prime_tVars)
            else:
                self.tVar_map_sym[player] = self.cube_to_add(bit_str, self.tVars)
    
    
    def create_action_map(self) -> None:
        offset = 0
        for p in range(self.players['sys']):
            for ridx, ract in enumerate(self.sys_actions):
                rbit_str = f"{ridx + offset:0{len(self.rVars)}b}" 
                act_str = f"sys{p}_{ract}"
                self.sys_action_map[act_str] = rbit_str
                self.action_map_sym[act_str] = self.cube_to_add(rbit_str, self.rVars)
            offset += len(self.sys_actions)

        # offset = len(self.sys_actions)
        for p in range(self.players['env']):
            for eidx, eact in enumerate(self.env_actions):
                ebit_str = f"{eidx + offset:0{len(self.rVars)}b}"
                act_str = f"env{p}_{eact}"
                self.env_action_map[act_str] = ebit_str
                self.action_map_sym[act_str] = self.cube_to_add(ebit_str, self.rVars)
            offset += len(self.env_actions)
    

    def create_sym_weight_dict(self, debug: bool = False) -> None:
        for ract_full, rdd in self.action_map_sym.items():
            # env actions have 0 weight and sys actions have the weight defined in self.weight_dict
            if ract_full.startswith('sys'):
                player, ract = ract_full.split('_')[0], ract_full.split('_')[1]
                assert ract in self.sys_actions, f"Action {ract} not found in sys_actions list."
                self.symbolic_weight_dict[ract_full] = rdd.ite(self.manager.addConst(self.weight_dict[ract]), self.manager.addZero()) & self.tVar_map_sym[player]
        
        self.weight = reduce(lambda x, y: x | y, self.symbolic_weight_dict.values())

        if debug:
            print("Debug: Dumping computed weights (state-action pairs):")
            self.convert_cube_to_state_ADD(self.weight, state_flag=True, action=True, verbose=True)
    
    
    def get_valid_transitions(self, rPos: int, cPos: int, player: str) -> List[str]:
        """
         Returns a list valid agent actions you can take given  row and column position
        """
        # sys can always choose to stay
        valid_actions = set({'STAY'}) if player.startswith('sys') else set({})
        if rPos + 1 < self.rows:
            valid_actions.add('NORTH')
        if rPos - 1 >= 0:
            valid_actions.add('SOUTH')
        if cPos + 1 < self.columns:
            valid_actions.add('EAST')
        if cPos - 1 >= 0:
            valid_actions.add('WEST')
        
        return valid_actions
        

    def add_turn_var_update_rule(self):
        """
         A method to add turn variable update rule. Irrespective of the action taken, after every turn, the turn variable is updated.
        """
        curr_pred = list(self.tVar_map_sym.values())
        next_pred_str = [bit_str for k, bit_str in self.tVar_map.items() if k != 'sys0']
        next_pred_str.append(self.tVar_map['sys0'])
        self.turn_update_rule = {self.tVar_map_sym.inv[i]: self.tVar_map.inv[j] for i, j in zip(curr_pred, next_pred_str)}
        for turn_bit, turn_prime_string in zip(curr_pred, next_pred_str):
            for sidx, s in enumerate(turn_prime_string):
                if s == '1':
                    self.transition_relation[self.tVars[sidx].bddPattern().__str__()] |= turn_bit
    
    def add_error_state_self_loops(self):
        """
         A small helper method that add a self loop from the Sys and Env error states
        """
        # add self-loop for the error states
        self.transition_relation[self.env_error_cube.bddPattern().__str__()] |=  self.env_error_cube & self.env_tVar_cube
        self.transition_relation[self.sys_error_cube.bddPattern().__str__()] |=  self.sys_error_cube & self.sys_tVar_cube
    

    def create_actions(self, player: str):
        """
        New method where, I reasons the product of row and column transitions together.
          The main motivation is that, when we reaosn over Env player, I can map invalid moves to STAY action.

        Else, I had to remove the invalid Env transition and remap those to STAY action which is more computational expensive.
         The remap, could convert cubes to cubestring , an expensive process which I do not see scaling well (for 3 or more players).
        """
        turn_bit: ADD = self.tVar_map_sym[player]
        p_idx = self.pidx_to_pstr.inv[player]
        for r in range(self.rows):
            rVar_add: ADD = self.xVar_map_sym[p_idx][r]

            for c in range(self.columns):
                cVar_add: ADD = self.yVar_map_sym[p_idx][c]

                # get valid acts for grid position (r, c) - this does check for wall or other obstacles in the successor step.
                valid_actions = self.get_valid_transitions(rPos=r, cPos=c, player=player)
                # all env players have same action set. If not, this must be updated to be per player.
                invalid_actions = set(self.env_actions) - valid_actions if player.startswith('env') else set(self.sys_actions) - valid_actions

                # check if the next position is valid or not - here the next pos is invalid as it is going outside the gridworld boundary.
                if len(invalid_actions) > 0:
                    # invalid action must be mapped to an error state
                    invalid_act_cube = reduce(lambda x, y: x | y, [self.action_map_sym[f'{player}_{act}'] for act in invalid_actions])
                    tr_key = self.sys_error_cube.bddPattern().__str__() if player.startswith('sys') else self.env_error_cube.bddPattern().__str__()
                    self.transition_relation[tr_key] |= turn_bit & rVar_add & cVar_add & self.not_error_state_cube & invalid_act_cube & ~self.obsatcle_constraint_cube                  

                for act in valid_actions:
                    act_cube: str = self.action_map_sym[f'{player}_{act}']
                    nxt_rPos = r + Moves[act].value[0]
                    nxt_cPos = c + Moves[act].value[1]

                    # check if the next position is valid or not - only for Env player - here the next pos belongs to an obstacle cell.
                    if player.startswith('env') and (self.obsatcle_constraint_cube & self.tVar_map_sym[player] & self.xVar_map_sym[p_idx][nxt_rPos] & self.yVar_map_sym[p_idx][nxt_cPos]) != self.manager.addZero():
                        # invalid Env action must be mapped to error state
                        self.transition_relation[self.env_error_cube.bddPattern().__str__()] |= turn_bit & rVar_add & cVar_add & act_cube & self.not_error_state_cube & ~self.obsatcle_constraint_cube
                        continue
                    
                    transition_cube: ADD = turn_bit & rVar_add & cVar_add & act_cube & self.not_error_state_cube & ~self.obsatcle_constraint_cube

                    for idx, prime_rVar in enumerate(self.xVar_map[p_idx][nxt_rPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.xVars[p_idx][idx].bddPattern().__str__()] |= transition_cube

                    for idx, prime_rVar in enumerate(self.yVar_map[p_idx][nxt_cPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.yVars[p_idx][idx].bddPattern().__str__()] |= transition_cube  
    
    
    def add_frame_axioms(self):
        for active_player in self.tVar_map.keys():
            # all other player are passive players
            for passive_player in self.tVar_map.keys():
                if active_player == passive_player:
                    continue
                turn_bit: ADD = self.tVar_map_sym[active_player]
                pidx = self.pidx_to_pstr.inv[passive_player]
                for rPos in range(self.rows):
                    rVar_add = self.xVar_map_sym[pidx][rPos]
                    for cPos in range(self.columns):
                        cVar_add = self.yVar_map_sym[pidx][cPos]
                        act_cube: dict = self.sys_action_cube[active_player] if active_player.startswith('sys') else self.env_action_cube[active_player]
                        transition_cube: ADD = turn_bit & rVar_add & cVar_add & act_cube & self.not_error_state_cube & ~self.obsatcle_constraint_cube

                        for idx, prime_rVar in enumerate(self.xVar_map[pidx][rPos]):
                            if prime_rVar == '1':
                                self.transition_relation[self.xVars[pidx][idx].bddPattern().__str__()] |= transition_cube
                        
                        for idx, prime_rVar in enumerate(self.yVar_map[pidx][cPos]):
                            if prime_rVar == '1':
                                self.transition_relation[self.yVars[pidx][idx].bddPattern().__str__()] |= transition_cube
    

    def convert_mono_tr_to_action_tr(self):
        for tr_bdd in self.transition_relation.values():
            self.ts_bdd_transition_fun_list.append(tr_bdd.bddPattern())
    
    def get_states_per_cost(self):
        """
         A helper function that takes in the ADD weight and return a vector of 0-1 BDD per cost.
        """
        for w in self.weight_dict.values():
            self.states_per_cost[w] = self.weight.bddInterval(w, w)
        
        # manually add env action to cost zero
        self.states_per_cost[0] |= self.env_tVar_cube_bdd & reduce(lambda x, y: x | y, self.env_action_cube_list_bdd)
    

    def convert_vector_of_bdd_to_add(self, bdd_vector: Dict[int, BDD]) -> ADD:
        """
         A helper function that converts a vector of BDDs to an ADD. Used in pure BDD solver method for 
          (1) printing the winning states
          (2) returning the preimage strategy
        """
        result_add = self.manager.plusInfinity()
        for sval, sbdd in bdd_vector.items():
            result_add = sbdd.toADD().ite(self.manager.addConst(sval), result_add)
        return result_add
    
    
    def check_reached_fixpoint_bdd(self, curr_winning_states: Dict[int, BDD], next_winning_states: Dict[int, BDD]) -> bool:
        curr_winning_states_add = self.convert_vector_of_bdd_to_add(bdd_vector=curr_winning_states)
        next_winning_states_add = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states)
        if not next_winning_states_add.compare(curr_winning_states_add, 2):
            return False
        return True
        
    
    def post_process_transition_relation(self, debug: bool = False):
        """
         A function that removes the transitions that end up in a invalid state, such as action that fo to wall or lava cells. 
         This is done by adding a comuting all of the transition using the preimage operation and removing this from the transition relation.
        """
        tr_to_remove = self.manager.addZero()
        for obst in self.obstacles:
            if obst not in self.grid.keys():
                continue
            # we only remove invalid sys states to walls as the invalid env were already take care of during construction of the TR
            for curr_player, succ_player in self.turn_update_rule.items():
                if curr_player.startswith('sys'):
                    pidx = self.pidx_to_pstr.inv[succ_player]
                    for pos in self.grid[obst]:
                        state_primed: ADD = (self.xVar_map_sym[pidx][pos[0]] & self.yVar_map_sym[pidx][pos[1]]).swapVariables(self.latches, self.prime_latches)
                        tr_to_remove |= (state_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))) & self.sys_tVar_cube
        
        # remove the edges
        for tr in self.transition_relation.keys():
            self.transition_relation[tr] &= ~tr_to_remove
        
        # print for debugging
        if debug:
            print("Debug: Dumping transitions to remove (state-action pairs):")
            self.convert_cube_to_state_ADD(tr_to_remove, action=True, verbose=True)
    

    def create_transition_relation(self):
        """
         Create the transition relation for the gridworld. We will create a transition relation for each action and then combine them together at the end. 
        """
        for player in self.tVar_map.keys():
            self.create_actions(player=player)
        
        # need to add frame axioms, i.e., when it is env move Sys variables remain the same and vice versa.
        self.add_frame_axioms()

        self.add_turn_var_update_rule()

        # post process the transition relation to remove transitions that lead to invalid states, such as wall and lava cells.
        self.post_process_transition_relation(debug=False)
        self.add_error_state_self_loops()
        print("Finished creating transition relation.")
    

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
    

    def compute_min_max_preimage(self, preimage: ADD) -> ADD:
        all_sys_action_cube = reduce(lambda x, y: x | y, self.sys_action_cube.values())
        robot_states = preimage & self.sys_tVar_cube
        robot_states = all_sys_action_cube.ite(robot_states, self.manager.plusInfinity())
        next_winning_states_robot = self.symbolic_min_abstract(robot_states, self.rVars)
        
        # take max over Env player states; but first map the invalid env actions and robot action from these states to -inf
        next_winning_states_env = self.manager.plusInfinity()
        env_states = preimage & self.env_tVar_cube
        for pstr, eact_cube in self.env_action_cube.items():
            preimage_for_max = eact_cube.ite(env_states, self.manager.minusInfinity())
            winning_states_env = self.symbolic_max_abstract(preimage_for_max, self.rVars)
            next_winning_states_env = self.tVar_map_sym[pstr].ite(winning_states_env, next_winning_states_env)

        return next_winning_states_robot | next_winning_states_env
        # return self.sys_tVar_cube.ite(next_winning_states_robot, next_winning_states_env)

    # def compute_min_max_preimage(self, preimage: ADD) -> ADD:
    #     next_winning_states = self.manager.plusInfinity()
    #     # next_winning_states_robot = self.manager.plusInfinity()
    #     robot_states = preimage & self.sys_tVar_cube
    #     for pstr, ract_cube in self.sys_action_cube.items():
    #         preimage_for_min = ract_cube.ite(robot_states, self.manager.plusInfinity())
    #         winning_states_sys = self.symbolic_min_abstract(preimage_for_min, self.rVars)
    #         # next_winning_states_robot = self.tVar_map_sym[pstr].ite(winning_states_sys, next_winning_states_robot)
    #         next_winning_states = self.tVar_map_sym[pstr].ite(winning_states_sys, next_winning_states)
    #     # next_winning_states_robot = self.symbolic_min_abstract(robot_states, self.rVars)
        
    #     # take max over Env player states; but first map the invalid env actions and robot action from these states to -inf
    #     # next_winning_states_env = self.manager.plusInfinity()
    #     env_states = preimage & self.env_tVar_cube
    #     for pstr, eact_cube in self.env_action_cube.items():
    #         preimage_for_max = eact_cube.ite(env_states, self.manager.minusInfinity())
    #         winning_states_env = self.symbolic_max_abstract(preimage_for_max, self.rVars)
    #         # next_winning_states_env = self.tVar_map_sym[pstr].ite(winning_states_env, next_winning_states_env)
    #         next_winning_states = self.tVar_map_sym[pstr].ite(winning_states_env, next_winning_states)

    #     return next_winning_states
        # return next_winning_states_robot | next_winning_states_env


    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.latches, self.prime_latches)
        preimage = curr_winning_states_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))

        return preimage
    

    def compute_min_preimage_pure_bdd(self, preimage: Dict[int, BDD]) -> Dict[int, BDD]:
        minmin_preimage = defaultdict(self.manager.bddZero) 
        states_action_pairs: BDD = reduce(lambda x, y: x | y, preimage.values())
        states: BDD = states_action_pairs.existAbstract(self.rVars_cube_bdd)
        for sval in sorted(preimage.keys()):
            # intersect with finite valued states for Sys and Env player
            sval_to_keep = preimage[sval].existAbstract(self.rVars_cube_bdd) & states
            minmin_preimage[sval] |= sval_to_keep
            states &= ~sval_to_keep
        assert states.isZero() == True, "Error in computing min for system and env states"

        return minmin_preimage
    

    def compute_min_max_preimage_pure_bdd(self, preimage: Dict[int, BDD], debug: bool = False) -> Dict[int, BDD]:
        minmax_preimage = defaultdict(self.manager.bddZero)
        states_action_pairs: BDD = reduce(lambda x, y: x | y, preimage.values())
        states: BDD = states_action_pairs.existAbstract(self.rVars_cube_bdd)

        # Remove Env state(s) that do not have a finite value under ALL actions. 
        # For multiple Env player, we do it per player action basis.
        for player in self.tVar_map.keys():
            if not player.startswith('env'):
                continue
            env_state_w_inf_val = (self.tVar_map_sym[player] & self.env_action_cube[player]).bddPattern() & ~states_action_pairs
            env_state_w_inf_val = env_state_w_inf_val.existAbstract(self.rVars_cube_bdd)
            env_finite_valued_states = states & self.tVar_map_sym[player].bddPattern() & ~env_state_w_inf_val

            if env_finite_valued_states.isZero():
                continue
        
            for sval in sorted(preimage.keys(), reverse=True):
                # intersect with env finite valued states
                sval_to_keep = preimage[sval].existAbstract(self.rVars_cube_bdd) & env_finite_valued_states
                minmax_preimage[sval] |= sval_to_keep
                env_finite_valued_states &= ~sval_to_keep
        
        if debug:
            assert env_finite_valued_states.isZero() == True, "Error in computing max for env states"
        
        robot_state_bdd: BDD = states & ~self.env_tVar_cube_bdd

        for sval in sorted(preimage.keys()):
            # intersect with env finite valued states
            sval_to_keep = preimage[sval].existAbstract(self.rVars_cube_bdd) & robot_state_bdd
            minmax_preimage[sval] |= sval_to_keep
            robot_state_bdd &= ~sval_to_keep
        
        if debug:
            assert robot_state_bdd.isZero() == True, "Error in computing min for system states"

        return minmax_preimage


    def compute_min_goal_states(self, preimage: Dict[int, BDD], goal: Dict[int, BDD]) -> Dict[int, BDD]:
        for goal_sval in sorted(goal.keys()):
            for sval in sorted(preimage.keys()):
                # if there exists states in goal state, then we override the state value in preimage
                if sval != goal_sval:
                    sval_to_update = preimage[sval] & goal[goal_sval]
                    if not sval_to_update.isZero():
                        preimage[sval] &= ~sval_to_update
                        preimage[goal_sval] |= sval_to_update
                    
                    # add the goal states back to the preimage with their respective goal sval
                    preimage[goal_sval] |= goal[goal_sval]
        return preimage
    

    def hybrid_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            succ_states_prime = succ_states.swapVariables(self.latches_bdd, self.prime_latches_bdd)
            pre_states: BDD = succ_states_prime.vectorCompose(self.prime_latches_bdd, self.ts_bdd_transition_fun_list)

            if not pre_states.isZero():
                assert pre_buckets[sval] & pre_states == self.manager.bddZero(), "Make sure there are no overlapping states in the pre buckets..."
                pre_buckets[sval] |= pre_states

        # unions of all predecessors
        if not return_bdd:
            preimage = self.manager.plusInfinity()
            for sval, add_bucket in pre_buckets.items():
                preimage = add_bucket.toADD().ite(self.manager.addConst(sval), preimage)
            
            return preimage
        return pre_buckets


    def convert_monolithic_add_to_bdd_buckets(self, monolithic_add: ADD, layer: int, c_max: int) -> Dict[int, BDD]:    
        win_state_bucket: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        
        # convert the winning states into buckets of BDD
        _max_interval_val = layer * c_max
        for sval in range(_max_interval_val + 1):
            # get the states with state value equal to sval and store them in their respective bukcets
            win_sval = monolithic_add.bddInterval(sval, sval)

            if not win_sval.isZero():
                win_state_bucket[sval] |= win_sval
        
        return win_state_bucket
    

    def solve(self, verbose: bool = False) -> Union[ADD, None]:
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

        while True:
            print(f"**************************Layer: {layer}**************************")
            preimage: ADD = self.compute_preimage(curr_winning_states)
            preimage = preimage + self.weight
            
            # take min over Sys player states; as invalid actions and env action are mapped to inf, they will not affect the min operation
            if self.cooperative_game:
                next_winning_states = self.symbolic_min_abstract(preimage, self.rVars)
            else:
                next_winning_states = self.compute_min_max_preimage(preimage)
            
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, action=False, verbose=verbose)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                self._vi_layers = layer
                if curr_winning_states.restrict(self.init_latch) != self.manager.plusInfinity():
                    if curr_winning_states.restrict(self.init_latch) == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the env can complete the task for the robot without expending energy!!")
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


    def hybrid_solve(self, verbose: bool = False) ->  Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.

          This method is different from the pure ADD approach as it first converts the current winning states 
           into BDD buckets based on the state values and then computes the preimage using BDD vectorCompose operation.
        """
        # create Partitiotned TR based on actions
        self.convert_mono_tr_to_action_tr()

        # initialize goal state with 0 state value and add it to the winnign regiom
        goal = self.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states = self.manager.plusInfinity().min(goal)
        next_winning_states = self.manager.plusInfinity()
        
        # intialize the iteration counter
        layer = 0
        c_max: int = int(list(self.weight.findMax().generate_cubes())[0][1])

        while True:
            print(f"**************************Layer: {layer}**************************")
            win_state_bucket = self.convert_monolithic_add_to_bdd_buckets(monolithic_add=curr_winning_states, layer=layer, c_max=c_max)
            preimage = self.hybrid_compute_preimage(win_state_bucket=win_state_bucket)
                    
            # add the action costs associated with the robot actions
            preimage = preimage + self.weight
            # take min over Sys player states; as invalid actions and env action are mapped to inf, they will not affect the min operation
            if self.cooperative_game:
                next_winning_states = self.symbolic_min_abstract(preimage, self.rVars)
            else:
                next_winning_states = self.compute_min_max_preimage(preimage=preimage)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, action=False, verbose=verbose)         
            
            if next_winning_states.compare(curr_winning_states, 2):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                self._vi_layers = layer
                if curr_winning_states.restrict(self.init_latch) != self.manager.plusInfinity():
                    if curr_winning_states.restrict(self.init_latch) == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the env can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    strategy = preimage.min(goal)
                    if init_val < math.inf:
                        return strategy, curr_winning_states
                    else:
                        return None, None
                else:
                    print(f"No Winning Strategy Exists!! The State value is {math.inf}")
                return None, None
            
            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def pure_bdd_solve(self, verbose: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm to compute the optimal cost strategy for the Sys player (robot)
          to reach the goal state.
        """
        # create Partitioned TR based on actions
        self.convert_mono_tr_to_action_tr()
        self.get_states_per_cost()
        goal_states_buckets = defaultdict(lambda: self.manager.bddZero())

        # initialize goal state with 0 state value and add it to the winning region
        curr_winning_states = defaultdict(lambda: self.manager.bddZero())
        next_winning_states = defaultdict(lambda: self.manager.bddZero())
        
        # initialize the goal state bucket with 0 cost
        curr_winning_states[0] |= self.goal_latch.bddPattern()
        goal_states_buckets[0] |= self.goal_latch.bddPattern()
       
        # intialize the iteration counter
        layer = 0

        # print the initial winning states
        if verbose:
            print("Initial Winning States:")
            # by default generate cubes does not return cubes that point to 0 leaf. 
            # So, we manually convert the 0 leaf to a cube with leaf value 1 here for printing.
            self.convert_cube_to_state_ADD(curr_winning_states[0].toADD(), action=False, verbose=True)

        while True:
            print(f"**************************Layer: {layer}**************************")
            # compute preimage
            vector_preimage: Dict[int, BDD] = self.hybrid_compute_preimage(win_state_bucket=curr_winning_states, return_bdd=True)
            
            # add the action costs associated with the robot actions
            next_winning_states = defaultdict(lambda: self.manager.bddZero())
            for sCost, sbdd in self.states_per_cost.items():
                for pre_sVal, pre_sbdd in vector_preimage.items():
                    total_cost: int = sCost + pre_sVal
                    
                    common_states: BDD = sbdd & pre_sbdd
                    if not common_states.isZero():
                        next_winning_states[total_cost] |= common_states
            
            # take min over Sys player states; as invalid actions and env action are mapped to inf, they will not affect the min operation
            if self.cooperative_game:
                next_winning_states_opt = self.compute_min_preimage_pure_bdd(preimage=next_winning_states)
            else:
                next_winning_states_opt = self.compute_min_max_preimage_pure_bdd(preimage=next_winning_states, debug=False)
            # retain the min over goal states - here all goal states are at 0 cost
            next_winning_states_opt = self.compute_min_goal_states(preimage=next_winning_states_opt, goal=goal_states_buckets)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                # unions of all predecessors along with their state values - ADD used for easy printing only
                preimage = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states_opt)
                self.convert_cube_to_state_ADD(preimage, action=False, verbose=verbose)
            
            
            if self.check_reached_fixpoint_bdd(curr_winning_states=curr_winning_states, next_winning_states=next_winning_states_opt):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                init_val = math.inf
                for sval, sbdd in curr_winning_states.items():
                    if sbdd & self.init_latch.bddPattern() != self.manager.bddZero():
                        init_val: int = sval
                        print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                        break
                self._vi_layers = layer
                self.comp_winning_states = self.convert_vector_of_bdd_to_add(bdd_vector=curr_winning_states)
                # post process the strategy to return as monolithic ADD that corresponds to strategy
                strategy: ADD = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states)
                goal: ADD = self.convert_vector_of_bdd_to_add(bdd_vector=goal_states_buckets)
                if init_val < math.inf:
                    return strategy.min(goal), self.comp_winning_states.min(goal)
                else:
                    print(f"No Winning Strategy Exists!! The State value is {math.inf}")
                    return None, None
            
            # update the counter
            layer += 1

            # swap the winning states; can't do  curr_winning_states = next_winning_states_opt as python is pass by value of reference
            curr_winning_states = defaultdict(lambda: self.manager.bddZero())
            for sval in next_winning_states_opt.keys():
                curr_winning_states[sval] |= next_winning_states_opt[sval]

    
    
    def convert_exlpicit_state_to_cube(self, state) -> ADD:
        """
         A smaller helper function to convert an explicit state representation to a cube. This is useful for debugging and printing purposes.
        """
        state_cube = self.manager.addOne()
        p_idx = 0
        for e in state:
            if 'sys' in e or 'env' in e:
                state_cube &= self.tVar_map_sym[e]
            elif isinstance(e, list):
                x, y = e[0], e[1]
                state_cube &= self.xVar_map_sym[p_idx][x] & self.yVar_map_sym[p_idx][y]
                p_idx += 1
        
        return state_cube
    

    def get_next_state(self, curr_state_exp: Tuple, act: str) -> Tuple[ADD, str, Tuple]:        
        act_name = act.split('_')[1]
        # update this for the foor env
        next_state = [i for i in curr_state_exp[:self.total_players + 1]]
        pidx = int(self.pidx_to_pstr.inv[curr_state_exp[0]])
        # get the next state + 1 is account for the turn var at the start
        nxt_x = curr_state_exp[pidx + 1][0] + Moves[act_name].value[0]
        nxt_y = curr_state_exp[pidx + 1][1] + Moves[act_name].value[1]
        next_state[0] = self.pidx_to_pstr[pidx + 1] if pidx + 1 < len(self.pidx_to_pstr) else self.pidx_to_pstr[0]  # switch turn after the move
        next_state[pidx + 1] = [nxt_x, nxt_y]  # sys pos
        return self.convert_exlpicit_state_to_cube(next_state) & self.not_error_state_cube, act, next_state + ['0', '0']
    

    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a given strategy
        """
        curr_state = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        self.invalid_env_state_action_cube = self.transition_relation[self.env_error_cube.bddPattern().__str__()]
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
            act_cube: BDD = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')
            # only choose valid action. By constuction env will always have atleast one valid action. 
            while not (curr_state & act_cube.toADD() & self.invalid_env_state_action_cube).isZero():
                act_cube: BDD = (strategy.restrict(curr_state)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
                act_cube_string = act_cube.cubeString().replace('-', '')

            # get the action to be taken at the current state
            turn = 'sys' if curr_state_exp[0][0][0][0].startswith('sys') else 'env'
            try:
                act_name = self.sys_action_map.inv[act_cube_string] if turn == 'sys' else self.env_action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return

            # get the next state
            curr_state, act_name, _ = self.get_next_state(curr_state_exp=curr_state_exp[0][0][0], act=act_name)       
            
            # printing the action here as the Env action is overriden above. This because invalid Env moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Sys Action: {act_name}") if turn == 'sys' else print(f"Env Action: {act_name}")
    

    def test_preimage(self):
        sys_pos0 = (1, 1)
        goal_cube_sys0 = self.xVar_map_sym[0][sys_pos0[0]] & self.yVar_map_sym[0][sys_pos0[1]]
        sys_pos1 = (1, 1)
        goal_cube_sys1 = self.xVar_map_sym[1][sys_pos1[0]] & self.yVar_map_sym[1][sys_pos1[1]]
        env_pos1 = (1, 2)
        goal_cube_env1 = self.xVar_map_sym[2][env_pos1[0]] & self.yVar_map_sym[2][env_pos1[1]]
        env_pos2 = (2, 0)
        goal_cube_env2 = self.xVar_map_sym[3][env_pos2[0]] & self.yVar_map_sym[3][env_pos2[1]]
        goal_cube = self.tVar_map_sym['env1'] & goal_cube_sys0 & goal_cube_sys1 & goal_cube_env1 & goal_cube_env2
        print('Goal state:', goal_cube)
        # compute preimage 
        From = goal_cube.swapVariables(self.latches, self.prime_latches)
        preimage = From.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        print('Preimage of goal state:', preimage)
        self.convert_cube_to_state_ADD(preimage, state_flag=True, action=True, verbose=True)

        preimage = preimage.ite(self.manager.addZero(), self.manager.plusInfinity())
        preimage = preimage + self.weight

        # now let takes min and max
        new_preimage = self.compute_min_max_preimage(preimage)
        # print('Preimage after min max abstraction:', new_preimage)
        self.convert_cube_to_state_ADD(new_preimage, state_flag=True, action=False, verbose=True)
    

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


    def convert_cube_to_state_ADD(self,
                                  dd: ADD, state_flag: bool = True,
                                  action: bool = False, verbose: bool = False,
                                  table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
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
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)
        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + self.eVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])

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
                sys_err_state = cube.bddPattern().cubeString()[sys_eidx].replace('-', '')
                env_err_state = cube.bddPattern().cubeString()[env_eidx].replace('-', '')
                state = [self.tVar_map.inv[tConf_cube_str]] + pos + [sys_err_state, env_err_state]
                states_action_pairs.append([((self.tVar_map.inv[tConf_cube_str], *pos, sys_err_state, env_err_state), val), None])
            except KeyError:
                continue
            
            if action:
                try:
                    if self.tVar_map.inv[tConf_cube_str].startswith('sys'):
                        rCube_str = cube.bddPattern().cubeString()[start_rvar_idx:end_rvar_idx + 1].replace('-', '')
                        action_str = self.sys_action_map.inv[rCube_str]
                    elif self.tVar_map.inv[tConf_cube_str].startswith('env'):
                        eCube_str = cube.bddPattern().cubeString()[start_rvar_idx:end_rvar_idx + 1].replace('-', '')
                        action_str = self.env_action_map.inv[eCube_str]
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