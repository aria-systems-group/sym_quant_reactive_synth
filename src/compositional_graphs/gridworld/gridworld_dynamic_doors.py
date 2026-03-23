import math
import warnings

from bidict import bidict
from tabulate import tabulate

from functools import reduce
from typing import List, Tuple, Dict, Union, Optional

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame, Moves, CELL


class GridWorldDynamicDoorsGame(GridWorldDynamicGame):
    def __init__(self, rows: int, columns: int, init: List[Tuple[int, int]], goal: List[Tuple[int, int]], grid: Dict[str, List[Tuple[int, int]]], restricted_env_locs: Optional[List[CELL]] = None, enable_reordering=False):
        """
         Inherit the Gridworld Dyanmic Game and augment it with doors.  
        """
        assert isinstance(grid['door'], list) and len(grid['door']) > 0, "If you are including door in the grid dict, make sure to provide a non-empty list of door locations."
        self._door_status = ['unclaimed', 'sys', 'env']
        self.dVar_map = {d: bidict({}) for d in range(len(grid['door']))}
        self.dVar_map_sym = {d: bidict({}) for d in range(len(grid['door']))}
        super().__init__(rows=rows, columns=columns, init=init, goal=goal, grid=grid, restricted_env_locs=restricted_env_locs, enable_reordering=enable_reordering)

    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()    
        self.dVars = self.create_door_vars()
        self.create_dVar_map(prime=False)
        self.all_door_uncalimed = reduce(lambda a, b: a & b, [self.dVar_map_sym[d_idx]['unclaimed'] for d_idx in range(len(self.grid['door']))])
    

    def create_all_prime_boolean_state_vars_and_maps(self):
        super().create_all_prime_boolean_state_vars_and_maps()
        self.prime_dVar_map_sym = {d: bidict({}) for d in range(len(self.grid['door']))}
        self.prime_dVars = self.create_prime_door_vars()
        self.create_dVar_map(prime=True)
    

    def create_door_vars(self) -> List[List[ADD]]:
        door_vars = math.ceil(math.log2(len(self._door_status)))
        dVars: List[List[ADD]] = [] 
        for d_idx in range(len(self.grid['door'])):
            varsize = self.manager.size()
            dVars.append([self.manager.addVar(s + varsize, f'd{d_idx}{s}') for s in range(door_vars)])
        return dVars

    def create_prime_door_vars(self) -> List[List[ADD]]:
        dVars_prime: List[List[ADD]] = [] 
        for d_idx, d in enumerate(self.dVars):
            varsize = self.manager.size()
            dVars_prime.append([self.manager.addVar(s + varsize, f'pd{d_idx}{s}') for s in range(len(d))])
        return dVars_prime
    
    
    def create_dVar_map(self, prime: bool = False) -> None:
        for d_idx in range(len(self.grid['door'])):
            for s_idx, d_status in enumerate(self._door_status):
                bit_str = f"{s_idx:0{len(self.dVars[d_idx])}b}"
                if prime:
                    self.prime_dVar_map_sym[d_idx][d_status] = self.cube_to_add(bit_str, self.prime_dVars[d_idx])
                else:
                    self.dVar_map[d_idx][d_status] = bit_str
                    self.dVar_map_sym[d_idx][d_status] = self.cube_to_add(bit_str, self.dVars[d_idx])
    

    def set_latches(self):
        super().set_latches()
        self.latches += [var for dVar_adds in self.dVars for var in dVar_adds]
        self.latches_bdd += [var.bddPattern() for dVar_adds in self.dVars for var in dVar_adds]
    
    def set_prime_latches(self):
        super().set_prime_latches()
        self.prime_latches += [var for dVar_adds in self.prime_dVars for var in dVar_adds]
        self.prime_latches_bdd += [var.bddPattern() for dVar_adds in self.prime_dVars for var in dVar_adds]
    

    def set_init_latch(self) -> ADD:
        return super().set_init_latch() & self.all_door_uncalimed
    

    def get_number_of_states(self, verbose: bool = True) -> Tuple[int, int]:
        """
        A method to to compute the |Sys States| and |Env states| in the game.
         Sys States = rows x columns x |door_status|^number_of_doors
         Env States = rows x columns x |door_status|^number_of_doors
        """
        sys_states = self.rows * self.columns * (len(self._door_status) ** len(self.grid['door']))
        env_states = self.rows * self.columns * (len(self._door_status) ** len(self.grid['door']))
        if verbose:
            print(f'Number of States in Game: \n Sys States: {sys_states:,} \n Env States: {env_states:,} \n Total States: {sys_states + env_states:,}')
        return sys_states, env_states

    def get_door_constraint(self, cell: CELL, player: str) -> ADD:
        """
        Given the grid cell, this method looks up if there is a door in the cell. If yes, it returns the corresponding constraint
           that encodes that the door must be open for the player to pass through.
        """
        door_constraint = self.manager.addZero()
        if cell in self.grid['door']:
            # we have a door constraint for this cell
            d_idx: int = self.grid['door'].index(cell)
            not_door_owner = 'env' if player == 'sys' else 'sys'
            return ~self.dVar_map_sym[d_idx][not_door_owner]
        
        return ~door_constraint

    
    def create_door_transitions(self):
        for didx, (dr, dc) in enumerate(self.grid['door']):
            sys_at_door = (self.xVar_map_sym[0][dr] & self.yVar_map_sym[0][dc])
            env_at_door = (self.xVar_map_sym[1][dr] & self.yVar_map_sym[1][dc])
            
            for idx, prime_dVar in enumerate(self.dVar_map[didx]['sys']):
                if prime_dVar == '1':
                    self.transition_relation[self.dVars[didx][idx].bddPattern().__str__()] |=  ~self.dVar_map_sym[didx]['env'] & sys_at_door
            
            for idx, prime_dVar in enumerate(self.dVar_map[didx]['env']):
                if prime_dVar == '1':
                    self.transition_relation[self.dVars[didx][idx].bddPattern().__str__()] |=  ~self.dVar_map_sym[didx]['sys'] & env_at_door
    

    def create_actions(self, player: str):
        """
        Override base class method to create transition relation with door constraints. 
        The main difference is that when the player is trying to pass through a door cell, we need to check if the door is open for the player or not. If not, then the player cannot pass through and must stay in the same cell.
        """
        turn_bit: ADD = self.tVar_map_sym[player]
        p_idx = 0 if player == 'sys' else 1
        for d_idx in range(len(self.grid['door'])):
            for d_status in self._door_status:
                dConf_cube: ADD = self.dVar_map_sym[d_idx][d_status]
                
                for r in range(self.rows):
                    rVar_add: ADD = self.cube_to_add(self.xVar_map[p_idx][r], self.xVars[p_idx])

                    for c in range(self.columns):
                        cVar_add: ADD = self.cube_to_add(self.yVar_map[p_idx][c], self.yVars[p_idx])
                        
                        # get valid acts for grid position (r, c) - this does check for wall or other obstacles in the successor step.
                        valid_actions = self.get_valid_transitions(rPos=r, cPos=c, player=player)
                        invalid_actions = set(self.env_actions) - valid_actions

                        if player == 'env' and len(invalid_actions) > 0:
                            # invalid action must be mapped to an error state
                            invalid_act_cube = reduce(lambda x, y: x | y, [self.action_map_sym[f'{player}_{e_act}'] for e_act in invalid_actions])
                            self.invalid_env_state_action_cube |= turn_bit & dConf_cube & rVar_add & cVar_add & ~self.eVar[0] & invalid_act_cube & ~self.obsatcle_constraint_cube 
                            self.transition_relation[self.eVar[0].bddPattern().__str__()] |= turn_bit & dConf_cube & rVar_add & cVar_add & ~self.eVar[0] & invalid_act_cube & ~self.obsatcle_constraint_cube

                        for act in valid_actions:
                            act_cube: str = self.action_map_sym[f'{player}_{act}']
                            nxt_rPos = r + Moves[act].value[0]
                            nxt_cPos = c + Moves[act].value[1]

                            # check if the next position is valid or not - only for Env player
                            if player == 'env' and (self.obsatcle_constraint_cube & self.tVar_map_sym[player] & self.xVar_map_sym[p_idx][nxt_rPos] & self.yVar_map_sym[p_idx][nxt_cPos]) != self.manager.addZero():
                                # invalid Env action must be mapped to error state
                                self.invalid_env_state_action_cube |= turn_bit & dConf_cube & rVar_add & cVar_add & act_cube & ~self.eVar[0] & ~self.obsatcle_constraint_cube
                                self.transition_relation[self.eVar[0].bddPattern().__str__()] |= turn_bit & dConf_cube & rVar_add & cVar_add & act_cube & ~self.eVar[0] & ~self.obsatcle_constraint_cube
                                continue
                            
                            transition_cube: ADD = turn_bit & dConf_cube & rVar_add & cVar_add & act_cube & ~self.obsatcle_constraint_cube

                            if (r, c) == self.grid['door'][d_idx]:
                                transition_cube &= self.get_door_constraint(cell=(r, c), player=player)

                            for idx, prime_rVar in enumerate(self.xVar_map[p_idx][nxt_rPos]):
                                if prime_rVar == '1':
                                    self.transition_relation[self.xVars[p_idx][idx].bddPattern().__str__()] |= transition_cube

                            for idx, prime_rVar in enumerate(self.yVar_map[p_idx][nxt_cPos]):
                                if prime_rVar == '1':
                                    self.transition_relation[self.yVars[p_idx][idx].bddPattern().__str__()] |= transition_cube
                            
                            # updated door status
                            if d_status == 'unclaimed' and (r, c) == self.grid['door'][d_idx]:
                                # if the door is unclaimed and the player is passing through it, then the door becomes claimed by the player
                                for idx, prime_dVar in enumerate(self.dVar_map[d_idx][player]):
                                    if prime_dVar == '1':
                                        self.transition_relation[self.dVars[d_idx][idx].bddPattern().__str__()] |= transition_cube
                            else:
                                # if the door is already unclaimed or already claimed by the same player, then the door status does not change
                                for idx, prime_dVar in enumerate(self.dVar_map[d_idx][d_status]):
                                    if prime_dVar == '1':
                                        self.transition_relation[self.dVars[d_idx][idx].bddPattern().__str__()] |= transition_cube
    

    def get_next_state(self, turn: str, curr_state_exp: ADD, act: str) -> Tuple[ADD, str, Tuple]:        
        """
        In addition to updating the state, we need to update the door status as well. So, we need to check if a player is passing through a door or not. 
         If yes, then we need to update the door status accordingly. We update the dorr status, when the player is at the door cell.
        """
        act_name = act.split('_')[1]
        next_state = [i for i in curr_state_exp]
        # get the next state
        if turn == 'sys':
            # Sys action are always valid
            nxt_x = curr_state_exp[1][0] + Moves[act_name].value[0]
            nxt_y = curr_state_exp[1][1] + Moves[act_name].value[1]
            next_state[0] = 'env'  # switch turn after sys move
            next_state[1] = [nxt_x, nxt_y]  # sys pos

            # update door status - starts after 3rd index in the state representation
            # TODO: Hardcoding the start of index. Need to change when we have more agents
            # The status only chnages from unclaimed to claimed by one of the player and then remains the same.
            # During Sys's turn, we check if the Env player state is passing through the door
            for d_status in curr_state_exp[3:]:
                if d_status == 'unclaimed' and (curr_state_exp[2][0], curr_state_exp[2][1]) in self.grid['door']:
                    # if the door is unclaimed and the Sys is passing through it, then the door becomes claimed by Sys
                    next_state[curr_state_exp.index(d_status)] = 'env'
            
            door_statuses: ADD = reduce(lambda a, b: a & b, [self.dVar_map_sym[idx][e] for idx, e in enumerate(next_state[3:-1])])

            return self.convert_exlpicit_state_to_cube(next_state[:3]) & door_statuses & ~self.eVar[0], act, next_state
        
        elif turn == 'env':
            # first check if it is a valid move or not; if not valid, then map it to STAY action
            if (self.invalid_env_state_action_cube & self.convert_exlpicit_state_to_cube(curr_state_exp[:3]) & self.action_map_sym[act]).isZero() is False:
                raise warnings.warn(f"Invalid Env action {act_name} taken at state {curr_state_exp}. This should not happen. Fix this!!!")
                # invalid Env action, map it to STAY action
                nxt_x = curr_state_exp[2][0]
                nxt_y = curr_state_exp[2][1]
                next_state[0] = 'sys'  # switch turn after sys move 
                next_state[2] = [nxt_x, nxt_y]
                # TODO: Hardcoding the start of index. Need to change when we have more agents
                for d_status in curr_state_exp[3:]:
                    # During Env's turn, we check if the Sys player state is passing through the door
                    if d_status == 'unclaimed' and (curr_state_exp[1][0], curr_state_exp[1][1]) in self.grid['door']:
                        # if the door is unclaimed and the Sys is passing through it, then the door becomes claimed by Sys
                        next_state[curr_state_exp.index(d_status)] = 'sys'
                
                door_statuses: ADD = reduce(lambda a, b: a & b, [self.dVar_map_sym[idx][e] for idx, e in enumerate(next_state[3:-1])])

                return self.convert_exlpicit_state_to_cube(next_state[:3]) & door_statuses & ~self.eVar[0], 'ENV_STAY', next_state  # return the STAY action for invalid Env action
            
            nxt_x = curr_state_exp[2][0] + Moves[act_name].value[0]
            nxt_y = curr_state_exp[2][1] + Moves[act_name].value[1]
            next_state[0] = 'sys'  # switch turn after sys move 
            next_state[2] = [nxt_x, nxt_y]  # env pos
            # TODO: Hardcoding the start of index. Need to change when we have more agents
            for d_status in curr_state_exp[3:]:
                # During Env's turn, we check if the Sys player state is passing through the door
                if d_status == 'unclaimed' and (curr_state_exp[1][0], curr_state_exp[1][1]) in self.grid['door']:
                    # if the door is unclaimed and the Sys is passing through it, then the door becomes claimed by Sys
                    next_state[curr_state_exp.index(d_status)] = 'sys'
                
                door_statuses: ADD = reduce(lambda a, b: a & b, [self.dVar_map_sym[idx][e] for idx, e in enumerate(next_state[3:-1])])

            return self.convert_exlpicit_state_to_cube(next_state[:3]) & door_statuses & ~self.eVar[0], act, next_state
    
    


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
        eidx = self.manager.addVariables().index(self.eVar[0])
        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + self.eVar + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds])
        dConf_exist_cube = dict({})
        for didx in range(len(self.dVars)):
            if len(self.grid['door']) == 1:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVar + self.eVar +  self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds])
            else:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.dVars[:didx] + self.dVars[didx+1:])
        
        xConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])
        
        yConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])

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
            
            dCube_str = []
            for e in dConf_exist_cube.values():
                dCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                row_states = [self.xVar_map[pidx].inv[e] for pidx, e in enumerate(xCube_str)]
            except KeyError:
                continue

            try:
                column_states = [self.yVar_map[pidx].inv[e] for pidx, e in enumerate(yCube_str)]
            except KeyError:
                continue
            
            try:
                door_states = [self.dVar_map[didx].inv[e] for didx, e in enumerate(dCube_str)]
            except KeyError:
                continue

            try:
                pos = []
                for r, c in zip(row_states, column_states):
                    pos.append([r, c])
                eVar_state = self.eVar_map.inv[cube.bddPattern().cubeString()[eidx].replace('-', '')]
                state = [self.tVar_map.inv[tConf_cube_str]] + pos + door_states + [eVar_state]
                states_action_pairs.append([((self.tVar_map.inv[tConf_cube_str], *pos, *door_states, eVar_state), val), None])
            except KeyError:
                continue
            
            if action:
                try:
                    if self.tVar_map.inv[tConf_cube_str] == 'sys':
                        rCube_str = cube.bddPattern().cubeString()[start_rvar_idx:end_rvar_idx + 1].replace('-', '')
                        action_str = self.sys_action_map.inv[rCube_str]
                    elif self.tVar_map.inv[tConf_cube_str] == 'env':
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
    

    def test_preimage(self):
        sys_pos = (1, 2)
        goal_cube_sys = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (0, 2)
        goal_cube_env = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        goal_cube = self.tVar_map_sym['env'] & goal_cube_sys & goal_cube_env & self.dVar_map_sym[0]['env']
        print('Goal state:', goal_cube)
        self.convert_cube_to_state_ADD(goal_cube, state_flag=True, action=False, verbose=True)
        
        # compute preimage 
        From = goal_cube.swapVariables(self.latches, self.prime_latches)
        preimage = From.vectorCompose(self.prime_latches, list(self.transition_relation.values()))
        print('Preimage of goal state:', preimage)
        self.convert_cube_to_state_ADD(preimage, state_flag=True, action=True, verbose=True)

        preimage = preimage.ite(self.manager.addZero(), self.manager.plusInfinity())
        preimage = preimage + self.weight

        # now let takes min and max
        new_preimage = self.compute_min_max_preimage(preimage,
                                                     valid_env_action_mask=reduce(lambda x, y: x | y, self.env_action_cube_list))
        # print('Preimage after min max abstraction:', new_preimage)
        self.convert_cube_to_state_ADD(new_preimage, state_flag=True, action=False, verbose=True)
