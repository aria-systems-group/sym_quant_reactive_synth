import math

from bidict import bidict
from tabulate import tabulate

from functools import reduce
from typing import List, Tuple, Dict, Optional

from cudd import ADD

from src.compositional_graphs.gridworld.gridworld_dynamic_no_prime import GridWorldDynamicGameNoPrime, Moves, CELL


class GridWorldDynamicDoorsGameNoPrime(GridWorldDynamicGameNoPrime):
    def __init__(self,
                 rows: int, columns: int,
                 init: List[Tuple[int, int]], goal: List[Tuple[int, int]],
                 grid: Dict[str, List[Tuple[int, int]]],
                 players: Dict[str, int] = {'sys': 1, 'env': 1},
                 restricted_env_locs: Optional[List[CELL]] = None,
                 cooperative_game: bool = False,
                 enable_reordering=False, **kwargs):
        """
         Inherit the Gridworld Dyanmic Game and augment it with doors.  
        """
        assert isinstance(grid['door'], list) and len(grid['door']) > 0, "If you are including door in the grid dict, make sure to provide a non-empty list of door locations."
        self._door_status = ['unclaimed', 'sys', 'env']
        self.dVar_map = {d: bidict({}) for d in range(len(grid['door']))}
        self.dVar_map_sym = {d: bidict({}) for d in range(len(grid['door']))}
        super().__init__(rows=rows, columns=columns,
                         init=init, goal=goal,
                         grid=grid, players=players,
                         cooperative_game=cooperative_game,
                         restricted_env_locs=restricted_env_locs,
                         enable_reordering=enable_reordering)

    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()    
        self.dVars = self.create_door_vars()
        self.create_dVar_map()
        self.all_door_uncalimed = reduce(lambda a, b: a & b, [self.dVar_map_sym[d_idx]['unclaimed'] for d_idx in range(len(self.grid['door']))])
    

    def create_door_vars(self) -> List[List[ADD]]:
        door_vars = math.ceil(math.log2(len(self._door_status)))
        dVars: List[List[ADD]] = [] 
        for d_idx in range(len(self.grid['door'])):
            varsize = self.manager.size()
            dVars.append([self.manager.addVar(s + varsize, f'd{d_idx}{s}') for s in range(door_vars)])
        return dVars

    
    def create_dVar_map(self) -> None:
        for d_idx in range(len(self.grid['door'])):
            for s_idx, d_status in enumerate(self._door_status):
                bit_str = f"{s_idx:0{len(self.dVars[d_idx])}b}"
                self.dVar_map[d_idx][d_status] = bit_str
                self.dVar_map_sym[d_idx][d_status] = self.cube_to_add(bit_str, self.dVars[d_idx])
    

    def set_latches(self):
        super().set_latches()
        self.latches += [var for dVar_adds in self.dVars for var in dVar_adds]
        self.latches_bdd += [var.bddPattern() for dVar_adds in self.dVars for var in dVar_adds]
    

    def set_init_latch(self) -> ADD:
        return super().set_init_latch() & self.all_door_uncalimed
    

    def get_number_of_states(self, verbose: bool = True) -> Tuple[int, int]:
        """
        A method to to compute the |Sys States| and |Env states| in the game.
         Sys States = rows x columns x |door_status|^number_of_doors
         Env States = rows x columns x |door_status|^number_of_doors
         We have two copies of every states - one that belong to sys error and one belongs to enc error. 
         So we need to multiply by 2 to account for that.
        """
        sys_states = 2 * ((self.rows * self.columns) ** self.players['sys'])  * (len(self._door_status) ** len(self.grid['door']))
        env_states = 2 * ((self.rows * self.columns) ** self.players['env'])  * (len(self._door_status) ** len(self.grid['door']))
        if verbose:
            print(f'Number of States in Game: \n Sys States: {sys_states:,} \n Env States: {env_states:,} \n Total States: {sys_states + env_states:,}')
        return sys_states, env_states
    

    def log_game_details(self) -> Dict[str, int]:
        sys_states, env_states = self.get_number_of_states(False)
        try:
            num_opt_svals = self.comp_winning_states.countLeaves()
        except (AttributeError, TypeError):
            num_opt_svals = math.inf
        abs_dict = {
            'total_latches': len(self.latches) + len(self.rVars),
            'latches': len(self.latches),
            'action_vars': len(self.rVars),
            'turn_vars': len(self.tVars),
            'error_vars': len(self.eVars),
            'door_vars': sum([len(dVars) for dVars in self.dVars]),
            'xVars': sum([len(player_xVars) for player_xVars in self.xVars]),
            'yVars': sum([len(player_yVars) for player_yVars in self.yVars]),
            'total_states': sys_states + env_states,
            'sys_states': sys_states,
            'env_states': env_states,
            'num_opt_sVals': num_opt_svals
            }
        return abs_dict

    
    def get_door_constraint(self, cell: CELL, player: str) -> ADD:
        """
        Given the grid cell, this method looks up if there is a door in the cell. If yes, it returns the corresponding constraint
         that encodes that the door must be open for that player's coalition (Sys or Env) to pass through.
        """
        door_constraint = self.manager.addZero()
        if cell in self.grid['door']:
            # we have a door constraint for this cell
            d_idx: int = self.grid['door'].index(cell)
            not_door_owner = 'env' if player.startswith('sys') else 'sys'
            return ~self.dVar_map_sym[d_idx][not_door_owner]
        
        return ~door_constraint


    def create_actions(self, player: str):
        """
        Override base class method to create transition relation with door constraints. 
        The main difference is that when the player is trying to pass through a door cell, we need to check if the door is open for the player or not. If not, then the player cannot pass through and must stay in the same cell.
        """
        turn_bit: ADD = self.tVar_map_sym[player]
        p_idx = self.pidx_to_pstr.inv[player]
        for d_idx in range(len(self.grid['door'])):
            for d_status in self._door_status:
                dConf_cube: ADD = self.dVar_map_sym[d_idx][d_status]
                
                for r in range(self.rows):
                    rVar_add: ADD = self.xVar_map_sym[p_idx][r]

                    for c in range(self.columns):
                        cVar_add: ADD = self.yVar_map_sym[p_idx][c]
                        
                        # get valid acts for grid position (r, c) - this does check for wall or other obstacles in the successor step.
                        valid_actions = self.get_valid_transitions(rPos=r, cPos=c, player=player)
                        invalid_actions = set(self.env_actions) - valid_actions if player.startswith('env') else set(self.sys_actions) - valid_actions

                        if len(invalid_actions) > 0:
                            # invalid action must be mapped to an error state
                            invalid_act_cube = reduce(lambda x, y: x | y, [self.action_map_sym[f'{player}_{act}'] for act in invalid_actions])
                            tr_key = self.sys_error_cube.bddPattern().__str__() if player.startswith('sys') else self.env_error_cube.bddPattern().__str__()
                            self.transition_relation[tr_key] |= turn_bit & dConf_cube & rVar_add & cVar_add & self.not_error_state_cube & invalid_act_cube & ~self.obsatcle_constraint_cube

                        for act in valid_actions:
                            act_cube: str = self.action_map_sym[f'{player}_{act}']
                            nxt_rPos = r + Moves[act].value[0]
                            nxt_cPos = c + Moves[act].value[1]

                            # check if the next position is valid or not - only for Env player
                            if player.startswith('env') and (self.obsatcle_constraint_cube & self.tVar_map_sym[player] & self.xVar_map_sym[p_idx][nxt_rPos] & self.yVar_map_sym[p_idx][nxt_cPos]) != self.manager.addZero():
                                # invalid Env action must be mapped to error state
                                self.transition_relation[self.env_error_cube.bddPattern().__str__()] |= turn_bit & dConf_cube & rVar_add & cVar_add & act_cube & self.not_error_state_cube & ~self.obsatcle_constraint_cube
                                continue
                            
                            transition_cube: ADD = turn_bit & dConf_cube & rVar_add & cVar_add & act_cube & ~self.obsatcle_constraint_cube

                            if (r, c) == self.grid['door'][d_idx]:
                                transition_cube &= self.get_door_constraint(cell=(r, c), player=player)
                            
                            if transition_cube.isZero():
                                continue

                            for idx, prime_rVar in enumerate(self.xVar_map[p_idx][nxt_rPos]):
                                if prime_rVar == '1':
                                    self.transition_relation[self.xVars[p_idx][idx].bddPattern().__str__()] |= transition_cube

                            for idx, prime_rVar in enumerate(self.yVar_map[p_idx][nxt_cPos]):
                                if prime_rVar == '1':
                                    self.transition_relation[self.yVars[p_idx][idx].bddPattern().__str__()] |= transition_cube
                            
                            # updated door status
                            if d_status == 'unclaimed' and (nxt_rPos, nxt_cPos) == self.grid['door'][d_idx]:
                                # if the door is unclaimed and the player is passing through it, then the door becomes claimed by the player
                                _door_player = 'sys' if player.startswith('sys') else 'env'
                                for idx, prime_dVar in enumerate(self.dVar_map[d_idx][_door_player]):
                                    if prime_dVar == '1':
                                        self.transition_relation[self.dVars[d_idx][idx].bddPattern().__str__()] |= transition_cube
                            else:
                                # if the door is already unclaimed or already claimed by the same player, then the door status does not change
                                for idx, prime_dVar in enumerate(self.dVar_map[d_idx][d_status]):
                                    if prime_dVar == '1':
                                        self.transition_relation[self.dVars[d_idx][idx].bddPattern().__str__()] |= transition_cube
    
    def add_frame_axioms(self):
        for active_player in self.tVar_map.keys():
            # all other player are passive players
            for passive_player in self.tVar_map.keys():
                if active_player == passive_player:
                    continue
                turn_bit: ADD = self.tVar_map_sym[active_player]
                pidx = self.pidx_to_pstr.inv[passive_player]
                pstr = 'sys' if passive_player.startswith('sys') else 'env'
                for d_idx in range(len(self.grid['door'])):
                    for d_status in self._door_status:
                        dConf_cube: ADD = self.dVar_map_sym[d_idx][d_status]
                
                        for rPos in range(self.rows):
                            rVar_add = self.xVar_map_sym[pidx][rPos]
                            for cPos in range(self.columns):
                                cVar_add = self.yVar_map_sym[pidx][cPos]
                                act_cube: dict = self.sys_action_cube[active_player] if active_player.startswith('sys') else self.env_action_cube[active_player]
                                transition_cube: ADD = turn_bit & rVar_add & cVar_add & act_cube & self.not_error_state_cube & ~self.obsatcle_constraint_cube & dConf_cube & self.get_door_constraint(cell=(rPos, cPos), player=pstr)

                                for idx, prime_rVar in enumerate(self.xVar_map[pidx][rPos]):
                                    if prime_rVar == '1':
                                        self.transition_relation[self.xVars[pidx][idx].bddPattern().__str__()] |= transition_cube
                                
                                for idx, prime_rVar in enumerate(self.yVar_map[pidx][cPos]):
                                    if prime_rVar == '1':
                                        self.transition_relation[self.yVars[pidx][idx].bddPattern().__str__()] |= transition_cube


    def get_next_state(self, curr_state_exp: Tuple, act: str) -> Tuple[ADD, str, Tuple]:
        """
        Here we also update the door status. We check if a player is passing through a door or not. 
         If yes, then we need to update the door status accordingly. 
        
        NOTE: We update the door status, when the player is at the door cell.
        """
        next_state_sym, act, next_state = super().get_next_state(curr_state_exp=curr_state_exp, act=act)
        next_state_door = [i for i in curr_state_exp[self.total_players + 1:-2]]
        for di, d_status in enumerate(next_state_door):
            if d_status != 'unclaimed':
                continue
            # + 1 is to taccount for the player Var at index 0. The player Var is followed by the position vars for each player and then the door status vars.
            for pidx, pstr in self.pidx_to_pstr.items():
                if (next_state[pidx + 1][0], next_state[pidx + 1][1]) in self.grid['door']:
                    # if the door is unclaimed and the Sys is passing through it, then the door becomes claimed by Sys
                    next_state_door[di] = 'sys' if pstr.startswith('sys') else 'env'
        
        door_statuses: ADD = reduce(lambda a, b: a & b, [self.dVar_map_sym[idx][e] for idx, e in enumerate(next_state_door)])
        
        return next_state_sym & door_statuses, act, next_state[:-2] + next_state_door + list(curr_state_exp[-2:])


    def post_process_transition_relation(self, debug: bool = False):
        super().post_process_transition_relation(debug=debug)
        # now we remove states wwhere the player is at door but the door belongs another team
        bad_state_acts = self.manager.addZero()
        for didx, (dx, dy) in enumerate(self.grid['door']):
            for d_status in self._door_status:
                # if d_status is env then sys player cannot be at this door cell.
                if d_status == 'env':
                    for pidx in range(self.players['sys']):
                        sys_player_at_door = self.xVar_map_sym[pidx][dx] & self.yVar_map_sym[pidx][dy] & self.not_error_state_cube
                        # bad_state_acts |= super().compute_preimage(sys_player_at_door & self.dVar_map_sym[didx][d_status])
                        bad_state_acts |= (sys_player_at_door & self.dVar_map_sym[didx][d_status]).vectorCompose(self.latches, list(self.transition_relation.values()))
                elif d_status == 'sys':
                    idx_offset = self.players['sys']
                    env_player_at_door = reduce(lambda a, b: a | b, [self.xVar_map_sym[pidx + idx_offset][dx] & self.yVar_map_sym[pidx + idx_offset][dy] & self.not_error_state_cube for pidx in range(self.players['env'])])
                    # bad_state_acts |= super().compute_preimage(env_player_at_door & self.dVar_map_sym[didx][d_status])
                    bad_state_acts |= (env_player_at_door & self.dVar_map_sym[didx][d_status]).vectorCompose(self.latches, list(self.transition_relation.values()))
        
        # remove bad state action pairs
        for var in self.transition_relation.keys():
            self.transition_relation[var] &= ~bad_state_acts
        
        # must map invalid env action to error state
        self.transition_relation[self.env_error_cube.bddPattern().__str__()] |= bad_state_acts & self.env_tVar_cube & self.not_error_state_cube
        
                    

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
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + self.eVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds])
        dConf_exist_cube = dict({})
        for didx in range(len(self.dVars)):
            if len(self.grid['door']) == 1:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars +  self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds])
            else:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.dVars[:didx] + self.dVars[didx+1:])
        
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])
        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])

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
                sys_err_state = cube.bddPattern().cubeString()[sys_eidx].replace('-', '')
                env_err_state = cube.bddPattern().cubeString()[env_eidx].replace('-', '')
                state = [self.tVar_map.inv[tConf_cube_str]] + pos + door_states + [sys_err_state, env_err_state]
                states_action_pairs.append([((self.tVar_map.inv[tConf_cube_str], *pos, *door_states, sys_err_state, env_err_state), val), None])
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