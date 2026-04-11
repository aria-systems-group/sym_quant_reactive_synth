from functools import reduce
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

from cudd import ADD

from bidict import bidict
from tabulate import tabulate

from src.compositional_graphs.gridworld.gridworld_dynamic import CELL
from src.compositional_graphs.gridworld.gridworld_dynamic_doors_dfa_game import GridWorldDynamicDoorsDFAGame
from src.compositional_graphs.gridworld.gridworld_dynamic_regret import GridWorldDynamicRegretGame



class GridWorldDynamicDoorsRegretGame(GridWorldDynamicRegretGame, GridWorldDynamicDoorsDFAGame):
    def __init__(self, 
                 rows: int, columns: int,
                 init: List[CELL], goal: List[CELL],
                 formula: str, budget: int,
                 grid: Optional[Dict['str', List[CELL]]] = dict({}),
                 players: Dict[str, int] = {'sys': 1, 'env': 1},
                 restricted_env_locs: Optional[List[CELL]] = [],
                 camera: bool = False,
                 ltlf_flag: bool = True,
                 cooperative_game: bool = False,
                 enable_reordering: bool = False,
                 **kwargs):
        super().__init__(rows=rows, columns=columns,
                         formula=formula, init=init,
                         goal=goal, grid=grid, camera=camera,
                         ltlf_flag=ltlf_flag, players=players,
                         budget=budget,
                         cooperative_game=cooperative_game,
                         restricted_env_locs=restricted_env_locs,
                         enable_reordering=False)
        # call it 3rd time here to override the base method
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl & self.all_door_uncalimed & self.not_error_state_cube
        if enable_reordering:
            self.manager.autodynEnable()
    

    def create_all_boolean_state_vars_and_maps(self):
        self.tVars = self.create_player_latches()
        self.xVars, self.yVars = self.create_latches()
        self.eVars = self.create_error_latches()
        self.dVars = self.create_door_vars()
        self.lbls_list = [ob for ob in self.grid.keys() if ob not in self.obstacles] + ['c'] + (['p'] if self.camera else [])
        self.lVars = self.create_state_lbls_vars()
        self.uVars = self.create_utility_latches()
        self.uVars_bdd = [u.bddPattern() for u in self.uVars]

        self.create_xVar_map()
        self.create_yVar_map()
        self.create_tVar_map()
        self.create_eVar_map()
        self.create_lVars_map()
        self.create_uVar_map()
        self.create_dVar_map(prime=False)
        self.all_door_uncalimed = reduce(lambda a, b: a & b, [self.dVar_map_sym[d_idx]['unclaimed'] for d_idx in range(len(self.grid['door']))])
        self.create_symbolic_maps(prime=False)
        self.state_lbl_map_sym: Dict[CELL, ADD] = defaultdict(lambda: reduce(lambda x, y: x & y, [~e for e in self.lVars]))
        self.lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        self.create_state_lbl_map()
        # create the dfa state variables and maps
        self.create_dfa_latches_and_maps()


    def create_all_prime_boolean_state_vars_and_maps(self):
        self.prime_tVars = self.create_prime_player_latches()
        self.prime_xVars, self.prime_yVars = self.create_prime_latches()
        self.prime_eVars = self.create_prime_error_vars()
        self.prime_dVar_map_sym = {d: bidict({}) for d in range(len(self.grid['door']))}
        self.prime_dVars = self.create_prime_door_vars()
        self.prime_lVars = self.create_prime_state_lbls_vars()
        self.prime_lVar_map_sym = {lbl: cube.swapVariables(self.lVars, self.prime_lVars) for lbl, cube in self.lVar_map_sym.items()}
        self.prime_uVars = self.create_prime_utility_latches()
        self.prime_uVars_bdd = [u.bddPattern() for u in self.prime_uVars]
        self.create_symbolic_maps(prime=True)
        self.create_dVar_map(prime=True)

        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
        self.prime_qVars_bdd = [var.bddPattern() for var in self.prime_qVars]
    
    def get_number_of_states(self, verbose: bool = True) -> Tuple[int, int]:
        total_dfa_game_state = super().get_number_of_states(verbose=verbose)
        total_regret_game_state = total_dfa_game_state * ((self.budget + 2)**2)
        if verbose:
            print(f'Number of States in Regret Game: {total_regret_game_state:,}')
        return total_regret_game_state
    
    
    def gou_convert_cube_to_state_ADD(self,
                                      dd: ADD, state_flag: bool = True,
                                      lbl_flag: bool = False, dfa_flag: bool = True,
                                      action: bool = False, verbose: bool = False,
                                      table_header: bool = True, print_val: bool = True) -> None:
        """
        Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True and state lbl flag is set to False.
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches + self.uVars) # includes uVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)
        if not lbl_flag:
            for lbl_var in self.lVars:
                relevant_vars.remove(lbl_var)
        
        headers = []
        if verbose:
            headers.append('state')
        if lbl_flag:
            headers.append('labels')
        if action:
            headers.append('action')
        if print_val:
            headers.append('value')
        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        flattened_dVars = [var for dVar_adds in self.dVars for var in dVar_adds] 

        # the next vars are l' vars - we ignore them for now. The next ones are action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.eVars + self.rVars + self.uVars + flattened_dVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.uVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVars + self.lVars + self.rVars + flattened_dVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        
        dConf_exist_cube = dict({})
        for didx in range(len(self.dVars)):
            if len(self.grid['door']) == 1:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars)
            else:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.dVars[:didx] + self.dVars[didx+1:])
        
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + flattened_dVars + self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + flattened_dVars + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = []
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_cube_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
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
                uVar_state = self.uVar_map.inv[uConf_cube_str]
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos + door_states + [sys_err_state, env_err_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state)
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos, *door_states, sys_err_state, env_err_state), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), val), None])
                
            except KeyError:
                continue
        
            # print the robot and human actions as well
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
            
            if lbl_flag:
                try:
                    lbl_list = []
                    for lvar in self.lVars:
                        lbl_idx = self.manager.addVariables().index(lvar)
                        if cube.bddPattern().cubeString()[lbl_idx].replace('-', '') == '1':
                            lbl_list.append(self.lVar_map.inv[lvar.bddPattern().__str__()])
                except KeyError:
                    continue
            
            row = []
            if verbose:
                row.append(state)
            if lbl_flag:
                row.append(lbl_list)
            if action:
                row.append(action_str)
            if print_val:
                row.append(val)
            states_bookkeeping.append(tuple(row))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))
        
        return states_action_pairs
    

    def gobr_convert_cube_to_state_ADD(self,
                                       dd: ADD, state_flag: bool = True,
                                       lbl_flag: bool = False, dfa_flag: bool = True,
                                       action: bool = False, verbose: bool = False,
                                       table_header: bool = True, print_val: bool = True) -> None:
        """
        Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True and state lbl flag is set to False.
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches + self.uVars + self.brVars) # includes uVars, brVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)
        if not lbl_flag:
            for lbl_var in self.lVars:
                relevant_vars.remove(lbl_var)
        
        headers = []
        if verbose:
            headers.append('state')
        if lbl_flag:
            headers.append('labels')
        if action:
            headers.append('action')
        if print_val:
            headers.append('value')
        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        flattened_dVars = [var for dVar_adds in self.dVars for var in dVar_adds]
        
        # the next vars are l' vars - we ignore them for now. The next ones are action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.eVars + self.rVars + flattened_dVars + self.uVars + self.brVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.uVars + self.brVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVars + self.lVars + self.rVars + flattened_dVars + self.brVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        brConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVars + self.lVars + self.rVars + flattened_dVars + self.uVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        
        dConf_exist_cube = dict({})
        for didx in range(len(self.dVars)):
            if len(self.grid['door']) == 1:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.lVars + self.uVars + self.brVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars)
            else:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.lVars + self.uVars + self.brVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.dVars[:didx] + self.dVars[didx+1:])

        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + flattened_dVars + self.brVars +  self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + flattened_dVars + self.brVars + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = []
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_cube_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            brConf_cube_str = cube.existAbstract(brConf_exist_cube).bddPattern().cubeString().replace('-', '')
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
                uVar_state = self.uVar_map.inv[uConf_cube_str]
                brVar_state = self.brVar_map.inv[brConf_cube_str]
                state = ((([self.tVar_map.inv[tConf_cube_str]] + pos + door_states + [sys_err_state, env_err_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), brVar_state)
                states_action_pairs.append([((((self.tVar_map.inv[tConf_cube_str], *pos, *door_states, sys_err_state, env_err_state), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), brVar_state), val), None])
                
            except KeyError:
                continue
        
            # print the robot and human actions as well
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
            
            if lbl_flag:
                try:
                    lbl_list = []
                    for lvar in self.lVars:
                        lbl_idx = self.manager.addVariables().index(lvar)
                        if cube.bddPattern().cubeString()[lbl_idx].replace('-', '') == '1':
                            lbl_list.append(self.lVar_map.inv[lvar.bddPattern().__str__()])
                except KeyError:
                    continue
            
            row = []
            if verbose:
                row.append(state)
            if lbl_flag:
                row.append(lbl_list)
            if action:
                row.append(action_str)
            if print_val:
                row.append(val)
            states_bookkeeping.append(tuple(row))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))
        
        return states_action_pairs


    