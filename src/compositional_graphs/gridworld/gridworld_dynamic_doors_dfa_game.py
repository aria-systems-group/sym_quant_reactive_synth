import itertools

from bidict import bidict
from tabulate import tabulate

from functools import reduce
from typing import List, Union, Optional, Dict, Tuple, Set

from cudd import ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic import CELL
from src.compositional_graphs.gridworld.gridworld_dynamic_doors import GridWorldDynamicDoorsGame
from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame


class GridWorldDynamicDoorsDFAGame(GridWorldDynamicDFAGame, GridWorldDynamicDoorsGame):
    def __init__(self, 
                 rows: int, columns: int,
                 formula: str, 
                 init: List[CELL], goal: List[CELL],
                 grid: Dict[str, List[CELL]],
                 restricted_env_locs: Optional[List[CELL]] = [],
                 camera: bool = False,
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False):
        super().__init__(rows=rows, columns=columns,
                         formula=formula, init=init,
                         goal=goal, grid=grid, camera=camera,
                         ltlf_flag=ltlf_flag,
                         restricted_env_locs=restricted_env_locs,
                         enable_reordering=False)
        # call it 3rd time here to override the base method
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl & self.all_door_uncalimed & ~self.eVar[0]
        if enable_reordering:
            self.manager.autodynEnable()
    

    def add_lbl_evolution_to_TR(self):
        """
         A function that add the state lbl evolution to the existing the TR. We override the base method to incorporate door vars (dVars) from the door game.
        """
        game_latch: List[ADD] = self.tVar + self.eVar + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds]
        game_prime_latch: List[ADD] = self.prime_tVar + self.prime_eVar + [var for prime_xVar_adds in self.prime_xVars for var in prime_xVar_adds] + [var for prime_yVar_adds in self.prime_yVars for var in prime_yVar_adds] + [var for prime_dVar_adds in self.prime_dVars for var in prime_dVar_adds]
        
        primed_state = self.state_lbl.swapVariables(game_latch, game_prime_latch)
        pre_state_nxt_lbl = primed_state.vectorCompose(game_prime_latch, list(self.transition_relation.values())[:-len(self.lVars)])
        
        for cube_string in itertools.product([0, 1], repeat=len(self.lVars)):
            lbl_cube = reduce(lambda a,b: a & b, [self.lVars[idx] if bit else ~self.lVars[idx] for idx, bit in enumerate(cube_string)])
            pre_state_action: ADD = pre_state_nxt_lbl.restrict(lbl_cube)
            for idx, prime_lVar in enumerate(cube_string):
                if prime_lVar == 1:
                    self.transition_relation[self.lVars[idx].bddPattern().__str__()] |= pre_state_action & self.state_lbl
        
        # iterate through the tVars, xVars, yVars and add state lbls to all cubes
        for k in self.transition_relation.keys():
            # if we skip the lable vars in the TR as they are taken care of by the above code.
            if k.startswith('l'):
                continue
            self.transition_relation[k] &= self.state_lbl


    def convert_cube_to_state_ADD(self,
                                  dd: ADD,
                                  state_flag: bool = True, dfa_flag: bool = True,
                                  action: bool = False, lbl_flag: bool = False,
                                  verbose: bool = False, table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
        Convert a cube to a state representation. Override DFA Game's method to include door status in the state representation.
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches)  # includes tVars, xVars, yVars, lVars, dVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars
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
        if verbose:
            headers.append('value')
        
        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])
        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + self.eVar + self.lVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.rVars)
        eidx = self.manager.addVariables().index(self.eVar[0])
        
        dConf_exist_cube = dict({})
        for didx in range(len(self.dVars)):
            if len(self.grid['door']) == 1:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars)
            else:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.dVars[:didx] + self.dVars[didx+1:])
        
        xConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])
        
        yConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.eVar + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])

        # print the states
        states_action_pairs = []
        states_bookkeeping = [] 
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
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
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos + door_states + [eVar_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str])
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos, *door_states, eVar_state), self.dfa_handle.qVar_map.inv[qConf_cube_str]), val), None])
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
            if verbose:
                row.append(val)
            states_bookkeeping.append(tuple(row))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))
        
        return states_action_pairs
