import math

from functools import reduce

from bidict import bidict
from tabulate import tabulate

from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple, Set

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame, Moves
from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot


CELL = Tuple[int, int]


class GridWorldDynamicDFAGame(GridWorldDynamicGame):
    
    def __init__(self,
                 rows: int, columns: int,
                 init: List[tuple], goal: tuple,
                 formula: str, 
                 grid: Optional[Dict['str', List[CELL]]] = dict({}),
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False):
        """
        Initializes the GridWorldDynamicDFAGame with the given parameters. 
         
        This class extends GridWorldDynamicGame by construcitng a DFA based on the provided LTLf formula using either Mona or SPOT.
         We then play the game on the symbolic product.

        Args:
            rows (int): Number of rows in the grid.
            columns (int): Number of columns in the grid.
            init (tuple): Initial state configuration.
            goal (tuple): Goal state configuration.
            ltlf_flag (bool): Flag indicating whether the formula is LTLf (True) or LTL (False).
            formula (str): The LTL/LTLf formula specifying the objective of the game.
        """
        self.formula: str = formula
        self.qVars: List[ADD] = []
        self.qVars_bdd: List[BDD] = []
        self.qVar_map: List[ADD] = {} 
        self.qVar_map_sym: List[ADD] = {} 
        self.ltlf_flag: bool = ltlf_flag
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(rows=rows, columns=columns, init=init, goal=goal, grid=grid, enable_reordering=False)

        # set up dfa init and goal states
        self.dfa_handle.set_init_latch()
        self.dfa_handle.set_goal_latch()
        # call it 2nd time here to override the base method - is this the best way?
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch
        self.goal_latch: ADD = self.set_goal_latch()
        # self.create_state_lbls()

        # now we create the TR for the dfa
        self.dfa_handle.game_latches = self.latches

        # by default variable reordering is disabled for DFA games - to check for computation time without this optimization
        # however, switching variable ordering makes the code faster.
        if enable_reordering:
            self.manager.autodynEnable()


    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()

        self.state_lbl_map: Dict[CELL, Set[str]] = defaultdict(lambda: set('e'))
        self.lVar_map = bidict({})
        self.lVars = self.create_state_lbls_vars()
        self.lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        self.create_lVars_map()
        self.lVar_map_sym = bidict({k: self.cube_to_add(v, self.lVars) for k, v in self.lVar_map.items()})
        self.state_lbl_map_sym: Dict[CELL, ADD] = defaultdict(lambda: self.lVar_map_sym['e'])
        self.create_explicit_state_lbl_map()

        # create the dfa state variables and maps
        self.create_dfa_latches_and_maps()
    

    def create_all_prime_boolean_state_vars_and_maps(self):
        """
         The main method that creates all primed version of the boolean variables for the FrankaDynamic Turn-Based Game.
          1. prime turn variables - tVars
          2. prime ratio variables - kVars
          3. prime predicate variables - pVars
          4. prime box predicate variables - bVars
        """
        super().create_all_prime_boolean_state_vars_and_maps()
        self.prime_lVars = self.create_prime_state_lbls_vars()
        self.prime_lVar_map_sym = {lbl: self.cube_to_add(cube_str, self.prime_lVars) for lbl, cube_str in self.lVar_map.items()}
        
        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
        self.prime_qVars_bdd = [var.bddPattern() for var in self.prime_qVars]
    

    def set_latches(self):
        super().set_latches()
        self.latches.extend(self.lVars)
    

    def set_prime_latches(self):
        super().set_prime_latches()
        self.prime_latches.extend(self.prime_lVars)
    

    def set_goal_latch(self):
        return self.dfa_handle.goal_latch


    def create_explicit_state_lbl_map(self):
        """
         A method that parse the grid dictionary construction a diction that maps every cell to a set of AP.
        """
        for ap, loc in self.grid.items():
            for cell in loc:
                self.state_lbl_map[cell].add(ap)
                self.state_lbl_map_sym[cell] |= self.lVar_map_sym[ap]


    def create_state_lbls_vars(self) -> List[ADD]:
        """
         A method to create state labels for the gridworld. This is used for labeling states with propositions for LTL synthesis. 
        """
        # collision ap is always part of the state lbl
        varsize = self.manager.size()
        num_of_lbls =  len(self.grid.keys()) + 3 # for collision ap and empty ap and offset by 1 for 0-bit vector.
        lVars_size = math.ceil(math.log2(num_of_lbls))
        lVars: List[ADD] =  [self.manager.addVar(l + varsize , f'l{l}') for l in range(lVars_size)]
        return lVars
    
    def create_prime_state_lbls_vars(self) -> List[ADD]:
        """
         A method to create prime version of the state label variables. This is used for labeling next states with propositions for LTL synthesis. 
        """
        varsize = self.manager.size()
        prime_lVars: List[ADD] =  [self.manager.addVar(l + varsize , f"pl{l}") for l in range(len(self.lVars))]
        return prime_lVars
    
    
    def create_lVars_map(self):
        # collision ap is always the first label - offset by 1
        self.lVar_map['e'] = f"{1:0{len(self.lVars)}b}"  # empty ap
        self.lVar_map['c'] = f"{2:0{len(self.lVars)}b}"  # collision ap
        for ap_idx, ap in enumerate(self.grid.keys()):
            bit_str = f"{ap_idx + 3:0{len(self.lVars)}b}"
            self.lVar_map[ap] = bit_str
    

    def create_state_lbls(self, debug: bool = False):
        self.state_lbl: ADD = self.manager.addZero()
        self.state_lbl |= self.valid_state_constraint & self.lVar_map_sym['e']  # start with empty ap
        for r in range(self.rows):
            for c in range(self.columns):
                self.state_lbl |= reduce(lambda a, b: a & b, [self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] for p in range(2)]) & self.lVar_map_sym['c']
                
                # can optimize this by using the obstavle constraint cube that we already have
                if (r, c) in self.grid.get('wall', []):
                    # TODO: hard coding for two agents: update for all agents in the future
                    for p in range(2):
                        self.state_lbl |= self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] & self.lVar_map_sym['wall']
                
                if (r, c) in self.grid.get('lava', []):
                    # TODO: hard coding for two agents: update for all agents in the future
                    for p in range(2):
                        self.state_lbl |= self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] & self.lVar_map_sym['lava']
                
                if (r, c) in self.grid.get('goal', []):
                    # TODO: hard coding for 1 Sys agent: update for multiple agents in the future
                    # for p in range(2):
                    self.state_lbl |= self.xVar_map_sym[0][r] & self.yVar_map_sym[0][c] & self.lVar_map_sym['goal']
    
        # lets try this
        # all_lbls_wo_empty = reduce(lambda a, b: a | b, self.lVar_map_sym.values()) & ~self.lVar_map_sym['e']

        # now lets take preimage to get previous states
        # test = super().compute_preimage(self.valid_state_constraint & all_lbls_wo_empty)
        self.state_lbl = super().compute_preimage(self.state_lbl)
        # need to remove dependecy on action vars
        # state_lbl_bdd = self.state_lbl.bddPattern().existAbstract(self.rVars_cube.bddPattern())
        # self.state_lbl = state_lbl_bdd.toADD()
        if debug:
            print("--- State labels after preimage computation ---")
            self.convert_cube_to_state_ADD(self.state_lbl , state_flag=True, dfa_flag=True, lbl_flag=True, action=True, verbose=True)
        


    

    def create_dfa_latches_and_maps(self):
        """
        Create DFA latches and their symbolic maps based on the provided LTL/LTLf formula. Unloike the manipulator domain,
         the gridworld domain does not have spearate prime variables for state lbls as we are do compose over these variables. 

        This menas, the monolithic DFA TR will be over qVars, lVars, prime_qVars.
        """
        # here we only create the variables and maps for the dfa
        if self.ltlf_flag:
            dfa_handle = SymbolicPartitionedDFAFromMona(formula=self.formula,
                                                        manager=self.manager,
                                                        latches_map=self.lVar_map_sym,
                                                        domain='gridworld',
                                                        game_latches=None,
                                                        prime_game_latches=None)
        else:
            dfa_handle = SymbolicPartitionedDFAFromSpot(formula=self.formula,
                                                        manager=self.manager,
                                                        latches_map=self.lVar_map_sym,
                                                        domain='gridworld',
                                                        game_latches=None,
                                                        prime_game_latches=None)
        
        self.dfa_handle = dfa_handle
        self.dfa_handle.create_latches_and_map()

        self.qVars = dfa_handle.qVars
        self.qVars_bdd = [var.bddPattern() for var in self.qVars]
        self.dfa_latches = dfa_handle.qVars
        self.qVar_map = dfa_handle.qVar_map
        self.qVar_map_sym = dfa_handle.qVar_map_sym
    

    def create_actions(self, player: str):
        """
         This method is simialr to the base method except with have additional set of state lbl variables that we add to the transition cube/
        """
        turn_bit: ADD = self.tVar_map_sym[player]
        p_idx = 0 if player == 'sys' else 1
        for r in range(self.rows):
            rVar_add: ADD = self.cube_to_add(self.xVar_map[p_idx][r], self.xVars[p_idx])

            for c in range(self.columns):
                cVar_add: ADD = self.cube_to_add(self.yVar_map[p_idx][c], self.yVars[p_idx])
                curr_state_lbl_cube = self.state_lbl_map_sym[(r, c)]
                
                # get valid acts for grid position (r, c) - this does check for wall or other obstacles in the successor step.
                valid_actions = self.get_valid_transitions(rPos=r, cPos=c)
                invalid_actions = set(self.env_actions) - valid_actions

                for act in valid_actions:
                    act_cube: str = self.action_map_sym[f'{player}_{act}']
                    nxt_rPos = r + Moves[act].value[0]
                    nxt_cPos = c + Moves[act].value[1]

                    # check if the next position is valid or not - only for Env player
                    if player == 'env' and (self.obsatcle_constraint_cube & self.tVar_map_sym[player] & self.xVar_map_sym[p_idx][nxt_rPos] & self.yVar_map_sym[p_idx][nxt_cPos]) != self.manager.addZero():
                        # invalid Env action must be mapped as STAY action
                        nxt_rPos, nxt_cPos = r, c
                        # book keeping
                        self.invalid_env_state_action_cube |= turn_bit & rVar_add & cVar_add & act_cube

                    # check if the next position is valid or not
                    if player == 'env' and act == 'STAY' and len(invalid_actions) > 0:
                        # invalid action must mapped as STAY action
                        invalid_act_cube = reduce(lambda x, y: x | y, [self.action_map_sym[f'{player}_{e_act}'] for e_act in invalid_actions])
                        self.invalid_env_state_action_cube |= turn_bit & rVar_add & cVar_add & invalid_act_cube
                        act_cube |= invalid_act_cube
                    
                    transition_cube: ADD = turn_bit & rVar_add & cVar_add & act_cube & ~self.obsatcle_constraint_cube & curr_state_lbl_cube
                    # nxt_state_lbl_cube = self.state_lbl_map_sym[(nxt_rPos, nxt_cPos)]
                    # nxt_state_lbl = self.state_lbl_map[(nxt_rPos, nxt_cPos)]

                    for idx, prime_rVar in enumerate(self.xVar_map[p_idx][nxt_rPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.xVars[p_idx][idx].bddPattern().__str__()] |= transition_cube

                    for idx, prime_rVar in enumerate(self.yVar_map[p_idx][nxt_cPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.yVars[p_idx][idx].bddPattern().__str__()] |= transition_cube
                    
                    # add lbl evolution to the transition relation
                    for ap_cube_str, _ in self.get_all_cubes(self.state_lbl_map_sym[(nxt_rPos, nxt_cPos)], self.lVars):
                        for idx, prime_lVar in enumerate(ap_cube_str.bddPattern().cubeString().replace('-', '')):
                            if prime_lVar == '1':
                                self.transition_relation[self.lVars[idx].bddPattern().__str__()] |= transition_cube
    

    def add_sys_frame_axioms(self):
        """
         This method is similar to the base method except with have additional set of state lbl variables that we add to the transition cube.
        """
        turn_bit: ADD = self.tVar_map_sym['env']
        for rPos in range(self.rows):
            rVar_add = self.cube_to_add(self.xVar_map[0][rPos], self.xVars[0])
            for cPos in range(self.columns):
                cVar_add = self.cube_to_add(self.yVar_map[0][cPos], self.yVars[0])
                curr_state_lbl_cube = self.state_lbl_map_sym[(rPos, cPos)]
                for act in self.env_action_map.keys():
                    act_cube: str = self.action_map_sym[act]
                    transition_cube: ADD = turn_bit & rVar_add & cVar_add & act_cube & ~self.obsatcle_constraint_cube #& curr_state_lbl_cube
                    for idx, prime_rVar in enumerate(self.xVar_map[0][rPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.xVars[0][idx].bddPattern().__str__()] |= transition_cube
                    
                    for idx, prime_rVar in enumerate(self.yVar_map[0][cPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.yVars[0][idx].bddPattern().__str__()] |= transition_cube
    
    
    def add_env_frame_axioms(self):
        turn_bit: ADD = self.tVar_map_sym['sys']
        for rPos in range(self.rows):
            rVar_add = self.cube_to_add(self.xVar_map[1][rPos], self.xVars[1])
            for cPos in range(self.columns):
                cVar_add = self.cube_to_add(self.yVar_map[1][cPos], self.yVars[1])
                curr_state_lbl_cube = self.state_lbl_map_sym[(rPos, cPos)]
                for act in self.sys_action_map.keys():
                    act_cube: str = self.action_map_sym[act]
                    transition_cube: ADD = turn_bit & rVar_add & cVar_add & act_cube & ~self.obsatcle_constraint_cube #& curr_state_lbl_cube
                    for idx, prime_rVar in enumerate(self.xVar_map[1][rPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.xVars[1][idx].bddPattern().__str__()] |= transition_cube
                    
                    for idx, prime_rVar in enumerate(self.yVar_map[1][cPos]):
                        if prime_rVar == '1':
                            self.transition_relation[self.yVars[1][idx].bddPattern().__str__()] |= transition_cube
    
    # def state_lbl_axiom(self):
    #     """
    #      Newed to add the empty ap evolves to empty ap irresprective of state and action taken
    #     """
    #     for sidx, s in enumerate(self.lVar_map['e']):
    #         if s == '1':
    #             self.transition_relation[self.lVars[sidx].bddPattern().__str__()] |=  self.lVar_map_sym['e']
    

    def create_transition_relation(self):
        """
         Call the base method's create transition relation for the Game Construction.  
        """
        # DFA TR
        self.dfa_handle.create_dfa_transition_relation()

        # game TR
        super().create_transition_relation()
        # self.state_lbl_axiom()

        # bookeeping
        self.monolithic_dfa_state_prime_state_trns: ADD = self.dfa_handle.monolithic_valid_q_ps_pq
    

    def post_process_transition_relation(self, debug: bool = False):
        super().post_process_transition_relation(debug=debug)

        # self.create_state_lbls(debug=False)

        # add the state lbls to the transition relation
        # for tr, tr_dd in self.transition_relation.items():
        #     self.transition_relation[tr] = tr_dd & self.state_lbl
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # add state lbls
        # curr_winning_states &= self.state_lbl
        # curr_winning_states = curr_winning_states.ite(self.state_lbl, curr_winning_states)
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches, self.prime_latches)
        preimage = dfa_preimage_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))

        # exist abstract state lbls here
        # preimage = preimage.existAbstract(self.lVars_cube)

        return preimage
    
    
    def convert_cube_to_state_ADD(self,
                                  dd: ADD,
                                  state_flag: bool = True, dfa_flag: bool = True,
                                  action: bool = False, lbl_flag: bool = False,
                                  verbose: bool = False, table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
        Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the action as well, set action flag to True. 
        """
        relevant_vars = []
        if state_flag:
            relevant_vars.extend(self.latches) # includes pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars
        # if lbl_flag:
        #     relevant_vars.extend(self.lVars) # label vars
        
        headers = []
        if verbose:
            headers.append('state')
        if lbl_flag:
            headers.append('Succ labels')
        if action:
            headers.append('action')
        if verbose:
            headers.append('value')

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.rVars)     
        lbl_exist_cube = reduce(lambda a, b: a & b, self.tVar + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        xConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = [] 
        for cube, val in cubes:
            state = None
            action_str = None
            lbl_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            lbl_cube_str = cube.existAbstract(lbl_exist_cube).bddPattern().cubeString().replace('-', '')
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
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos), self.dfa_handle.qVar_map.inv[qConf_cube_str])
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos), self.dfa_handle.qVar_map.inv[qConf_cube_str]), val), None])
                
            except KeyError:
                continue
            
            # print the robot and human actions as well
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
                    lbl_str = self.lVar_map.inv[lbl_cube_str]
                except KeyError:
                    continue
            
            row = []
            if verbose:
                row.append(state)
            if lbl_flag:
                row.append(lbl_str)
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
    

    def preimage_test(self, From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
        From = From.swapVariables(latches, prime_latches)
        return From.vectorCompose(prime_latches, ts_action)
        

    
    def symbolic_abstract(self, add_function, variables_to_abstract: List[ADD]):
        """
        Eliminates variables by taking the cofactors.
        """
        result_add = add_function

        for var_add in variables_to_abstract:
            pos_cofactor = result_add.cofactor(var_add)
            neg_cofactor = result_add.cofactor((~var_add))
            result_add = pos_cofactor | neg_cofactor 
            
        return result_add
    

    def test_preimage_2(self):
        sys_pos = (1, 1)
        goal_cube_sys = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (1, 0)
        goal_cube_env = self.tVar_map_sym['env'] & self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        goal_cube = goal_cube_sys & goal_cube_env
        goal_cube_lbl = goal_cube & self.state_lbl_map_sym[(1, 1)]
        dfa_goal_cube =  goal_cube_lbl & self.dfa_handle.goal_latch 

        # lets try evlving over dfa
        dfa_preimage = self.preimage_test(From=dfa_goal_cube, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        print('DFA preimage:', dfa_preimage)

        self.convert_cube_to_state_ADD(dfa_preimage, lbl_flag=True, action=False, verbose=True)
        
        dfa_game_preimage = self.preimage_test(From=goal_cube & self.dfa_handle.init_latch, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)

        self.convert_cube_to_state_ADD(dfa_game_preimage, lbl_flag=False, action=True, verbose=True)
    

    def test_preimage(self):
        sys_pos = (1, 1)
        goal_cube_sys = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (1, 0)
        goal_cube_env = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        # goal_cube = self.tVar_map_sym['env'] & goal_cube_sys & self.dfa_handle.goal_latch & goal_cube_env
        goal_cube = self.tVar_map_sym['env'] & goal_cube_sys & self.dfa_handle.goal_latch & goal_cube_env
        # goal_cube = self.dfa_handle.goal_latch
        print('Goal state:', goal_cube)
        # goal_cube = (goal_cube & self.state_lbl).ite(self.manager.addOne(), self.manager.plusInfinity())
        goal_cube = (goal_cube & self.state_lbl_map_sym[(1, 1)]).ite(self.manager.addOne(), self.manager.plusInfinity())
        # need to hook each state with lbl of nxt state
        # goal_cube = (goal_cube & (self.lVar_map_sym['e'] | self.lVar_map_sym['c'] | self.lVar_map_sym['goal'])).ite(self.manager.addOne(), self.manager.plusInfinity())

        # first AND with state lbl
        goal_cube_lbl = goal_cube #& self.state_lbl
        print('Goal state with lbl:', goal_cube_lbl)

        # self.convert_cube_to_state_ADD(goal_cube_lbl, action=False, verbose=True)

        # let try this with compute preimage
        # dfa_game_preimage = self.compute_preimage(goal_cube_lbl)

        # # first evolve over the DFA
        # dfa_preimage = self.preimage_test(From=goal_cube_lbl, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        # print('DFA preimage:', dfa_preimage)

        # self.convert_cube_to_state_ADD(dfa_preimage, lbl_flag=True, action=False, verbose=True)

        # remove the lbl
        # dfa_preimage = dfa_preimage.existAbstract(self.lVars_cube)
        # dfa_preimage = self.symbolic_min_abstract(dfa_preimage, self.lVars)
        # print('DFA preimage w/o lbl:', dfa_preimage)
        
        # then evolve over the game
        dfa_game_preimage = self.preimage_test(From=goal_cube_lbl, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)

        self.convert_cube_to_state_ADD(dfa_game_preimage, lbl_flag=False, action=True, verbose=True)

        # dfa_game_preimage = dfa_game_preimage.ite(self.manager.addOne(), self.manager.plusInfinity()) # set the value of states not in preimage to infinity

        valid_env_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        winning_states: ADD = self.compute_min_max_preimage(dfa_game_preimage, valid_env_action_mask=valid_env_action_mask)

        print("After taking Min-Max over Preimage: ", winning_states)
        self.convert_cube_to_state_ADD(winning_states,  lbl_flag=True, action=False, verbose=True)

        print("Done taking Min-Max")

        # lets try to get rid of state lbl here
        # valid_state_lbls = reduce(lambda a, b: a | b, [self.lVar_map_sym[ap] for ap in self.grid.keys()] + [self.lVar_map_sym['c']])

        # dfa_game_preimage = dfa_game_preimage & self.weight

        # test: ADD = self.symbolic_abstract(dfa_game_preimage, variables_to_abstract=self.lVars)
        # lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        # test2 = dfa_game_preimage.existAbstract(self.lVars_cube)
        # print('DFA Game w/o state lbl preimage: ', test)
        # print('DFA Game w/o state lbl preimage using ExistAbstract Operation: ', test2)