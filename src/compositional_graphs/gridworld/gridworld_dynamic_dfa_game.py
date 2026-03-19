import math

from functools import reduce

from bidict import bidict
from tabulate import tabulate

from typing import List, Union, Optional, Dict, Tuple

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot


class GridWorldDynamicDFAGame(GridWorldDynamicGame):
    
    def __init__(self,
                 rows: int, columns: int,
                 init: List[tuple], goal: tuple,
                 formula: str, 
                 grid: Optional[Dict] = dict({}),
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
        self.create_state_lbls()

        # now we create the TR for the dfa
        self.dfa_handle.game_latches = self.latches

        # by default variable reordering is disabled for DFA games - to check for computation time without this optimization
        # however, switching variable ordering makes the code faster.
        if enable_reordering:
            self.manager.autodynEnable()
    

    # def parent_boolean_state_vars_and_maps(self):
    #     super().parent_boolean_state_vars_and_maps()
        
    #     # create lbl - different manipulator domain
    #     self.lVars_map = bidict({})
    #     self.lVars = self.create_state_lbls_vars()
    #     self.create_lbl_map()
    #     self.lVars_map_sym = bidict({k: self.cube_to_add(v, self.lVars) for k, v in self.lVars_map.items()})


    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()

        self.lVars_map = bidict({})
        self.lVars = self.create_state_lbls_vars()
        self.lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        self.create_lbl_map()
        self.lVars_map_sym = bidict({k: self.cube_to_add(v, self.lVars) for k, v in self.lVars_map.items()})

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
        
        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
        self.prime_qVars_bdd = [var.bddPattern() for var in self.prime_qVars]
    

    def set_goal_latch(self):
        return self.dfa_handle.goal_latch


    def create_state_lbls_vars(self) -> List[ADD]:
        """
         A method to create state labels for the gridworld. This is used for labeling states with propositions for LTL synthesis. 
        """
        # collision ap is always part of the state lbl
        varsize = self.manager.size()
        num_of_lbls =  len(self.grid.keys()) + 2 # for collision ap and empty ap
        lVars_size = math.ceil(math.log2(num_of_lbls))
        lVars: List[ADD] =  [self.manager.addVar(l + varsize , f'l{l}') for l in range(lVars_size)]
        return lVars
    
    
    def create_lbl_map(self):
        # collision ap is always the first label
        self.lVars_map['e'] = f"{0:0{len(self.lVars)}b}"  # empty ap
        self.lVars_map['c'] = f"{1:0{len(self.lVars)}b}"  # collision ap
        for ap_idx, ap in enumerate(self.grid.keys()):
            bit_str = f"{ap_idx + 2:0{len(self.lVars)}b}"
            self.lVars_map[ap] = bit_str
    

    def create_state_lbls(self):
        self.state_lbl: ADD = self.manager.addZero()
        self.state_lbl |= self.valid_state_constraint & self.lVars_map_sym['e']  # start with empty ap
        # test = self.manager.addOne()
        # test = test.ite(self.lVars_map_sym['e'], self.manager.addOne())
        for r in range(self.rows):
            for c in range(self.columns):
                self.state_lbl |= reduce(lambda a, b: a & b, [self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] for p in range(2)]) & self.lVars_map_sym['c']
                # test = reduce(lambda a, b: a & b, [self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] for p in range(2)]) #& self.lVars_map_sym['c']
                # self.state_lbl = (reduce(lambda a, b: a & b, [self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] for p in range(2)]).ite(self.lVars_map_sym['c'], self.state_lbl)
                # test = (reduce(lambda a, b: a & b, [self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] for p in range(2)])).ite(self.lVars_map_sym['c'], test)
                
                # can optimize this by using the obstavle constraint cube that we already have
                if (r, c) in self.grid.get('wall', []):
                    # TODO: hard coding for two agents: update for all agents in the future
                    for p in range(2):
                        self.state_lbl |= self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] & self.lVars_map_sym['wall']
                        # self.state_lbl = (self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c]).ite(self.lVars_map_sym['wall'], self.state_lbl)
                        # test = (self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c]).ite(self.lVars_map_sym['wall'], test)
                
                if (r, c) in self.grid.get('lava', []):
                    # TODO: hard coding for two agents: update for all agents in the future
                    for p in range(2):
                        self.state_lbl |= self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] & self.lVars_map_sym['lava']
                        # self.state_lbl = (self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c]).ite(self.lVars_map_sym['lava'], self.state_lbl)
                        # test = (self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c]).ite(self.lVars_map_sym['lava'], test)
                
                if (r, c) in self.grid.get('goal', []):
                    # TODO: hard coding for 1 Sys agent: update for multiple agents in the future
                    # for p in range(2):
                    self.state_lbl |= self.xVar_map_sym[0][r] & self.yVar_map_sym[0][c] & self.lVars_map_sym['goal']
                    # self.state_lbl = (self.tVar_map_sym['sys'] & self.xVar_map_sym[0][r] & self.yVar_map_sym[0][c]).ite(self.lVars_map_sym['goal'], self.state_lbl)
                    # test = (self.tVar_map_sym['sys'] & self.xVar_map_sym[0][r] & self.yVar_map_sym[0][c]).ite(self.lVars_map_sym['goal'], test)
        
        print("Hi, Mom!")
        # state without any lbl will ahave the empty atomic predicate, which is represented by all 0s in the lbl vars
        # state_lbls_cube = reduce(lambda a, b: a | b, self.lVars_map_sym.values())
        # self.state_lbl = (state_lbls_cube & ~self.state_lbl['e']).ite()
        # map invalud state lbls to inf value
        # self.state_lbl = self.state_lbl.ite(self.manager.addOne(), self.manager.plusInfinity())

    

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
                                                        latches_map=self.lVars_map_sym,
                                                        domain='gridworld',
                                                        game_latches=None,
                                                        prime_game_latches=None)
        else:
            dfa_handle = SymbolicPartitionedDFAFromSpot(formula=self.formula,
                                                        manager=self.manager,
                                                        latches_map=self.lVars_map_sym,
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
    

    def create_transition_relation(self):
        """
         Call the base method's create transition relation for the Game Construction.  
        """
        # DFA TR
        self.dfa_handle.create_dfa_transition_relation()

        # game TR
        super().create_transition_relation()

        # bookeeping
        self.monolithic_dfa_state_prime_state_trns: ADD = self.dfa_handle.monolithic_valid_q_ps_pq
    

    def post_process_transition_relation(self, debug: bool = False):
        super().post_process_transition_relation(debug=debug)

        # add the state lbls to the transition relation
        # for tr, tr_dd in self.transition_relation.items():
        #     self.transition_relation[tr] = tr_dd & self.state_lbl
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # add state lbls
        curr_winning_states &= self.state_lbl
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
        preimage = preimage.existAbstract(self.lVars_cube)

        return preimage
    
    
    def convert_cube_to_state_ADD(self,
                                  dd: ADD,
                                  state_flag: bool = True, dfa_flag: bool = True,
                                  action: bool = False, verbose: bool = False,
                                  table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
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
        
        headers = []
        if verbose and action:
            headers = ['state', 'action', 'value']
        elif verbose and not action:
            headers = ['state', 'value']

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.rVars)     
        xConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.qVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        # TODO: hard coding for 2 agents, need to update for n agents
        for pidx in range(2):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVar + self.qVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
        
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
            
            if action:
                states_bookkeeping.append((state, action_str))
            else:
                states_bookkeeping.append((state, val))
        
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
    

    def test_preimage(self):
        sys_pos = (1, 1)
        goal_cube_sys = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (1, 0)
        goal_cube_env = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        goal_cube = self.tVar_map_sym['env'] & goal_cube_sys & self.dfa_handle.goal_latch & goal_cube_env
        # goal_cube = self.dfa_handle.goal_latch
        print('Goal state:', goal_cube)

        # first AND with state lbl
        # goal_cube_lbl = goal_cube & self.state_lbl
        # print('Goal state with lbl:', goal_cube_lbl)

        # self.convert_cube_to_state_ADD(goal_cube_lbl, action=False, verbose=True)

        # # first evolve over the DFA
        # dfa_preimage = self.preimage_test(From=goal_cube_lbl, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        # print('DFA preimage:', dfa_preimage)

        # remove the lbl
        # dfa_preimage = dfa_preimage.existAbstract(self.lVars_cube)
        # print('DFA preimage w/o lbl:', dfa_preimage)
        
        # then evolve over the game
        dfa_game_preimage = self.preimage_test(From=goal_cube, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)

        self.convert_cube_to_state_ADD(dfa_game_preimage, action=True, verbose=True)

        dfa_game_preimage = dfa_game_preimage.ite(self.manager.addOne(), self.manager.plusInfinity()) # set the value of states not in preimage to infinity

        valid_env_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        winning_states: ADD = self.compute_min_max_preimage(dfa_game_preimage & self.weight, valid_env_action_mask=valid_env_action_mask)

        print("After taking Min-Max over Preimage: ", winning_states)
        self.convert_cube_to_state_ADD(winning_states, action=True, verbose=True)

        # lets try to get rid of state lbl here
        # valid_state_lbls = reduce(lambda a, b: a | b, [self.lVars_map_sym[ap] for ap in self.grid.keys()] + [self.lVars_map_sym['c']])

        # dfa_game_preimage = dfa_game_preimage & self.weight

        # test: ADD = self.symbolic_abstract(dfa_game_preimage, variables_to_abstract=self.lVars)
        # lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        # test2 = dfa_game_preimage.existAbstract(self.lVars_cube)
        # print('DFA Game w/o state lbl preimage: ', test)
        # print('DFA Game w/o state lbl preimage using ExistAbstract Operation: ', test2)