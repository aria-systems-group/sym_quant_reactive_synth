import itertools

from functools import reduce

from bidict import bidict
from tabulate import tabulate

from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple, Set

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic_no_prime import GridWorldDynamicGameNoPrime, CELL
from src.compositional_graphs.gridworld.gridworld_dynamic_doors_no_prime import GridWorldDynamicDoorsGameNoPrime
from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMonaNoPrime, SymbolicPartitionedDFAFromSpotNoPrime


class GridWorldDynamicDFAGameNoPrime(GridWorldDynamicGameNoPrime):
    
    def __init__(self,
                 rows: int, columns: int,
                 init: List[CELL], goal: List[CELL],
                 formula: str, 
                 grid: Optional[Dict['str', List[CELL]]] = dict({}),
                 players: Dict[str, int] = {'sys': 1, 'env': 1},
                 restricted_env_locs: Optional[List[CELL]] = [],
                 camera: bool = False,
                 ltlf_flag: bool = True,
                 cooperative_game: bool = False,
                 enable_reordering: bool = False,
                 **kwargs):
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
            camera (bool): Flag indicating whether to include the camera predicate in the state labeling. 
            formula (str): The LTL/LTLf formula specifying the objective of the game.
        """
        self.formula: str = formula
        self.qVars: List[ADD] = []
        self.qVars_bdd: List[BDD] = []
        self.qVar_map: List[ADD] = {} 
        self.qVar_map_sym: List[ADD] = {}
        self.lVar_map = bidict({})
        self.lVar_map_sym = dict({})
        self.state_lbl_map: Dict[CELL, Set[str]] = defaultdict(lambda: set())
        self.ltlf_flag: bool = ltlf_flag
        self.camera: bool = camera
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMonaNoPrime, SymbolicPartitionedDFAFromSpotNoPrime] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(rows=rows, columns=columns,
                         init=init, goal=goal,
                         grid=grid, players=players,
                         restricted_env_locs=restricted_env_locs,
                         cooperative_game=cooperative_game,
                         enable_reordering=False)

        # set up dfa init and goal states
        self.create_state_lbls(debug=False)
        self.dfa_handle.set_init_latch()
        self.dfa_handle.set_goal_latch()
        # call it 2nd time here to override the base method
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl & self.not_error_state_cube
        self.goal_latch: ADD = (self.set_goal_latch() & self.state_lbl &  self.not_error_state_cube) | (self.env_error_cube & ~self.sys_error_cube)

        # now we create the TR for the dfa
        self.dfa_handle.game_latches = self.latches

        # by default variable reordering is disabled for DFA games - to check for computation time without this optimization
        # however, switching variable ordering makes the code faster.
        if enable_reordering:
            self.manager.autodynEnable()
    
    @property
    def formula(self):
        return self._formula
    
    @formula.setter
    def formula(self, formula_str: str):
        assert formula_str is not None, "Please provide a valid LTL/LTLf formula string."
        self._formula = formula_str


    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()

        # list of labels excluding obstacle label - including collision and camera if specified.
        self.lbls_list = [ob for ob in self.grid.keys() if ob not in self.obstacles] + ['c'] + (['p'] if self.camera else [])
        self.lVars = self.create_state_lbls_vars()
        self.lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        self.create_lVars_map()
        self.state_lbl_map_sym: Dict[CELL, ADD] = defaultdict(lambda: reduce(lambda x, y: x & y, [~e for e in self.lVars]))
        self.create_state_lbl_map()

        # create the dfa state variables and maps
        self.create_dfa_latches_and_maps()
    

    def set_latches(self):
        super().set_latches()
        self.latches.extend(self.lVars)
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]

    
    def set_goal_latch(self):
        return self.dfa_handle.goal_latch


    def get_number_of_states(self, verbose: bool = True) -> Tuple[int, int]:
        num_sys_states, num_env_states = super().get_number_of_states(verbose=verbose)
        # multiple it by the numbers of the states
        total_dfa_game_state = (num_sys_states + num_env_states) * self.dfa_handle.num_of_states
        if verbose:
            print(f'Number of States in DFA Game: {total_dfa_game_state:,}')
        return total_dfa_game_state

    def create_state_lbl_map(self):
        """
         A method that parse the grid dictionary construction a diction that maps every cell to a set of AP.
        """
        for ap, loc in self.grid.items():
            if ap not in self.lbls_list:
                continue
            for cell in loc:
                self.state_lbl_map[cell].add(ap)
        
        for cell, set_ap in self.state_lbl_map.items():
            self.state_lbl_map_sym[cell] = reduce(lambda a, b: a & b, [self.lVar_map_sym[lbl] if lbl in set_ap else ~self.lVar_map_sym[lbl] for lbl in self.lVar_map_sym.keys()])


    def create_state_lbls_vars(self) -> List[ADD]:
        """
         A method to create state labels for the gridworld. This is used for labeling states with propositions for LTL synthesis. 

         We create one for each label. This is done to avoid non-determinism in vectorCompose().
        """
        # collision ap is always part of the state lbl
        varsize = self.manager.size()
        num_of_lbls =  len(self.lbls_list)
        lVars: List[ADD] =  [self.manager.addVar(l + varsize , f'l{l}') for l in range(num_of_lbls)]
        return lVars
            
    
    def create_lVars_map(self):
        for ap_idx, ap in enumerate(self.lbls_list):
            self.lVar_map[ap] = self.lVars[ap_idx].bddPattern().__str__()
            self.lVar_map_sym[ap] = self.lVars[ap_idx]
    

    def compute_photograph_predicate(self) -> ADD:
        """
        Computes the symbolic set (ADD) of states where the Env player is in the 
        Sys player's 3x3 camera field of view.
        
        Field of View:
        XXX
        RXX
        XXX
        where R is Sys player at (r, c), and X are the 3x3 relative cells:
        Rows: [r-1, r, r+1], Columns: [c, c+1, c+2]
        """
        # 1. Construct the Row Relation: r_env \in {r_sys-1, r_sys, r_sys+1}
        row_rel = self.manager.addZero()
        # TODO: every player is equipped with camera? and photographing any Env player is sufficient?
        for sys_p in range(self.players['sys']): # for each player
            for env_p in range(self.players['env']):
                for dr in [-1, 0, 1]:
                    for r in range(self.rows):
                        nr = r + dr
                        if 0 <= nr < self.rows:
                            # Relation: (Sys is at r) AND (Env is at nr)
                            row_rel |= (self.xVar_map_sym[sys_p][r] & self.xVar_map_sym[env_p + self.players['sys']][nr])
        
        # 2. Construct the Col Relation: c_env \in {c_sys, c_sys+1, c_sys+2}
        col_rel = self.manager.addZero()
        for sys_p in range(self.players['sys']):
            for env_p in range(self.players['env']):
                for dc in [0, 1, 2]:
                    for c in range(self.columns):
                        nc = c + dc
                        if 0 <= nc < self.columns:
                            col_rel |= (self.yVar_map_sym[sys_p][c] & self.yVar_map_sym[env_p + self.players['sys']][nc])
        
        # 3. The predicate p is simply the conjunction of these two independent relations
        p_true_set = row_rel & col_rel
        return p_true_set
 

    def create_state_lbls(self, debug: bool = False):
        ap_conditions = defaultdict(lambda: self.manager.addZero())
        # Create the characteristic ADD for each property
        if self.camera:
            # self.compute_photograph_predicate()
            ap_conditions['p'] |= self.compute_photograph_predicate()
        
        for lbl in self.lbls_list:
            # we handle collision/photograph label separately as it is not associated with a single player's cell but is a function of both players position. We add this later.
            if lbl == 'c' or lbl == 'p':
                continue
            for r, c in self.grid.get(lbl, []):
                # Only Sys player(s) position matters for lbls
                for p in range(self.players['sys']):
                    ap_conditions[lbl] |= (self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c])

        for r in range(self.rows):
            for c in range(self.columns):
                # Both 2 players are at the same (r,c)
                sys_at = reduce(lambda a, b: a | b, [self.xVar_map_sym[p][r] & self.yVar_map_sym[p][c] for p in range(self.players['sys'])])
                env_at = reduce(lambda a, b: a | b, [self.xVar_map_sym[p + self.players['sys']][r] & self.yVar_map_sym[p + self.players['sys']][c] for p in range(self.players['env'])])
                # TODO: update this to be a uniqe label for collision with each sys player. 
                ap_conditions['c'] |= (sys_at & env_at)

        self.state_lbl = self.manager.addOne()
        for name, condition in ap_conditions.items():
            l_var = self.lVar_map_sym[name]
            self.state_lbl &= (l_var.ite(condition, ~condition))
        
        if debug:
            print("--- State labels ADDs  ---")
            self.convert_cube_to_state_ADD(self.state_lbl , state_flag=True, dfa_flag=True, lbl_flag=True, action=False, verbose=True)
        

    def add_lbl_evolution_to_TR(self):
        """
         A function that add the state lbl evolution to the existing the TR
        """
        game_latch = self.tVars + self.eVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds]
        pre_state_nxt_lbl = self.state_lbl.vectorCompose(game_latch, list(self.transition_relation.values())[:-len(self.lVars)])
        
        for cube_string in itertools.product([0, 1], repeat=len(self.lVars)):
            lbl_cube = reduce(lambda a,b: a & b, [self.lVars[idx] if bit else ~self.lVars[idx] for idx, bit in enumerate(cube_string)])
            pre_state_action: ADD = pre_state_nxt_lbl.restrict(lbl_cube) & self.not_error_state_cube
            for idx, prime_lVar in enumerate(cube_string):
                if prime_lVar == 1:
                    self.transition_relation[self.lVars[idx].bddPattern().__str__()] |= pre_state_action & self.state_lbl
        
        # iterate through the tVars, xVars, yVars and add state lbls to all cubes
        for k in self.transition_relation.keys():
            # if we skip the lable vars in the TR as they are taken care of by the above code.
            if k.startswith('l'):
                continue
            self.transition_relation[k] &= self.state_lbl

    

    def create_dfa_latches_and_maps(self):
        """
        Create DFA latches and their symbolic maps based on the provided LTL/LTLf formula.
        """
        # here we only create the variables and maps for the dfa
        if self.ltlf_flag:
            dfa_handle = SymbolicPartitionedDFAFromMonaNoPrime(formula=self.formula,
                                                               manager=self.manager,
                                                               latches_map=self.lVar_map_sym,
                                                               domain='gridworld',
                                                               game_latches=None)
        else:
            dfa_handle = SymbolicPartitionedDFAFromSpotNoPrime(formula=self.formula,
                                                               manager=self.manager,
                                                               latches_map=self.lVar_map_sym,
                                                               domain='gridworld',
                                                               game_latches=None)
        
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
        for player in self.tVar_map.keys():
            self.create_actions(player=player)
        
        # need to add frame axioms, i.e., when it is env move Sys variables remain the same and vice versa.
        self.add_frame_axioms()
        self.add_turn_var_update_rule()
        self.add_lbl_evolution_to_TR()

        # remove invalid Sys moves to wall
        self.post_process_transition_relation(debug=False)
        self.add_error_state_self_loops()
        self.add_invalid_state_acts_to_tr()
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # first evolve over the DFA
        # dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        preimage = dfa_preimage.vectorCompose(self.latches, list(self.transition_relation.values()))

        return preimage
    

    def hybrid_compute_preimage(self, win_state_bucket, return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            # dfa_pre_states: BDD = dfa_succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_pre_states: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))

            pre_states: BDD = dfa_pre_states.vectorCompose(self.latches_bdd, self.ts_bdd_transition_fun_list)

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
            relevant_vars.extend(self.latches) # includes tVars, xVars, yVars, lVars
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
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.eVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.rVars)     
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
        
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
                sys_err_state = cube.bddPattern().cubeString()[sys_eidx].replace('-', '')
                env_err_state = cube.bddPattern().cubeString()[env_eidx].replace('-', '')
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos + [sys_err_state, env_err_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str])
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos, sys_err_state, env_err_state), self.dfa_handle.qVar_map.inv[qConf_cube_str]), val), None])
                
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
            if verbose:
                row.append(val)
            states_bookkeeping.append(tuple(row))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))
        
        return states_action_pairs


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a given strategy
        """
        curr_state = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        self.invalid_env_state_action_cube = self.transition_relation[self.env_error_cube.bddPattern().__str__()] | self.transition_relation[self.sys_error_cube.bddPattern().__str__()]
        while (curr_state & self.goal_latch).isZero():
            curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state, lbl_flag=False, action=False, table_header=False, verbose=False)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval: int = list((curr_state & self.state_lbl & self.comp_winning_states).generate_cubes())[0][1]
            except IndexError:
                opt_sval: int = 0
            
            if verbose:
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)]))
            
            # get the action to be taken at the current state
            act_cube: BDD = (strategy.restrict(curr_state & self.state_lbl)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')
            # only choose valid action. By constuction env will always have atleast one valid action. 
            while not (curr_state & self.state_lbl & act_cube.toADD() & self.invalid_env_state_action_cube).isZero():
                act_cube: BDD = (strategy.restrict(curr_state & self.state_lbl)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
                act_cube_string = act_cube.cubeString().replace('-', '')
            
            curr_dfa_state: int = curr_state_exp[0][0][0][1]

            # get the action to be taken at the current state
            turn = 'sys' if curr_state_exp[0][0][0][0][0].startswith('sys') else 'env'
            try:
                act_name = self.sys_action_map.inv[act_cube_string] if turn == 'sys' else self.env_action_map.inv[act_cube_string]
            except KeyError:
                print(f"turn: {turn}")
                print("act_cube_string: ", act_cube_string)
                print("No action found!!")
                return

            # get the next state
            curr_state, act_name, curr_state_exp = self.get_next_state(curr_state_exp=curr_state_exp[0][0][0][0], act=act_name)

            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_state & self.state_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym
                    break
            
            curr_state: ADD = curr_state & curr_dfa_state

            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Sys Action: {act_name}") if turn == 'sys' else print(f"Env Action: {act_name}")
    


class GridWorldDynamicDoorsDFAGameNoPrime(GridWorldDynamicDFAGameNoPrime, GridWorldDynamicDoorsGameNoPrime):
    def __init__(self, 
                 rows: int, columns: int,
                 formula: str, 
                 init: List[CELL], goal: List[CELL],
                 grid: Dict[str, List[CELL]],
                 players: Dict[str, int] = {'sys': 1, 'env': 1},
                 restricted_env_locs: Optional[List[CELL]] = [],
                 camera: bool = False,
                 cooperative_game: bool = False,
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False,
                 **kwargs):
        super().__init__(rows=rows, columns=columns,
                         formula=formula, init=init,
                         goal=goal, grid=grid, camera=camera,
                         ltlf_flag=ltlf_flag, players=players,
                         cooperative_game=cooperative_game,
                         restricted_env_locs=restricted_env_locs,
                         enable_reordering=False)
        # call it 3rd time here to override the base method
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl & self.all_door_uncalimed & self.not_error_state_cube
        if enable_reordering:
            self.manager.autodynEnable()
    
    def add_lbl_evolution_to_TR(self):
        """
         A function that add the state lbl evolution to the existing the TR. We override the base method to incorporate door vars (dVars) from the door game.
        """
        game_latch: List[ADD] = self.tVars + self.eVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds]
        pre_state_nxt_lbl = self.state_lbl.vectorCompose(game_latch, list(self.transition_relation.values())[:-len(self.lVars)])
        
        for cube_string in itertools.product([0, 1], repeat=len(self.lVars)):
            lbl_cube = reduce(lambda a,b: a & b, [self.lVars[idx] if bit else ~self.lVars[idx] for idx, bit in enumerate(cube_string)])
            pre_state_action: ADD = pre_state_nxt_lbl.restrict(lbl_cube) & self.not_error_state_cube
            for idx, prime_lVar in enumerate(cube_string):
                if prime_lVar == 1:
                    self.transition_relation[self.lVars[idx].bddPattern().__str__()] |= pre_state_action & self.state_lbl
        
        # iterate through the tVars, xVars, yVars and add state lbls to all cubes
        for k in self.transition_relation.keys():
            # we skip the lable vars in the TR as they are taken care of by the above code.
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
        tConf_exist_cube = reduce(lambda a, b: a & b, self.rVars + self.eVars + self.lVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.rVars)
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)
        
        dConf_exist_cube = dict({})
        for didx in range(len(self.dVars)):
            if len(self.grid['door']) == 1:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars)
            else:
                dConf_exist_cube[didx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for xVar_adds in self.xVars for var in xVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.dVars[:didx] + self.dVars[didx+1:])
        
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])
        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for dVar_adds in self.dVars for var in dVar_adds] + self.qVars) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])

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
                sys_err_state = cube.bddPattern().cubeString()[sys_eidx].replace('-', '')
                env_err_state = cube.bddPattern().cubeString()[env_eidx].replace('-', '')
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos + door_states + [sys_err_state, env_err_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str])
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos, *door_states, sys_err_state, env_err_state), self.dfa_handle.qVar_map.inv[qConf_cube_str]), val), None])
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