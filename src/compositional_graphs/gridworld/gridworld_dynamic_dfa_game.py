import itertools

from functools import reduce

from bidict import bidict
from tabulate import tabulate

from collections import defaultdict
from typing import List, Union, Optional, Dict, Tuple, Set

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame, Moves, CELL
from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot


class GridWorldDynamicDFAGame(GridWorldDynamicGame):
    
    def __init__(self,
                 rows: int, columns: int,
                 init: List[CELL], goal: List[CELL],
                 formula: str, 
                 grid: Optional[Dict['str', List[CELL]]] = dict({}),
                 restricted_env_locs: Optional[List[CELL]] = [],
                 camera: bool = False,
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
            camera (bool): Flag indicating whether to include the camera predicate in the state labeling. 
            formula (str): The LTL/LTLf formula specifying the objective of the game.
        """
        self.formula: str = formula
        self.qVars: List[ADD] = []
        self.qVars_bdd: List[BDD] = []
        self.qVar_map: List[ADD] = {} 
        self.qVar_map_sym: List[ADD] = {} 
        self.ltlf_flag: bool = ltlf_flag
        self.camera: bool = camera
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(rows=rows, columns=columns, init=init, goal=goal, grid=grid,restricted_env_locs=restricted_env_locs, enable_reordering=False)

        # set up dfa init and goal states
        self.create_state_lbls(debug=False)
        self.dfa_handle.set_init_latch()
        self.dfa_handle.set_goal_latch()
        # call it 2nd time here to override the base method
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl
        self.goal_latch: ADD = self.set_goal_latch() & self.state_lbl

        # now we create the TR for the dfa
        self.dfa_handle.game_latches = self.latches

        # by default variable reordering is disabled for DFA games - to check for computation time without this optimization
        # however, switching variable ordering makes the code faster.
        if enable_reordering:
            self.manager.autodynEnable()


    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()

        self.state_lbl_map: Dict[CELL, Set[str]] = defaultdict(lambda: set())
         # list of labels excluding obstacle label - including collision and camera if specified.
        self.lbls_list = [ob for ob in self.grid.keys() if ob not in self.obstacles] + ['c'] + ['p'] if self.camera else []
        self.lVar_map = bidict({})
        self.lVar_map_sym = dict({}) 
        self.lVars = self.create_state_lbls_vars()
        self.lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        self.create_lVars_map()
        self.state_lbl_map_sym: Dict[CELL, ADD] = defaultdict(lambda: reduce(lambda x, y: x & y, [~e for e in self.lVars]))
        self.create_state_lbl_map()

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
        self.prime_lVar_map_sym = {lbl: cube.swapVariables(self.lVars, self.prime_lVars) for lbl, cube in self.lVar_map_sym.items()}
        
        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
        self.prime_qVars_bdd = [var.bddPattern() for var in self.prime_qVars]
    

    def set_latches(self):
        super().set_latches()
        self.latches.extend(self.lVars)
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]
    

    def set_prime_latches(self):
        super().set_prime_latches()
        self.prime_latches.extend(self.prime_lVars)
        self.prime_latches_bdd: List[BDD] = [var.bddPattern() for var in self.prime_latches]
    

    def set_goal_latch(self):
        return self.dfa_handle.goal_latch


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
    
    def create_prime_state_lbls_vars(self) -> List[ADD]:
        """
         A method to create prime version of the state label variables. This is used for labeling next states with propositions for LTL synthesis. 
        """
        varsize = self.manager.size()
        prime_lVars: List[ADD] =  [self.manager.addVar(l + varsize , f"pl{l}") for l in range(len(self.lVars))]
        return prime_lVars
            
    
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
        for dr in [-1, 0, 1]:
            for r in range(self.rows):
                nr = r + dr
                if 0 <= nr < self.rows:
                    # Relation: (Sys is at r) AND (Env is at nr)
                    row_rel |= (self.xVar_map_sym[0][r] & self.xVar_map_sym[1][nr])
        
        # 2. Construct the Col Relation: c_env \in {c_sys, c_sys+1, c_sys+2}
        col_rel = self.manager.addZero()
        for dc in [0, 1, 2]:
            for c in range(self.columns):
                nc = c + dc
                if 0 <= nc < self.columns:
                    col_rel |= (self.yVar_map_sym[0][c] & self.yVar_map_sym[1][nc])
        
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
                # Only Sys player (0) position matters for lbls
                ap_conditions[lbl] |= (self.xVar_map_sym[0][r] & self.yVar_map_sym[0][c])

        for r in range(self.rows):
            for c in range(self.columns):
                # Both players at same (r,c)
                sys_at = self.xVar_map_sym[0][r] & self.yVar_map_sym[0][c]
                env_at = self.xVar_map_sym[1][r] & self.yVar_map_sym[1][c]
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
        game_latch = self.tVar + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds]
        game_prime_latch = self.prime_tVar + [var for prime_xVar_adds in self.prime_xVars for var in prime_xVar_adds] + [var for prime_yVar_adds in self.prime_yVars for var in prime_yVar_adds]
        
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

    

    def create_dfa_latches_and_maps(self):
        """
        Create DFA latches and their symbolic maps based on the provided LTL/LTLf formula. Unloike the manipulator domain,
         the gridworld domain does not have separate prime variables for state lbls as we are do compose over these variables. 

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
    

    def create_transition_relation(self):
        """
         Call the base method's create transition relation for the Game Construction.  
        """
        # DFA TR
        self.dfa_handle.create_dfa_transition_relation()

        # game TR
        for player in ['sys', 'env']:
            self.create_actions(player=player)
        
        # need to add frame axioms, i.e., when it is env move Sys variables remain the same and vice versa.
        self.add_sys_frame_axioms()
        self.add_env_frame_axioms()

        self.add_turn_var_update_rule()
        self.add_lbl_evolution_to_TR()

        # remove invalid Sys moves to wall
        self.post_process_transition_relation(debug=False)

        # bookeeping
        self.monolithic_dfa_state_prime_state_trns: ADD = self.dfa_handle.monolithic_valid_q_ps_pq
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches, self.prime_latches)
        preimage = dfa_preimage_primed.vectorCompose(self.prime_latches, list(self.transition_relation.values()))

        return preimage
    

    def hybrid_compute_preimage(self, win_state_bucket, return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            # prime the vars
            dfa_succ_states_primed = succ_states.swapVariables(self.qVars_bdd, self.prime_qVars_bdd)
            # dfa_pre_states: BDD = dfa_succ_states_primed.vectorCompose(self.prime_qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_pre_states: BDD = dfa_succ_states_primed.vectorCompose(self.prime_qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))

            dfa_pre_states_primed = dfa_pre_states.swapVariables(self.latches_bdd, self.prime_latches_bdd)
            pre_states: BDD = dfa_pre_states_primed.vectorCompose(self.prime_latches_bdd, self.ts_bdd_transition_fun_list)

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

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.rVars)     
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
    

    def preimage_test(self, From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
        From = From.swapVariables(latches, prime_latches)
        return From.vectorCompose(prime_latches, ts_action)


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a given strategy
        """
        curr_state = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        
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
            curr_dfa_state: int = curr_state_exp[0][0][0][1]

            # get the action to be taken at the current state
            turn = 'sys' if curr_state_exp[0][0][0][0][0] == 'sys' else 'env'
            try:
                act_name = self.sys_action_map.inv[act_cube_string] if turn == 'sys' else self.env_action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return

            # get the next state
            curr_state, act_name, curr_state_exp = self.get_next_state(turn=turn, curr_state_exp=curr_state_exp[0][0][0][0], act=act_name)

            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_state_sym = dfa_state_sym.swapVariables(self.qVars, self.prime_qVars)
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                # FIX THIS: lbl map should be for sys state only
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_state & self.state_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym.swapVariables(self.prime_qVars, self.qVars)
                    break
            
            curr_state: ADD = curr_state & curr_dfa_state

            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Sys Action: {act_name}") if turn == 'sys' else print(f"Env Action: {act_name}")
    

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
        
        # dfa_game_preimage = self.preimage_test(From=goal_cube & self.dfa_handle.init_latch, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        dfa_game_preimage = self.preimage_test(From=dfa_preimage, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)

        self.convert_cube_to_state_ADD(dfa_game_preimage, lbl_flag=True, action=True, verbose=True)
    

    def test_preimage(self):
        sys_pos = (1, 1)
        goal_cube_sys = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (1, 0)
        goal_cube_env = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        goal_cube = self.tVar_map_sym['sys'] & goal_cube_sys & self.dfa_handle.goal_latch & goal_cube_env
        # goal_cube = self.tVar_map_sym['env'] & goal_cube_sys & self.dfa_handle.init_latch & goal_cube_env
        # goal_cube = self.dfa_handle.goal_latch
        print('Goal state:', goal_cube)
        goal_cube = (goal_cube & self.state_lbl).ite(self.manager.addOne(), self.manager.plusInfinity())
        # goal_cube = (goal_cube & self.state_lbl_map_sym[sys_pos]).ite(self.manager.addOne(), self.manager.plusInfinity())
        # need to hook each state with lbl of nxt state
        # goal_cube = (goal_cube & (self.lVar_map_sym['e'] | self.lVar_map_sym['c'] | self.lVar_map_sym['goal'])).ite(self.manager.addOne(), self.manager.plusInfinity())

        # first AND with state lbl
        goal_cube_lbl = goal_cube #& self.state_lbl
        print('Goal state with lbl:', goal_cube_lbl)

        # self.convert_cube_to_state_ADD(goal_cube_lbl, action=False, verbose=True)

        # let try this with compute preimage
        dfa_game_preimage = self.compute_preimage(goal_cube_lbl)

        # first evolve over the DFA
        # dfa_preimage = self.preimage_test(From=goal_cube_lbl, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        # dfa_preimage = self.preimage_test(From=goal_cube_lbl, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation.values()))
        # print('DFA preimage:', dfa_preimage)

        # self.convert_cube_to_state_ADD(dfa_preimage, lbl_flag=True, action=False, verbose=True)

        # remove the lbl
        # dfa_preimage = dfa_preimage.existAbstract(self.lVars_cube)
        # dfa_preimage = self.symbolic_min_abstract(dfa_preimage, self.lVars)
        # print('DFA preimage w/o lbl:', dfa_preimage)
        
        # then evolve over the game
        # dfa_game_preimage = self.preimage_test(From=dfa_preimage, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)

        self.convert_cube_to_state_ADD(dfa_game_preimage, lbl_flag=True, action=True, verbose=True)

        # dfa_game_preimage = dfa_game_preimage.ite(self.manager.addOne(), self.manager.plusInfinity()) # set the value of states not in preimage to infinity

        valid_env_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        winning_states: ADD = self.compute_min_max_preimage(dfa_game_preimage, valid_env_action_mask=valid_env_action_mask)

        print("After taking Min-Max over Preimage: ", winning_states)
        self.convert_cube_to_state_ADD(winning_states,  lbl_flag=True, action=False, verbose=True)

        print("Done taking Min-Max")