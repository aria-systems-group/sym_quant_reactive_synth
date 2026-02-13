from functools import reduce
from collections import defaultdict
from typing import List, Union, Tuple, Dict

from bidict import bidict
from tabulate import tabulate

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.frankadynamic_ratio_else_tb_noprime import FrankaWorldDynamicRatioTurnBasedElseNoPrime
from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMonaNoPrime, SymbolicPartitionedDFAFromSpotNoPrime


class SymbolicPartitionedDFAGameNoPrime(FrankaWorldDynamicRatioTurnBasedElseNoPrime):
    """
     This class extends the FrankaWorldDynamicRatioTurnBasedElseNoPrime class to create a game environment where the objective is specified by a LTL/LTLf formula.

     Unlike the SymbolicPartitionedDFAGame we do create two sets of latches. We skip the prime latches.
    """
    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 restricted_human_boxes: List[int],
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False):
        """
         Initialize the SymbolicPartitionedDFAGame with the given parameters and create DFA latches and maps.

         Args:
             boxes (int): Number of boxes in the environment.
             locs (int): Number of locations in the environment.
             ratio (int): Ratio parameter for the game.
             init (tuple): Initial state configuration.
             goal (tuple): Goal state configuration.
             restricted_human_locs (List[int]): List of restricted human locations.
             ltlf_flag (bool): Flag indicating whether the formula is LTLf (True) or LTL (False).
             formula (str): The LTL/LTLf formula specifying the objective of the game.
        """
        self.formula: str = formula
        self.qVars: List[ADD] = []
        self.qVars_bdd: List[BDD] = []
        self.qVar_map: List[ADD] = {} 
        self.qVar_map_sym: List[ADD] = {} 
        self.ltlf_flag: bool = ltlf_flag
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMonaNoPrime, SymbolicPartitionedDFAFromSpotNoPrime] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(boxes, locs, ratio, init, goal, restricted_human_locs, restricted_human_boxes, enable_reordering=False)

        # set up dfa init and goal states
        self.dfa_handle.set_init_latch()
        self.dfa_handle.set_goal_latch()
        # call it 2nd time here to override the base method - is this the best way?
        self.init_latch: ADD = self.dfa_handle.init_latch & self.init_latch
        self.goal_latch: ADD = self.set_goal_latch()

        # now we create the TR for the dfa
        self.dfa_handle.game_latches = self.latches

        # by default variable reordering is disabled for DFA games - to check for computation time without this optimization
        # however, switching variable ordering makes the code faster.
        if enable_reordering:
            self.manager.autodynEnable()
    

     # override the create lacthes method to include dfa latches
    def create_all_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the FrankaDynamic Turn-Based Game.
          1. turn variables - tVars
          2. ratio variables - kVars
          3. predicate variables - pVars
          4. box predicate variables - bVars
        """
        offset = self.manager.size()
        self.tVar: List[ADD] = [self.manager.addVar(offset, 't0')]
        self.kVars: List[ADD] = self.create_ratio_vars()
        self.pVars, self.bVars = self.create_latches()
        self.create_all_maps()
        self.create_all_sym_maps()

        # create DFA latches next
        self.create_dfa_latches_and_maps()
    

    def set_goal_latch(self):
        return self.dfa_handle.goal_latch
    

    def create_dfa_latches_and_maps(self):
        """
         Create DFA latches and their symbolic maps based on the provided LTL/LTLf formula.
        """
        # here we only create the variables and maps for the dfa
        if self.ltlf_flag:
            dfa_handle = SymbolicPartitionedDFAFromMonaNoPrime(formula=self.formula,
                                                               manager=self.manager,
                                                               latches_map=self.xVar_map_sym,
                                                               game_latches=None)
        else:
            dfa_handle = SymbolicPartitionedDFAFromSpotNoPrime(formula=self.formula,
                                                               manager=self.manager,
                                                               latches_map=self.xVar_map_sym,
                                                               game_latches=None)
        
        self.dfa_handle = dfa_handle
        self.dfa_handle.create_latches_and_map()

        self.qVars = dfa_handle.qVars
        self.qVars_bdd = [var.bddPattern() for var in self.qVars]
        self.dfa_latches: List[ADD] = dfa_handle.qVars
        self.qVar_map = dfa_handle.qVar_map
        self.qVar_map_sym = dfa_handle.qVar_map_sym
    

    def create_transition_relation(self):
        """
         Call the base method's create transition relation for the Game Construction.  
        """
        # game TR
        super().create_transition_relation()

        # DFA TR
        self.dfa_handle.create_dfa_transition_relation()
    

    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, dfa_flag: bool = True, action: bool = False, verbose: bool = False, table_header: bool = True) -> List[List[Tuple[Tuple[str, str, int], str]]]:
        """
        Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the action as well, set action flag to True. 
        """
        relevant_vars = [] + self.tVar + self.kVars
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
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.rVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.rVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.rVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.rVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.rVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = [] 
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_exist_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_exist_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_exist_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_exist_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = ((self.tVar_map.inv[tConf_exist_str], self.kVar_map.inv[kConf_exist_str], self.pVar_map.inv[rConf_exist_str], box_states), self.dfa_handle.qVar_map.inv[qConf_exist_str])
                states_action_pairs.append([
                    (((self.tVar_map.inv[tConf_exist_str],
                       self.kVar_map.inv[kConf_exist_str],
                       self.pVar_map.inv[rConf_exist_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_exist_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if action:
                rCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(rCube_str, self.rVars)]
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
    

    def get_next_state(self, turn: str, curr_state_exp: List[str], act_name: str, **kwargs) -> Tuple[ADD, str]:
        # get the next state in the game in explicit form
        curr_game_state = list(curr_state_exp[0][0][0][0])
        if turn == 'robot':
            curr_game_state_sym: ADD = self.get_next_state_robot(curr_game_state, act_name)
        else:
            curr_game_state_sym, act_name = self.get_next_state_human(curr_game_state, act_name)
        
        return curr_game_state_sym, act_name


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False) -> None:
        """
         A function to rollout a given strategy
        """
        curr_state_sym = self.init_latch & self.dfa_handle.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]

        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state_sym, action=False, verbose=verbose, table_header=False)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.comp_winning_states).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            turn = 'robot' if curr_state_exp[0][0][0][0][0] == 'robot' else'human'
            
            # get the action to be taken at the current state
            act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string] if turn == 'robot' else self.action_map.inv[act_cube_string]
            except KeyError:
                print("No robot action found!!")
                return
           
            # get the next state in the game
            curr_game_state_sym, act_name = self.get_next_state(turn, curr_state_exp, act_name, curr_state_sym=curr_state_sym)
            
            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            curr_dfa_state: int = curr_state_exp[0][0][0][1]
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_game_state_sym)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym
                    break
            
            curr_state_sym: ADD = curr_game_state_sym & curr_dfa_state
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Robot Action: {act_name}") if turn == 'robot' else print(f"Human Action: {act_name}")
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        # first evolve over the DFA
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        preimage: ADD = dfa_preimage.vectorCompose(self.latches, list(self.transition_relation.values()))

        return preimage
    

    def iros23_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            # first evolve over the DFA
            # dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))
            pre_states: BDD = dfa_preimage.vectorCompose(self.latches_bdd, self.ts_bdd_transition_fun_list)

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


    def test_pre_image(self):
        goal_cube = self.tVar_map_sym['human'] & self.xVar_map_sym['ready l1'] & self.xVar_map_sym['b0 l1'] & self.kVar_map_sym['k0'] & self.dfa_handle.goal_latch
        # goal state is b0 and l0 and ready l0
        print('Goal state:', goal_cube)

        dfa_pre = goal_cube.vectorCompose(self.dfa_latches, list(self.dfa_handle.dfa_transition_relation.values()))
        game_pre = dfa_pre.vectorCompose(self.latches, list(self.transition_relation.values()))
        dfa_pre_full = goal_cube.vectorCompose(self.dfa_latches, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        game_pre_full = dfa_pre_full.vectorCompose(self.latches, list(self.transition_relation.values()))

        print('DFA Preimage: ', dfa_pre)
        self.convert_cube_to_state_ADD(game_pre, state_flag=True, dfa_flag=True, action=False, verbose=True)

        print('DFA Preimage Full TR: ', dfa_pre_full)
        self.convert_cube_to_state_ADD(game_pre_full, state_flag=True, dfa_flag=True, action=False, verbose=True)