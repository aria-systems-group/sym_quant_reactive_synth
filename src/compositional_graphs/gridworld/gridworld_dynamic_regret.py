import time
import math

from bidict import bidict
from tabulate import tabulate

from functools import reduce 
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set

from cudd import ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame
from src.compositional_graphs.gridworld.gridworld_dynamic import CELL


class GridWorldDynamicRegretGame(GridWorldDynamicDFAGame):
    """
     This class extends the GridWorldDynamicDFAGame class to 
     (i) construct Graph of Utility (GoU) and Compute Min-Min stratgies and values. Then
     (ii) compute best-alternate respose values (BR) on GoU
     (iii) construct Graph of Regret (GoR) and Compute Min-Max regret-minimizing strategies and values.
    """

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
                 enable_reordering: bool = False):
        self.budget: int = budget
        self.uVars: List[ADD] = []
        self.uVars_bdd: List[BDD] = []
        self.prime_uVars: List[ADD] = []
        self.prime_uVars_bdd: List[BDD] = []
        self.brVars: List[ADD] = []
        self.brVars_bdd: List[BDD] = []
        self.prime_brVars: List[ADD] = []
        self.prime_brVars_bdd: List[BDD] = []
        self.uVar_map: List[ADD] = bidict({}) 
        self.uVar_map_sym: List[ADD] = bidict({}) 
        self.brVar_map: List[ADD] = bidict({}) 
        self.brVar_map_sym: List[ADD] = bidict({})
        self.gou_ts_bdd_transition_fun_list: List[BDD] = []
        self.gobr_ts_bdd_transition_fun_list: List[BDD] = []
        super().__init__(rows=rows, columns=columns,
                         init=init, goal=goal,
                         formula=formula, grid=grid,
                         players=players, camera=camera,
                         ltlf_flag=ltlf_flag,
                         restricted_env_locs=restricted_env_locs, 
                         cooperative_game=cooperative_game,
                         enable_reordering=enable_reordering)\
        
        # self.states_per_cost: Dict[int, ADD] = defaultdict(lambda: self.manager.addZero())
        self.uVars_transition_relation = None
        self.brVars_transition_relation = None
        self.graph_of_utility_tr = None
        self.graph_of_br_tr  = None

        self.cVals = None
        self.rVals = None

        self.gou_init_latch: ADD = None
        self.gou_goal_latch: ADD = None
        self.set_gou_init_latch()
        self.set_gou_goal_latch()

        # book keeping
        self.gou_game_latches = self.latches + self.uVars + self.qVars
        self.gou_game_prime_latches = self.prime_latches + self.prime_uVars + self.prime_qVars

        self.gobr_game_latches = None
        self.gobr_game_prime_latches = None
    

    def create_all_boolean_state_vars_and_maps(self):
        self.tVars = self.create_player_latches()
        self.xVars, self.yVars = self.create_latches()
        self.eVar = [self.manager.addVar(self.manager.size(), 'e')]
        self.lbls_list = [ob for ob in self.grid.keys() if ob not in self.obstacles] + ['c'] + (['p'] if self.camera else [])
        self.lVars = self.create_state_lbls_vars()
        self.uVars = self.create_utility_latches()
        self.uVars_bdd = [u.bddPattern() for u in self.uVars]

        self.create_xVar_map()
        self.create_yVar_map()
        self.create_tVar_map()
        self.create_lVars_map()
        self.create_uVar_map()
        self.create_symbolic_maps(prime=False)
        self.state_lbl_map_sym: Dict[CELL, ADD] = defaultdict(lambda: reduce(lambda x, y: x & y, [~e for e in self.lVars]))
        self.lVars_cube = reduce(lambda x, y: x & y, self.lVars)
        self.create_state_lbl_map()
        # create the dfa state variables and maps
        self.create_dfa_latches_and_maps()


    def create_all_prime_boolean_state_vars_and_maps(self):
        self.prime_tVars = self.create_prime_player_latches()
        self.prime_xVars, self.prime_yVars = self.create_prime_latches()
        self.prime_eVar = [self.manager.addVar(self.manager.size(), 'pe')]
        self.prime_lVars = self.create_prime_state_lbls_vars()
        self.prime_lVar_map_sym = {lbl: cube.swapVariables(self.lVars, self.prime_lVars) for lbl, cube in self.lVar_map_sym.items()}
        self.prime_uVars = self.create_prime_utility_latches()
        self.prime_uVars_bdd = [u.bddPattern() for u in self.prime_uVars]
        self.create_symbolic_maps(prime=True)

        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
        self.prime_qVars_bdd = [var.bddPattern() for var in self.prime_qVars]


    def create_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        uVar_size = math.ceil(math.log2(self.budget + 2)) # +1 here to account for 0-bit vector and another +1 for  budget +1 which is a sink state.
        # create an additional boolean var to skip the 0-vector latch
        uVar_size = uVar_size + 1 if pow(2, uVar_size) == self.budget + 2 else uVar_size 
        uVars: List[ADD] = [self.manager.addVar(u + varsize, 'u' + str(u)) for u in range(uVar_size)]
        return uVars
    

    def create_prime_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        return [self.manager.addVar(u + varsize, 'pu' + str(u)) for u in range(len(self.uVars))]
    

    def create_uVar_map(self):
        """
         Small function to create symbolic maps for the uVar_map. 
        """
        # budget + 1 is a state to represent budget exceeded; it will be a sink state.
        for u in range(self.budget + 2):
            ubit_str = f"{u + 1:0{len(self.uVars)}b}"
            self.uVar_map[f'u{u}'] = ubit_str
            self.uVar_map_sym[f'u{u}'] = self.cube_to_add(ubit_str, self.uVars)
    
    def set_gou_init_latch(self):
        self.gou_init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl & ~self.eVar[0] & self.uVar_map_sym['u0']
    
    def set_gou_goal_latch(self):
        self.gou_goal_latch: ADD = self.set_goal_latch() & self.state_lbl & ~self.eVar[0] & ~self.uVar_map_sym[f'u{self.budget + 1}']
    

    def create_utility_transition_relation(self):
        """
         Create the transition relation for utility variables.
        """
        self.uVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.uVars}
        self.get_states_per_cost()
        for u in range(self.budget + 1):
            uConf_cube = self.uVar_map_sym[f'u{u}']
            for state_cost in self.states_per_cost.keys():
                transition_cube = uConf_cube & self.states_per_cost[state_cost].toADD()
                
                prime_u_val = u + state_cost if (u + state_cost) <= self.budget else self.budget + 1 
                uConf_prime_cube_str = self.uVar_map[f'u{prime_u_val}']
                for sidx, s in enumerate(uConf_prime_cube_str):
                    if s == '1':
                        self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube     
                        # self.transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube      
        
        # add self-loop for the sink state budget + 1
        for sidx, s in enumerate(self.uVar_map[f'u{self.budget + 1}']):
            if s == '1':
                self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |=  self.uVar_map_sym[f'u{self.budget + 1}']
                # self.transition_relation[self.uVars[sidx].bddPattern().__str__()] |=  self.uVar_map_sym[f'u{self.budget + 1}']
    
    def create_goal_nodes_with_utility_values(self, verbose: bool = False) -> ADD:
        """
         Create Graph of Utility's accepting nodes with utility values.
        """
        uVars_add = self.manager.plusInfinity()
        # skip u = 0; we reason about u = 0 separately
        for u in range(1, self.budget + 1):
            uVars_add = uVars_add.min(self.uVar_map_sym[f'u{u}'].ite(self.manager.addConst(u), self.manager.plusInfinity()))

        dfa_goal_cube: ADD = self.gou_goal_latch.ite(self.manager.addOne(), self.manager.plusInfinity())
        goal_add: ADD = dfa_goal_cube.times(uVars_add)
        # final_goal_add = (self.uVar_map_sym[f'u{0}'] & self.dfa_handle.goal_latch).ite(self.manager.addZero(), goal_add)
        final_goal_add = (self.uVar_map_sym[f'u{0}'] & self.gou_goal_latch).ite(self.manager.addZero(), goal_add)
        if verbose:
            print(final_goal_add)
            print("Initialized the goal states with Utility values!")
        return final_goal_add
    

    def create_gou_sys_env_transition_relations(self):
        """
         A method to create separate transition relations for the system (robot) and environment (human) actions.
        """
        # post-process to get TR for robot and Human actions separately
        self.sys_gou_transition_relation = []
        self.env_gou_transition_relation = []
        for tr_dd in self.graph_of_utility_tr:
            self.sys_gou_transition_relation.append(tr_dd & self.sys_tVar_cube)
            self.env_gou_transition_relation.append(tr_dd & self.env_tVar_cube)

    

    def create_transition_relation(self):
        # creates game transition relation first and then dfa transition relation
        super().create_transition_relation()

        # now create utility transition relation
        self.create_utility_transition_relation()

        tic = time.time()
        strategy = self.gou_solve(verbose=False, optimized=False)
        # hybrid_strategy = self.hybrid_gou_solve(verbose=False)
        # bdd_strategy = self.pure_bdd_gou_solve(verbose=False)
        toc = time.time()
        print(f"Time to synthesize GOU values: {toc - tic} seconds")

        self.gou_roll_out_strategy(strategy=strategy, verbose=True)
        # self.test_pre_image()
    

    def gou_compute_preimage(self, curr_winning_states: ADD) -> ADD:
        """
         Preimage comptuation over the Graph of Utility transition relation.
        """
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars, self.prime_latches + self.prime_uVars)
        preimage = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.graph_of_utility_tr)

        return preimage
    
    def compute_preimage_optimized(self, curr_winning_states: ADD) -> Tuple[ADD, ADD]:
        """
         Optimized preimage comptuation over the Graph of Utility transition relation.
        """
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)

        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars, self.prime_latches + self.prime_uVars)
        pre_sys = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.sys_gou_transition_relation)
        pre_env = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.env_gou_transition_relation)

        return pre_sys, pre_env
    
    
    def gou_solve(self, verbose: bool = False, optimized: bool = False) -> Optional[ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        # if optimized:
        #     self.create_gou_sys_env_transition_relations()
        
        goal = self.create_goal_nodes_with_utility_values(verbose=False)
        # print("Goal States with Utility values:", goal)
        curr_winning_states =  self.manager.plusInfinity()
        curr_winning_states = curr_winning_states.min(goal)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            # if optimized:
            #     pre_sys, pre_env = self.compute_preimage_optimized(curr_winning_states)
            #     next_winning_states_sys = self.symbolic_min_abstract(pre_sys, variables_to_abstract=self.rVars)
            #     next_winning_states_env = self.symbolic_min_abstract(pre_env, variables_to_abstract=self.rVars)
            #     # TODO: updated for arbitrary number of players
            #     next_winning_states = self.tVar[0].ite(next_winning_states_sys, next_winning_states_env)
            # else:
            preimage: ADD = self.gou_compute_preimage(curr_winning_states)
            next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.gou_convert_cube_to_state_ADD(next_winning_states, action=False, verbose=True, print_val=True)
            
            if next_winning_states.compare(curr_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if self.gou_init_latch & curr_winning_states != self.manager.plusInfinity():
                    if self.gou_init_latch & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.gou_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Cooperation Strategy Exists!!. The State value is {init_val}")
                    self.cVals = curr_winning_states
                    # if optimized:
                    #     preimage = pre_sys.min(pre_env)
                    return preimage.min(goal) if init_val < math.inf else None
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def gou_get_next_state(self, curr_state_sym: ADD, curr_state_exp: ADD, act: str):
        next_state, act_name, next_state_exp = super().get_next_state(curr_state_exp[0][0][0][0], act)
        # update uVal - get the next utility value based on the current state and action
        # get the state cost
        if self.weight.cofactor(curr_state_sym & self.state_lbl & self.action_map_sym[act_name]).isZero():
            state_cost: int = 0
        else:
            state_cost: int = int(list((self.weight.cofactor(curr_state_sym & self.state_lbl & self.action_map_sym[act_name])).generate_cubes())[0][1])
        state_utl = int(curr_state_exp[0][0][0][-1][1:])

        if state_utl + state_cost <= self.budget:
            next_uVar_sym = self.uVar_map_sym[f'u{state_utl + state_cost}']
        else:
            next_uVar_sym = self.uVar_map_sym[f'u{self.budget + 1}']
        return next_state & next_uVar_sym, act_name, next_state_exp + [state_utl]
    

    def gou_roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a strategy on the graph of utility game.
        """
        curr_state_sym = self.gou_init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        self.invalid_env_state_action_cube = self.transition_relation['e']
        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            curr_state_exp: List[str] = self.gou_convert_cube_to_state_ADD(curr_state_sym,
                                                                           state_flag=True,
                                                                           lbl_flag=False,
                                                                           action=False,
                                                                           verbose=False,
                                                                           table_header=False,
                                                                           print_val=False)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.state_lbl & self.cVals).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            if verbose:
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)])) 
            
             # get the action to be taken at the current state
            act_cube: BDD = (strategy.restrict(curr_state_sym & self.state_lbl)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')
            # only choose valid action. By constuction env will always have atleast one valid action. 
            while not (curr_state_sym & self.state_lbl & act_cube.toADD() & self.invalid_env_state_action_cube).isZero():
                act_cube: BDD = (strategy.restrict(curr_state_sym & self.state_lbl)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
                act_cube_string = act_cube.cubeString().replace('-', '')
            
            curr_dfa_state: int = curr_state_exp[0][0][0][1]

            turn = 'sys' if curr_state_exp[0][0][0][0][0].startswith('sys') else'env'
            try:
                act_name =  self.sys_action_map.inv[act_cube_string] if turn == 'sys' else self.env_action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return
           
            curr_game_state_sym, act_name, _ = self.gou_get_next_state(curr_state_sym, curr_state_exp, act_name)
            
            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_state_sym = dfa_state_sym.swapVariables(self.qVars, self.prime_qVars)
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_game_state_sym & self.state_lbl)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym.swapVariables(self.prime_qVars, self.qVars)
                    break
            
            curr_state_sym: ADD = curr_game_state_sym & curr_dfa_state
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Sys Action: {act_name}") if turn == 'sys' else print(f"Env Action: {act_name}")
    

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
        
        # the next vars are l' vars - we ignore them for now. The next ones are action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])
        eidx = self.manager.addVariables().index(self.eVar[0])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.eVar + self.rVars + self.uVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.uVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVar + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVar + self.uVars + self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVar + self.uVars + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
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
                eVar_state = self.eVar_map.inv[cube.bddPattern().cubeString()[eidx].replace('-', '')]
                uVar_state = self.uVar_map.inv[uConf_cube_str]
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos + [eVar_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state)
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos, eVar_state), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), val), None])
                
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
    

    def test_pre_image(self):
        sys_pos = (1, 1)
        sys_cube = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (1, 1)
        env_cube = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        # goal_cube = sys_cube & env_cube & self.dfa_handle.qVar_map_sym[2] & self.uVar_map_sym['u2'] & self.tVar_map_sym['env0'] & ~self.eVar[0]
        goal_cube = sys_cube & env_cube & self.dfa_handle.qVar_map_sym[2] & self.tVar_map_sym['env0'] & ~self.eVar[0]
        print("Goal Cube: ",  goal_cube)
        self.gou_convert_cube_to_state_ADD(goal_cube & self.state_lbl, lbl_flag=True, action=False, verbose=True)

        # preiamge testing on dfa game only
        # preimage =  self.preimage_test(From=goal_cube & self.state_lbl,
        #                    latches=self.latches, prime_latches=self.prime_latches,
        #                    ts_action=list(self.transition_relation.values()))
        # print("Preimage: ", preimage)
        gou_preimage = self.gou_compute_preimage(goal_cube & self.state_lbl)

        # gou_preimage = self.gou_compute_preimage(goal_cube & self.state_lbl)
        print("Preimage: ", gou_preimage)
        self.gou_convert_cube_to_state_ADD(gou_preimage, action=False, verbose=True)