import time
import math

from bidict import bidict
from tabulate import tabulate

from functools import reduce 
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set, Union

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
                 enable_reordering: bool = False,
                 **kwargs):
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
                         enable_reordering=enable_reordering)
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

        self._gou_vi_layers = 0
        self._gobr_vi_layers = 0

        # book keeping
        self.gou_game_latches = self.latches + self.uVars + self.qVars
        self.gou_game_prime_latches = self.prime_latches + self.prime_uVars + self.prime_qVars

        self.gobr_game_latches = None
        self.gobr_game_prime_latches = None
    
    @property
    def budget(self):
        return self._budget
    
    @budget.setter
    def budget(self, budget: int):
        assert budget > 0, "Budget must be a positive integer."
        self._budget = budget
    
    @property
    def gou_vi_layers(self):
        return self._gou_vi_layers

    @property
    def gobr_vi_layers(self):
        return self._gobr_vi_layers


    def create_all_boolean_state_vars_and_maps(self):
        self.tVars = self.create_player_latches()
        self.xVars, self.yVars = self.create_latches()
        self.eVars = self.create_error_latches()
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

    
    def create_all_br_vars_maps(self):
        """
         Give, the set of best-response values, this method create all latches and maps for best-alternate response variables. 
          It also creates prime latches.
        """
        # create variables for best-alternate response values
        self.brVars = self.create_br_latches()
        self.brVars_bdd = [br.bddPattern() for br in self.brVars]
        self.prime_brVars = self.create_prime_br_latches()
        self.prime_brVars_bdd = [br.bddPattern() for br in self.prime_brVars]
        self.create_br_var_map()
        self.set_gobr_init_latch()
        self.set_gobr_goal_latch()
        self.gobr_game_latches = self.latches + self.uVars + self.brVars + self.qVars
        self.gobr_game_prime_latches = self.prime_latches + self.prime_uVars + self.prime_brVars + self.prime_qVars
    

    def create_prime_utility_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        return [self.manager.addVar(u + varsize, 'pu' + str(u)) for u in range(len(self.uVars))]
    
    
    def create_br_latches(self):
        """
         Create best-alterante response (br) variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        brVar_size = math.ceil(math.log2(len(self.brVals)))
        # create an additional boolean var to skip the 0-vector latch
        brVar_size = brVar_size + 1 if pow(2, brVar_size) == len(self.brVals) else brVar_size 
        brVars: List[ADD] = [self.manager.addVar(br + varsize, 'br' + str(br)) for br in range(brVar_size)]
        return brVars

    
    def create_prime_br_latches(self):
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        prime_brVars: List[ADD] = [self.manager.addVar(br + varsize, 'pbr' + str(br)) for br in range(len(self.brVars))]
        return prime_brVars
    

    def create_uVar_map(self):
        """
         Small function to create symbolic maps for the uVar_map. 
        """
        # budget + 1 is a state to represent budget exceeded; it will be a sink state.
        for u in range(self.budget + 2):
            ubit_str = f"{u + 1:0{len(self.uVars)}b}"
            self.uVar_map[f'u{u}'] = ubit_str
            self.uVar_map_sym[f'u{u}'] = self.cube_to_add(ubit_str, self.uVars)
    
    def create_br_var_map(self):
        """
         Small function to create symbolic maps for the brVar_map. 
        """
        for idx, val in enumerate(self.brVals):
            bit_str = f"{idx + 1:0{len(self.brVars)}b}"
            self.brVar_map[val] = bit_str
            self.brVar_map_sym[val] = self.cube_to_add(bit_str, self.brVars)
    
    def set_gou_init_latch(self):
        self.gou_init_latch: ADD = self.dfa_handle.init_latch & self.init_latch & self.state_lbl & self.not_error_state_cube & self.uVar_map_sym['u0']
    
    def set_gou_goal_latch(self):
        self.gou_goal_latch: ADD = self.set_goal_latch() & self.state_lbl &self.not_error_state_cube & ~self.uVar_map_sym[f'u{self.budget + 1}']

    def set_gobr_init_latch(self):
        # init latch had init game state, lbl and dfa handle
        self.gobr_init_latch: ADD = self.init_latch & self.uVar_map_sym['u0'] & self.brVar_map_sym[math.inf]
    
    def set_gobr_goal_latch(self):
        self.gobr_goal_latch: ADD = (self.dfa_handle.goal_latch & self.state_lbl & self.not_error_state_cube & ~self.uVar_map_sym[f'u{self.budget + 1}']) | self.env_error_cube
    

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
        
        # add self-loop for the sink state budget + 1
        for sidx, s in enumerate(self.uVar_map[f'u{self.budget + 1}']):
            if s == '1':
                self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |=  self.uVar_map_sym[f'u{self.budget + 1}']


    def create_best_alternate_response_transition_relation(self):
        self.brVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.brVars}
        
        for br in self.brVals:
            brConf_cube = self.brVar_map_sym[br]
            
            for prime_br in self.brVals:
                if prime_br <= br:
                    # create the transition cube
                    transition_cube = brConf_cube & self.vector_of_br[prime_br]
                    
                    prime_brConf_cube_str = self.brVar_map[prime_br]
                    for sidx, s in enumerate(prime_brConf_cube_str):
                        if s == '1':
                            self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= transition_cube
                    
                else:
                    state_act_pairs = self.manager.addZero()
                    for i in self.brVals:
                        if i > br:
                            state_act_pairs |= self.vector_of_br[i]
                    
                    transition_cube = brConf_cube & state_act_pairs
                
                    prime_brConf_cube_str = self.brVar_map[br]
                    for sidx, s in enumerate(prime_brConf_cube_str):
                        if s == '1':
                            self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= transition_cube
        
            # add that from human states, the best-alternate response remains the same
            human_transition_cube = brConf_cube & self.env_tVar_cube
            prime_brConf_cube_str = self.brVar_map[br]
            for sidx, s in enumerate(prime_brConf_cube_str):
                if s == '1':
                    self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= human_transition_cube
        
    
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
    

    def get_states_with_one_outgoing_transition_gou(self) -> BDD:
        """
         This method computes the set of GoU sys states that have exactly one outgoing robot action.
        """
        bdd_gou_state_single_act = self.manager.bddZero()
        for curr_player, succ_player in self.turn_update_rule.items():
            if curr_player.startswith('sys'):
                pre_state_action = self.gou_compute_preimage(self.tVar_map_sym[succ_player])
                bdd_gou_state_single_act |= pre_state_action.existAbstract(self.rVars_cube).bddInterval(1, 1)
                
                self.gou_convert_cube_to_state_ADD(bdd_gou_state_single_act.toADD(), action=False, verbose=False)      

        # as accepting states in DFA are sink states in GoU, we need to post-process the gou_state_act_count so that accepting states map to cardinality 1.
        bdd_gou_state_single_act |= self.dfa_handle.goal_latch.bddPattern()
        return bdd_gou_state_single_act
    

    def compute_best_alternate_response(self, verbose: bool = False) -> None:
        """
        A method to compute the best alterante response (ba). Given, tuple (s, s'), best-alternate response is the scalar value associated with:
            Informal: What if I took any other valid edge from (s, s'') where s'' =\= s' for every Sys player state.
            Mathermatically, given cVal (cooperative value) for every state s in G, we have

            ba(s, s') = +inf if s is Env player states
            ba(s, s') = min (s, s'') {cVal(s'')} if s is Sys plaeyr states

            min(s, s'') = +inf if no s'' exists, i.e., there does not exist an alternate edge.
        
        Note: we note that our Transition function is deterministic, i.e., given (si, ai) where i \in {Env, Sys}, 
        Tr(si, ai) -> sj' where j =\= i and s' is the next state such that |sj| = 1.

        Hence, the tuple (s, s') can be replaced with (s, as) which will be useful when constructing the TR for GoBR later.
        
        Method: Output ADD(s, as)-br where br is the best-response.
        """
        # compute preimage of ADD(s')-cVal to get ADD(s, a)-cVal(s')
        game_state_action = self.gou_compute_preimage(self.cVals & self.state_lbl)

        game_state_action = self.sys_tVar_cube.ite(game_state_action, self.manager.plusInfinity())

        # now compute the best alternate response
        self.vector_of_br = defaultdict(lambda: self.manager.addZero())
        not_dfa_goal_states_bdd: BDD = ~self.dfa_handle.goal_latch.bddPattern()
        
        lVals = {*range(0, self.budget + 1)} | {math.inf}
        for ract, ract_sym in self.action_map_sym.items():
            if ract.startswith('env'):
                continue
            print(f"Computing BR for Sys Act: {ract}")
            
            game_state_action_without_ract = ract_sym.ite(self.manager.plusInfinity(), game_state_action)
            ba_per_act = self.symbolic_min_abstract(game_state_action_without_ract, self.rVars)

            # chop the ADDs into vector of BDD(s), one for each leaf node
            for leaf_val in lVals:
                # leav_vals == inf may have invalid states into, so post-process and remove it later
                bdd_state_act = (ba_per_act.bddInterval(leaf_val, leaf_val))
                if not bdd_state_act.isZero():
                    bdd_state_act &= ract_sym.bddPattern()
                    # remove goal states from br computation; later we add them to +inf br value
                bdd_state_act &= not_dfa_goal_states_bdd
                self.vector_of_br[leaf_val] |= bdd_state_act.toADD()
        
        # print stuff for debugging
        print("Done computing BR")

        # ovveride the +inf BDD. The above code works for states with one egdes. 
        # The inf vector include these states as well as valid state conf. Further, we manually all accepting states in DFA to +inf as they are sink states in GoU.
        # This is not capture in the above code. Hence, we manually override the +inf BDD here.
        if math.inf in self.vector_of_br.keys():
            inf_states: BDD = self.get_states_with_one_outgoing_transition_gou()
            self.vector_of_br[math.inf] |= inf_states.toADD()
        
        self.brVals: Set[float] = sorted(set(self.vector_of_br.keys()))

        # post-processing best-response to only preserve the lwer states action pair value
        # unions of all predecessors
        pre_states: ADD = reduce(lambda x, y: x | y, self.vector_of_br.values())
        self.monolithic_br: ADD = pre_states.ite(self.manager.addOne(), self.manager.plusInfinity())
        for br in sorted(self.vector_of_br.keys(), reverse=True):
            br_states: ADD = self.vector_of_br[br]
            self.monolithic_br = br_states.ite(self.manager.addConst(br), self.monolithic_br)
        
        # finally put them back in vector_of_br
        for br in self.vector_of_br.keys():
            self.vector_of_br[br] = self.monolithic_br.bddInterval(br, br).toADD()

        if verbose:
            # l, h = min(set(self.brVals) - {math.inf}), max(set(self.brVals) - {math.inf})
            # t = self.monolithic_br.bddInterval(l, h).toADD()
            # self.gou_convert_cube_to_state_ADD(t, action=True, verbose=True)
            self.gou_convert_cube_to_state_ADD(self.monolithic_br, action=True, verbose=True)
    

    def create_transition_relation(self):
        # creates game transition relation first and then dfa transition relation
        super().create_transition_relation()

        # now create utility transition relation
        self.create_utility_transition_relation()

        tic = time.time()
        strategy = self.gou_solve(verbose=False)
        # hybrid_strategy = self.hybrid_gou_solve(verbose=False)
        # bdd_strategy = self.pure_bdd_gou_solve(verbose=False)
        toc = time.time()
        print(f"Time to synthesize GOU values: {toc - tic} seconds")

        # self.gou_convert_cube_to_state_ADD(self.cVals, action=False, verbose=True, print_val=True)

        # self.gou_roll_out_strategy(strategy=strategy, verbose=True)
        # import sys
        # sys.exit(-1)
        # compute best-alternate response
        tic = time.time()
        self.compute_best_alternate_response(verbose=False)
        toc = time.time()
        print(f"Time to compute Best-Alternate Response: {toc - tic} seconds")

        # create boolean vars and their prime versions for Best-alternate response values computed
        self.create_all_br_vars_maps()
        
        # create br Transition Relation
        tic = time.time()
        self.create_best_alternate_response_transition_relation()
        toc = time.time()
        print(f"Time to create GoBR Transition Relation: {toc - tic} seconds")

    

    def gou_compute_preimage(self, curr_winning_states: ADD) -> ADD:
        """
         Preimage comptuation over the Graph of Utility transition relation.
        """
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        # dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars, self.prime_latches + self.prime_uVars)
        preimage = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars, self.graph_of_utility_tr)

        return preimage
    
    def compute_regret_preimage(self, curr_winning_states: ADD) -> ADD:
        # prime the vars
        curr_winning_states_primed = curr_winning_states.swapVariables(self.qVars, self.prime_qVars)
        
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the DFA game state (s, u)
        dfa_preimage_primed = dfa_preimage.swapVariables(self.latches + self.uVars + self.brVars, self.prime_latches + self.prime_uVars + self.prime_brVars)
        preimage_subr: ADD = dfa_preimage_primed.vectorCompose(self.prime_latches + self.prime_uVars + self.prime_brVars, self.graph_of_br_tr)

        return preimage_subr
    

    def hybrid_gou_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            # prime the vars
            dfa_succ_states_primed: BDD = succ_states.swapVariables(self.qVars_bdd, self.prime_qVars_bdd)
            # first evolve over the DFA
            # dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_preimage: BDD = dfa_succ_states_primed.vectorCompose(self.prime_qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))
            dfa_preimage_primed: BDD = dfa_preimage.swapVariables(self.latches_bdd + self.uVars_bdd, self.prime_latches_bdd + self.prime_uVars_bdd)
            pre_states: BDD = dfa_preimage_primed.vectorCompose(self.prime_latches_bdd + self.prime_uVars_bdd, self.gou_ts_bdd_transition_fun_list)

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
    
    def hybrid_gobr_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            # prime the vars
            dfa_succ_states_primed: BDD = succ_states.swapVariables(self.qVars_bdd, self.prime_qVars_bdd)
            # first evolve over the DFA
            # dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_preimage: BDD = dfa_succ_states_primed.vectorCompose(self.prime_qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))
            dfa_preimage_primed: BDD = dfa_preimage.swapVariables(self.latches_bdd + self.uVars_bdd + self.brVars_bdd, self.prime_latches_bdd + self.prime_uVars_bdd + self.prime_brVars_bdd)
            pre_states: BDD = dfa_preimage_primed.vectorCompose(self.prime_latches_bdd + self.prime_uVars_bdd + self.prime_brVars_bdd, self.gobr_ts_bdd_transition_fun_list)

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

    
    def gou_convert_mono_tr_to_action_tr(self):
        for tr_bdd in self.graph_of_utility_tr:
            self.gou_ts_bdd_transition_fun_list.append(tr_bdd.bddPattern())
    
    def gobr_convert_mono_tr_to_action_tr(self):
        for tr_bdd in self.graph_of_br_tr:
            self.gobr_ts_bdd_transition_fun_list.append(tr_bdd.bddPattern())
    

    def gou_convert_monolithic_add_to_bdd_buckets(self, monolithic_add: ADD) -> Dict[int, BDD]:
        """
         Given a monolithic ADD of winning states, convert it into buckets of BDDs based on state values.

         The values the states can take are from 0 to [Budget] with increment of c_max
        """    
        win_state_bucket: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        
        # convert the winning states into buckets of BDD
        for sval in range(0, self.budget + 1):
            # get the states with state value equal to sval and store them in their respective bukcets
            win_sval = monolithic_add.bddInterval(sval, sval)

            if not win_sval.isZero():
                win_state_bucket[sval] |= win_sval
        
        return win_state_bucket

    def gobr_convert_monolithic_add_to_bdd_buckets(self, monolithic_add: ADD, reg_vals: List[int]) -> Dict[int, BDD]:
        """
         Given a monolithic ADD of winning states, convert it into buckets of BDDs based on state values.

         The values the states can take are from 0 to max(regret_vals).
        """    
        win_state_bucket: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        
        # convert the winning states into buckets of BDD
        for sval in reg_vals:
            # get the states with state value equal to sval and store them in their respective bukcets
            win_sval = monolithic_add.bddInterval(sval, sval)

            if not win_sval.isZero():
                win_state_bucket[sval] |= win_sval
        
        return win_state_bucket
    
    
    def gou_solve(self, verbose: bool = False) -> Optional[ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        
        goal = self.create_goal_nodes_with_utility_values(verbose=False)
        # print("Goal States with Utility values:", goal)
        curr_winning_states = self.manager.plusInfinity().min(goal)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            preimage: ADD = self.gou_compute_preimage(curr_winning_states)
            next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.gou_convert_cube_to_state_ADD(next_winning_states, action=False, verbose=True, print_val=True)
            
            if next_winning_states.compare(curr_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                self._gou_vi_layers = layer
                if self.gou_init_latch & curr_winning_states != self.manager.plusInfinity():
                    if self.gou_init_latch & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.gou_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Cooperative Opt. Strategy Exists!!. The State value is {init_val}")
                    self.cVals = curr_winning_states
                    return preimage.min(goal) if init_val < math.inf else None
                print(f"No Cooperative Opt. Strategy Exists!!. The State value is {math.inf}")
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    
    
    def hybrid_gou_solve(self, verbose: bool = False) -> Optional[ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))

        # preprocess TR into buckets of BDDs separated based on actions
        self.gou_convert_mono_tr_to_action_tr()
        
        goal = self.create_goal_nodes_with_utility_values(verbose=verbose)
        # print("Goal States with Utility values:", goal)
        curr_winning_states = self.manager.plusInfinity().min(goal)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            win_state_bucket = self.gou_convert_monolithic_add_to_bdd_buckets(monolithic_add=curr_winning_states)
            preimage: ADD = self.hybrid_gou_compute_preimage(win_state_bucket)
            next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.gou_convert_cube_to_state_ADD(next_winning_states, action=False, verbose=True, print_val=True)
            
            if next_winning_states.compare(curr_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                self._gou_vi_layers = layer
                if self.gou_init_latch  & curr_winning_states != self.manager.plusInfinity():
                    if self.gou_init_latch & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.gou_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Cooperative Opt. Strategy Exists!!. The State value is {init_val}")
                    self.cVals = curr_winning_states
                    return preimage.min(goal) if init_val < math.inf else None
                print(f"No Cooperative Opt. Strategy Exists!!. The State value is {math.inf}")
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def pure_bdd_gou_solve(self, verbose: bool = False) -> Optional[ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))

        # preprocess TR into buckets of BDDs separated based on actions
        self.gou_convert_mono_tr_to_action_tr()
        
        goal: ADD = self.create_goal_nodes_with_utility_values(verbose=verbose)
        goal_states_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        curr_winning_states: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval in range(0, self.budget + 1):
            goal_sval = goal.bddInterval(sval, sval)
            if not goal_sval.isZero():
                goal_states_buckets[sval] |= goal_sval
                curr_winning_states[sval] |= goal_sval
       
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            # compute preimage
            vector_preimage: Dict[int, BDD] = self.hybrid_gou_compute_preimage(win_state_bucket=curr_winning_states, return_bdd=True)

            # take min over Sys and Env player states
            next_winning_states_opt = self.compute_min_preimage_pure_bdd(preimage=vector_preimage)
            # retain the min over goal states - goal/sink states in GoU do not have outgoing transition. We add them back as preimage will not capture them
            # this was taken care by min operation in Pure and Hybrid Approach. Here, we have to do it manually
            next_winning_states_opt = self.compute_min_goal_states(preimage=next_winning_states_opt, goal=goal_states_buckets)
            for goal_sval in sorted(goal_states_buckets.keys()):
                next_winning_states_opt[goal_sval] |=  goal_states_buckets[goal_sval]

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                # unions of all predecessors along with their state values - ADD used for easy printing only
                preimage = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states_opt)
                self.gou_convert_cube_to_state_ADD(preimage, action=False, verbose=True, print_val=True)
            
            if self.check_reached_fixpoint_bdd(curr_winning_states=curr_winning_states, next_winning_states=next_winning_states_opt):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                self._gou_vi_layers = layer
                init_val = math.inf
                for sval, sbdd in curr_winning_states.items():
                    if sbdd & self.gou_init_latch.bddPattern() != self.manager.bddZero():
                        init_val: int = sval
                        print(f"A Cooperative Opt. Strategy Exists!!. The State value is {init_val}")
                        break
                self.cVals = self.convert_vector_of_bdd_to_add(bdd_vector=curr_winning_states)
                # post process the strategy to return as monolithic ADD that corresponds to strategy
                strategy: ADD = self.convert_vector_of_bdd_to_add(bdd_vector=vector_preimage)
                if init_val < math.inf:
                    return strategy.min(goal)
                else:
                    print(f"No Cooperative Opt. Strategy Exists!! The State value is {math.inf}")
                    return None
            
            # update the counter
            layer += 1

            # swap the winning states; can't do  curr_winning_states = next_winning_states_opt as python is pass by value of reference
            curr_winning_states = defaultdict(lambda: self.manager.bddZero())
            for sval in next_winning_states_opt.keys():
                curr_winning_states[sval] |= next_winning_states_opt[sval]
    

    def create_goal_nodes_with_regret_values(self) -> Tuple[ADD, Set[float]]:
        """
         Create Graph of Best-Response nodes with regret values.
        """
        # create ADD(s-u)-Val(u) for every acceting state in game
        uVars_add = self.manager.plusInfinity()
        for u in range(self.budget + 1):
            uVars_add = uVars_add.min(self.uVar_map_sym[f'u{u}'].ite(self.manager.addConst(u), self.manager.plusInfinity()))

        brVars_add = self.manager.plusInfinity()
        for br in self.brVals:
            if br != math.inf:
                brVars_add = brVars_add.min(self.brVar_map_sym[br].ite(self.manager.addConst(br), self.manager.plusInfinity()))

        min_utility_br: ADD = uVars_add.min(brVars_add)
        print("Done taking the min between br and utility values!")

        reg_vals_add: ADD = uVars_add.minus(min_utility_br)
        # process reg values - all invalid uVars conf. map to +inf
        valid_uVars_add = reduce(lambda x, y: x | y, self.uVar_map_sym.values())
        valid_brVars_add = reduce(lambda x, y: x | y, self.brVar_map_sym.values())
        reg_vals_add = valid_uVars_add.ite(reg_vals_add, self.manager.plusInfinity())

        # If uVars is budget + 1, then it is a sink states and hence also maps to +inf regret value
        reg_vals_add = self.uVar_map_sym[f'u{self.budget + 1}'].ite(self.manager.plusInfinity(), reg_vals_add)
        
        # invalid br vals also map to +inf regret value
        reg_vals_add = valid_brVars_add.ite(reg_vals_add, self.manager.plusInfinity())
        rVals = {int(leaf_value) if leaf_value != math.inf else leaf_value for _, leaf_value in reg_vals_add.generate_cubes()} | {0}
        # print(reg_vals)
        print("Processed the Regret Values!")

        goal_add: ADD = self.gobr_goal_latch.ite(reg_vals_add, self.manager.plusInfinity())
        # # now restrict it to the set of valid box conf.
        # goal_add = self.monolithic_relevant_box_preds.ite(goal_add, self.manager.plusInfinity())
        print("Initialized the goal states with regret values!")
        return goal_add, sorted(rVals)


    def regret_solver(self, verbose: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm For computing regret minimizing strategies. 
        """
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        # initialize goal state with respective regret values
        goal, sorted_reg_vals = self.create_goal_nodes_with_regret_values()
        curr_winning_states = goal

        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            preimage: ADD = self.compute_regret_preimage(curr_winning_states)
            next_winning_states = self.compute_min_max_preimage(preimage)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                print("State with regret value zero")
                self.gobr_convert_cube_to_state_ADD((next_winning_states.bddInterval(0, 0) & ~self.dfa_handle.goal_latch.bddPattern()).toADD(), action=False, verbose=True)
                print("State with regret values positive and within budget")
                self.gobr_convert_cube_to_state_ADD(next_winning_states.bddInterval(1, self.budget).toADD(), action=False, verbose=True)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                self._gobr_vi_layers = layer
                if curr_winning_states.restrict(self.gobr_init_latch) != self.manager.plusInfinity():
                    if self.gobr_init_latch & curr_winning_states == self.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((self.gobr_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Regret-Minimizing Strategy Exists!! The State value is {init_val}")
                    self.rVals = curr_winning_states
                    if init_val < math.inf:
                        return preimage.min(goal), self.rVals
                    else:
                        return None, None
                else:
                    print(f"No Regret-Minimizing Strategy Exists!! The State value is {math.inf}")
                return None, None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def hybrid_regret_solver(self, verbose: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm For computing regret minimizing strategies in hybrid fashion. 
        """
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        # initialize goal state with respective regret values
        goal, sorted_reg_vals = self.create_goal_nodes_with_regret_values()
        curr_winning_states = goal
        sorted_reg_vals.remove(math.inf)

        # preprocess TR into buckets of BDDs separated based on actions
        self.gobr_convert_mono_tr_to_action_tr()

        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            win_state_bucket = self.gobr_convert_monolithic_add_to_bdd_buckets(monolithic_add=curr_winning_states, reg_vals=sorted_reg_vals)
            preimage: ADD = self.hybrid_gobr_compute_preimage(win_state_bucket)
            next_winning_states = self.compute_min_max_preimage(preimage)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                print("State with regret value zero")
                self.gobr_convert_cube_to_state_ADD((next_winning_states.bddInterval(0, 0) & ~self.dfa_handle.goal_latch.bddPattern()).toADD(), action=False, verbose=True)
                print("State with regret values positive and within budget")
                self.gobr_convert_cube_to_state_ADD(next_winning_states.bddInterval(1, self.budget).toADD(), action=False, verbose=True)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                self._gobr_vi_layers = layer
                if curr_winning_states.restrict(self.gobr_init_latch) != self.manager.plusInfinity():
                    if self.gobr_init_latch & curr_winning_states == self.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((self.gobr_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Regret-Minimizing Strategy Exists!! The State value is {init_val}")
                    self.rVals = curr_winning_states
                    if init_val < math.inf:
                        return preimage.min(goal), self.rVals
                    else:
                        return None, None
                else:
                    print(f"No Regret-Minimizing Strategy Exists!! The State value is {math.inf}")
                return None, None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def pure_bdd_regret_solver(self, verbose: bool = False) -> Union[ADD, None]:
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))

        # preprocess TR into buckets of BDDs separated based on actions
        self.gobr_convert_mono_tr_to_action_tr()

        goal, sorted_reg_vals = self.create_goal_nodes_with_regret_values()
        sorted_reg_vals.remove(math.inf)

        goal_states_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        curr_winning_states: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval in sorted_reg_vals:
            goal_sval = goal.bddInterval(sval, sval)
            if not goal_sval.isZero():
                goal_states_buckets[sval] |= goal_sval
                curr_winning_states[sval] |= goal_sval

        # initialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            # compute preimage
            vector_preimage: Dict[int, BDD] = self.hybrid_gobr_compute_preimage(win_state_bucket=curr_winning_states, return_bdd=True)            
            next_winning_states_opt = self.compute_min_max_preimage_pure_bdd(vector_preimage, debug=False)
            # as in GoU Solver - goal/sink states in GoBR do not have outgoing transition. We add them back as preimage will not capture them
            # this was taken care by min operation in Pure and Hybrid Approach. Here, we have to do it manually
            for goal_sval in sorted(goal_states_buckets.keys()):
                next_winning_states_opt[goal_sval] |=  goal_states_buckets[goal_sval]

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                # unions of all predecessors along with their state values - ADD used for easy printing only
                preimage = self.convert_vector_of_bdd_to_add(bdd_vector=next_winning_states_opt)
                self.gobr_convert_cube_to_state_ADD(preimage, action=False, verbose=True, print_val=True)
            
            if self.check_reached_fixpoint_bdd(curr_winning_states=curr_winning_states, next_winning_states=next_winning_states_opt):
                print(f"**************************Reached a Fixed Point in {layer} layers**************************")
                self._gobr_vi_layers = layer
                init_val = math.inf
                for sval, sbdd in curr_winning_states.items():
                    if sbdd & self.gobr_init_latch.bddPattern() != self.manager.bddZero():
                        init_val: int = sval
                        print(f"A Regret-Minimizing Strategy Exists!!. The State value is {init_val}")
                        break
                self.rVals = self.convert_vector_of_bdd_to_add(bdd_vector=curr_winning_states)
                # post process the strategy to return as monolithic ADD that corresponds to strategy
                strategy: ADD = self.convert_vector_of_bdd_to_add(bdd_vector=vector_preimage)
                if init_val < math.inf:
                    return strategy.min(goal), self.rVals
                else:
                    print(f"No Regret Minimizing Strategy Exists!! The State value is {math.inf}")
                    return None, None
            
            # update the counter
            layer += 1

            # swap the winning states; can't do curr_winning_states = next_winning_states_opt as python is pass by value of reference
            curr_winning_states = defaultdict(lambda: self.manager.bddZero())
            for sval in next_winning_states_opt.keys():
                curr_winning_states[sval] |= next_winning_states_opt[sval]
    

    def get_next_state_br(self, turn: str, curr_state_exp: List[str], curr_action_sym: ADD, **kwargs) -> ADD:
        """
         A function to compute b' values during rollout in Graph of Best-response game. b' = min{b, br(s, a)}
        """
        curr_br_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        
        if turn == 'sys':
            cube = list(self.monolithic_br.restrict(curr_state_sym & self.state_lbl & curr_action_sym).generate_cubes())
            assert len(cube) == 1, "Make sure there is only one best-alternate response value for the given state-action pair."
            next_br = cube[0][1]
            if next_br <= curr_br_state_val:
                return self.brVar_map_sym[next_br]
        return self.brVar_map_sym[curr_br_state_val]
    

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
        self.invalid_env_state_action_cube = self.transition_relation[self.env_error_cube.bddPattern().__str__()]
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
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.eVars + self.rVars + self.uVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.uVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
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
                sys_err_state = cube.bddPattern().cubeString()[sys_eidx].replace('-', '')
                env_err_state = cube.bddPattern().cubeString()[env_eidx].replace('-', '')
                uVar_state = self.uVar_map.inv[uConf_cube_str]
                state = (([self.tVar_map.inv[tConf_cube_str]] + pos + [sys_err_state, env_err_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state)
                states_action_pairs.append([(((self.tVar_map.inv[tConf_cube_str], *pos,  sys_err_state, env_err_state), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), val), None])
                
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
    

    def gobr_roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a strategy on the graph of utility game.
        """
        curr_state_sym = self.gobr_init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]
        self.invalid_env_state_action_cube = self.transition_relation[self.env_error_cube.bddPattern().__str__()]# | self.invalid_sys_state_action_cube
        while (curr_state_sym & self.gobr_goal_latch).isZero():
            curr_state_exp: List[str] = self.gobr_convert_cube_to_state_ADD(curr_state_sym,
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
                opt_sval = list((curr_state_sym & self.state_lbl & self.rVals).generate_cubes())[0][1]
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

            curr_dfa_state: int = curr_state_exp[0][0][0][0][1]
            turn = 'sys' if curr_state_exp[0][0][0][0][0][0].startswith('sys') else'env'
            try:
                act_name = self.sys_action_map.inv[act_cube_string] if turn == 'sys' else self.env_action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return
           
            # get the next state in the GoU game
            # curr_gou_game_state_sym, act_name = self.get_next_state(turn, curr_state_exp[0], act_name, curr_state_sym=curr_state_sym)
            curr_game_state_sym, act_name, _ = self.gou_get_next_state(curr_state_sym, curr_state_exp[0], act_name)
            curr_gobr_br_sym = self.get_next_state_br(turn, curr_state_exp, curr_state_sym=curr_state_sym, curr_action_sym=act_cube.toADD())
            curr_game_state_sym = curr_game_state_sym & curr_gobr_br_sym
            
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
        
        # the next vars are l' vars - we ignore them for now. The next ones are action vars
        start_rvar_idx, end_rvar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])
        sys_eidx = self.manager.addVariables().index(self.sys_error_cube)
        env_eidx = self.manager.addVariables().index(self.env_error_cube)

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.lVars + self.eVars + self.rVars + self.uVars + self.brVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds] + self.qVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.latches + self.uVars + self.brVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVars + self.lVars + self.rVars + self.brVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        brConf_exist_cube = reduce(lambda a, b: a & b, self.tVars + self.qVars + self.eVars + self.lVars + self.rVars + self.uVars + [var for xVar_adds in self.xVars for var in xVar_adds] + [var for yVar_adds in self.yVars for var in yVar_adds])
        
        xConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            xConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + self.brVars +  self.qVars + self.lVars + self.rVars + [var for yVar_adds in self.yVars for var in yVar_adds]) & reduce(lambda x, y: x & y, self.xVars_cubes[:pidx] + self.xVars_cubes[pidx+1:])

        
        yConf_exist_cube = dict({})
        for pidx in range(self.total_players):
            yConf_exist_cube[pidx] = reduce(lambda a, b: a & b, self.tVars + self.eVars + self.uVars + self.brVars + self.qVars + self.lVars + self.rVars + [var for xVar_adds in self.xVars for var in xVar_adds]) & reduce(lambda x, y: x & y, self.yVars_cubes[:pidx] + self.yVars_cubes[pidx+1:])
        
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
                uVar_state = self.uVar_map.inv[uConf_cube_str]
                brVar_state = self.brVar_map.inv[brConf_cube_str]
                state = ((([self.tVar_map.inv[tConf_cube_str]] + pos + [sys_err_state, env_err_state]), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), brVar_state)
                states_action_pairs.append([((((self.tVar_map.inv[tConf_cube_str], *pos, sys_err_state, env_err_state), self.dfa_handle.qVar_map.inv[qConf_cube_str], uVar_state), brVar_state), val), None])
                
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
        sys_pos = (1, 0)
        sys_cube = self.xVar_map_sym[0][sys_pos[0]] & self.yVar_map_sym[0][sys_pos[1]]
        env_pos = (0, 0)
        env_cube = self.xVar_map_sym[1][env_pos[0]] & self.yVar_map_sym[1][env_pos[1]]
        # goal_cube = sys_cube & env_cube & self.dfa_handle.qVar_map_sym[2] & self.uVar_map_sym['u2'] & self.tVar_map_sym['env0'] & self.not_error_state_cube & self.rVar_map_sym[0] & self.brVar_map_sym[math.inf]
        goal_cube = sys_cube & env_cube & self.dfa_handle.qVar_map_sym[1] & self.tVar_map_sym['sys0'] & self.not_error_state_cube & self.uVar_map_sym['u1'] & self.brVar_map_sym[math.inf]
        print("Goal Cube: ",  goal_cube)
        self.gou_convert_cube_to_state_ADD(goal_cube & self.state_lbl, lbl_flag=True, action=False, verbose=True)

        # preiamge testing on dfa game only
        # preimage =  self.preimage_test(From=goal_cube & self.state_lbl,
        #                    latches=self.latches, prime_latches=self.prime_latches,
        #                    ts_action=list(self.transition_relation.values()))
        # print("Preimage: ", preimage)
        # gou_preimage = self.gou_compute_preimage(goal_cube & self.state_lbl)
        self.graph_of_br_tr = list(self.transition_relation.values())
        self.graph_of_br_tr.extend(list(self.uVars_transition_relation.values()))
        self.graph_of_br_tr.extend(list(self.brVars_transition_relation.values()))
        gobr_preimage = self.compute_regret_preimage(goal_cube & self.state_lbl)

        # gou_preimage = self.gou_compute_preimage(goal_cube & self.state_lbl)
        print("Preimage: ", gobr_preimage)
        self.gou_convert_cube_to_state_ADD(gobr_preimage, action=True, verbose=True)