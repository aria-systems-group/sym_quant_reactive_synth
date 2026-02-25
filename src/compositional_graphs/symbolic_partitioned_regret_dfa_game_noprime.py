import math
import time

from functools import reduce
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Set, Union

from bidict import bidict
from tabulate import tabulate

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.symbolic_partitioned_dfa_game_noprime import SymbolicPartitionedDFAGameNoPrime


class SymbolicPartitionedRegretDFAGameNoPrime(SymbolicPartitionedDFAGameNoPrime):
    """
     This class extends the SymbolicPartitionedDFAGameNoPrime class to construct Graph of Utility (GoU) and Compute Min-Min stratgies and values.

     Unlike the SymbolicPartitionedDFAGame class, here we do not create prime latches.
    """
    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 restricted_human_boxes: List[int],
                 budget: int,
                 ltlf_flag: bool = True,
                 enable_reordering: bool = False):
        self.budget: int = budget
        self.uVars: List[ADD] = []
        self.uVars_bdd: List[BDD] = []
        self.brVars: List[ADD] = []
        self.brVars_bdd: List[BDD] = []
        self.uVar_map: List[ADD] = bidict({}) 
        self.uVar_map_sym: List[ADD] = bidict({}) 
        self.brVar_map: List[ADD] = bidict({}) 
        self.brVar_map_sym: List[ADD] = bidict({})
        self.gou_ts_bdd_transition_fun_list: List[BDD] = []
        self.gobr_ts_bdd_transition_fun_list: List[BDD] = []
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(boxes, locs, ratio, init, goal, formula, restricted_human_locs, restricted_human_boxes, ltlf_flag=ltlf_flag, enable_reordering=enable_reordering)
        self.states_per_cost: Dict[int, ADD] = defaultdict(lambda: self.manager.addZero())
        self.uVars_transition_relation = None
        self.brVars_transition_relation = None
        self.graph_of_utility_tr = None
        self.graph_of_br_tr  = None
        
        # variables for storing optimal cooperative state values and regret optimal state values
        self.cVals = None
        self.rVals = None

        # book keeping
        self.gou_game_latches = self.latches + self.uVars + self.qVars

        self.gobr_game_latches = None
    

    # override the create lacthes method to include Graph of utility latches
    def create_all_boolean_state_vars_and_maps(self):
        """
         The main method that creates all boolean variables for the FrankaDynamic Turn-Based Game.
          1. turn variables - tVars
          2. ratio variables - kVars
          3. predicate variables - pVars
          4. box predicate variables - bVars
          5. Utiltiy variables - uVars
        """
        offset = self.manager.size()
        self.tVar: List[ADD] = [self.manager.addVar(offset, 't0')]
        self.kVars: List[ADD] = self.create_ratio_vars()
        self.pVars, self.bVars = self.create_latches()
        self.uVars = self.create_utility_latches()
        self.uVars_bdd = [u.bddPattern() for u in self.uVars]
        self.create_all_maps()
        self.create_all_sym_maps()

        # create DFA latches next
        self.create_dfa_latches_and_maps()

        # create uVars map
        self.create_utiltity_var_map()
    

    def set_init_latch(self) -> ADD:
        """
        Ovveride the base method. In Graph of Utility, the initial state also includes the utility variable set to 0.

        We assert that the init state should be fully defined, i.e., we need every box's conf. else the init state is a set of states. 
         This causes issue when checking for init state value after VI algorithm terminates.
        """
        assert len(self.init) == self.boxes + 1, "[Error]: The init state should be fully defined for Regret Synthesis code else the Synthesis code will not work correctly."
        init_cube = self.tVar_map_sym['robot'] & self.kVar_map_sym['k0'] & self.uVar_map_sym['u0']
        for s in self.init:
            init_cube &= self.xVar_map_sym[s]
        return init_cube
    
    
    def set_goal_latch(self) -> ADD:
        """
         Ovveride the base method. In Graph of Utility, the goal state also includes any acccepting states in the DFA game and utility values <= budget.
        """
        return (self.dfa_handle.goal_latch & ~self.uVar_map_sym[f'u{self.budget + 1}'])

    
    def create_utility_latches(self) -> List[ADD]:
        """
         Create utility variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        uVar_size = math.ceil(math.log2(self.budget + 2)) # +1 here to account for 0-bit vector and another +1 for  budget +1 which is a sink state.
        # create an additional boolean var to skip the 0-vector latch
        uVar_size = uVar_size + 1 if pow(2, uVar_size) == self.budget + 2 else uVar_size 
        uVars: List[ADD] = [self.manager.addVar(u + varsize, 'u' + str(u)) for u in range(uVar_size)]
        return uVars
    

    def create_br_latches(self) -> List[ADD]:
        """
         Create best-alterante response (br) variables for the Graph of Utility.
        """
        varsize = self.manager.size()
        brVar_size = math.ceil(math.log2(len(self.brVals)))
        # create an additional boolean var to skip the 0-vector latch
        brVar_size = brVar_size + 1 if pow(2, brVar_size) == len(self.brVals) else brVar_size 
        brVars: List[ADD] = [self.manager.addVar(br + varsize, 'br' + str(br)) for br in range(brVar_size)]
        return brVars
    

    def create_utiltity_var_map(self):
        """
         Small function to create symbolic maps for the uVar_map. 
        """
        # budget + 1 is a state to represent budget exceeded; it will be a sink state.
        for u in range(self.budget + 2):
            ubit_str = f"{u + 1:0{len(self.uVars)}b}"
            self.uVar_map[f'u{u}'] = ubit_str
            self.uVar_map_sym[f'u{u}'] = self.cube_to_add(ubit_str, self.uVars)
    
    
    def create_all_br_vars_maps(self):
        """
         Given, the set of best-response values, this method creates all latches and maps for best-alternate response variables. 
        """
        # create variables for best-alternate response values
        self.brVars = self.create_br_latches()
        self.brVars_bdd = [br.bddPattern() for br in self.brVars]
        self.create_br_var_map()
        self.gobr_game_latches = self.latches + self.uVars + self.brVars + self.qVars
    

    def log_game_details(self) -> Dict[str, int]:
        sys_states, env_states = self.get_number_of_states(False)
        abs_dict = {
            'total_latches': len(self.gobr_game_latches) + len(self.rVars),
            'latches': len(self.gobr_game_latches),
            'action_vars': len(self.rVars),
            'turn_vars': len(self.tVar),
            'ratio_vars': len(self.kVars),
            'state_vars': len(self.pVars) + len(reduce(lambda x, y: x + y, self.bVars)),
            'utility_vars': len(self.uVars),
            'ba_vars': len(self.brVars),
            'dfa_latches': len(self.qVars),
            'total_states': sys_states + env_states,
            'game_sys_states': sys_states,
            'game_env_states': env_states,
            'dfa_game_states': self.dfa_handle.num_of_states * (env_states + sys_states),
            'GoU_states': self.dfa_handle.num_of_states * (env_states + sys_states) * (self.budget + 1),
            'GoBR_states': (len(self.brVals) + 1) * (self.budget + 1) * self.dfa_handle.num_of_states * (env_states + sys_states),
            'num_cVals': self.cVals.countLeaves(),
            'brVals': self.brVals,
            }
        return abs_dict
    

    def create_br_var_map(self):
        """
         Small function to create symbolic maps for the brVar_map. 
        """
        for idx, val in enumerate(self.brVals):
            bit_str = f"{idx + 1:0{len(self.brVars)}b}"
            self.brVar_map[val] = bit_str
            self.brVar_map_sym[val] = self.cube_to_add(bit_str, self.brVars)
    

    def get_states_per_cost(self):
        """
         A helper function that takes in the ADD weight abd return a vector of 0-1 ADD per cost.
        """
        lVals = set({0})
        for _, leaf_value in self.weight.generate_cubes():
            if leaf_value != math.inf:
                lVals.add(int(leaf_value))
        
        for val in lVals:
            self.states_per_cost[val] |= self.weight.bddInterval(val, val).toADD() & self.monolithic_relevant_box_preds

    
    def count_actions_per_state_gou(self) -> ADD:
        """
         A function that counts the numbe of actions per state in graph of utility.
        """
        action_cube = reduce(lambda a, b: a & b, self.rVars)
        # convert to BDD and then exist abstract
        # monolithic_valid_state_robot_actions will be over states in the game (s, u) and not over DFA states
        state_action: BDD = self.monolithic_valid_state_robot_actions.bddPattern()

        # convert to 0 - 1 ADD and exist abstract rAct cubes to get ADD(s)->|s'| 
        state_ract_count: ADD = state_action.toADD().existAbstract(action_cube)
        return state_ract_count
    

    def get_states_with_one_outgoing_transition_gou(self) -> BDD:
        """
         This method computes the set of GoU states that have exactly one outgoing robot action.
        """
        gou_state_act_count: ADD = self.count_actions_per_state_gou()
        bdd_gou_state_single_act: BDD = gou_state_act_count.bddInterval(1, 1)

        # as accepting states in DFA are sink states in GoU, we need to post-process the gou_state_act_count so that accepting states map to cardinality 1.
        bdd_gou_state_single_act |= (self.dfa_handle.goal_latch & gou_state_act_count).bddPattern()
        return bdd_gou_state_single_act
    

    def create_utility_transition_relation(self):
        """
         Create the transition relation for utility variables.
        """
        self.uVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.uVars}
        self.get_states_per_cost()
        for u in range(self.budget + 1):
            uConf_cube = self.uVar_map_sym[f'u{u}']
            for state_cost in self.states_per_cost.keys():
                transition_cube = uConf_cube & self.states_per_cost[state_cost]
                
                prime_u_val = u + state_cost if (u + state_cost) <= self.budget else self.budget + 1 
                uConf_cube_str = self.uVar_map[f'u{prime_u_val}']
                for sidx, s in enumerate(uConf_cube_str):
                    if s == '1':
                        self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |= transition_cube                
        
        # add self-loop for the sink state budget + 1
        for sidx, s in enumerate(self.uVar_map[f'u{self.budget + 1}']):
            if s == '1':
                self.uVars_transition_relation[self.uVars[sidx].bddPattern().__str__()] |=  self.uVar_map_sym[f'u{self.budget + 1}']


    def compute_best_alternate_response(self, verbose: bool = False) -> None:
        """
        A method to compute the best alterante response (ba). Given, tuple (s, s'), best-alternate response is the scalar value associated with:
            Informal: What if I took any other valid edge from (s, s'') where s'' =\= s' for every Sys player state.
            Mathematically, given cVal (cooperative value) for every state s in G, we have

            ba(s, s') = +inf if s is Env player states
            ba(s, s') = min (s, s'') {cVal(s'')} if s is Sys plaeyr states

            min(s, s'') = +inf if no s'' exists, i.e., there does not exist an alternate edge.
        
        Output ADD(s, as)-br where br is the best-response.

        1. First compute ADD(s, a)-cVal(s') by vectorCompose-ing cVal(s) over GoU transition relation.
        2. For each robot action, mask out the action from ADD(s, a)-cVal(s') and compute min over rVars to get ADD(s)-ba_per_act
        3. For each leaf node in ADD(s)-ba_per_act, chop the ADD into BDD(s) & ract and store them in vector_of_br
        """        
        # compute preimage of ADD(s')-cVal to get ADD(s, a)-cVal(s')
        dfa_preimage = self.cVals.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        game_state_action = dfa_preimage.vectorCompose(self.latches + self.uVars, self.graph_of_utility_tr) 

        # any state with env action has infinity value. So, we mask them out
        game_state_action = self.tVar_map_sym['robot'].ite(game_state_action, self.manager.plusInfinity())

        # now compute the best alternate response
        self.vector_of_br = defaultdict(lambda: self.manager.addZero())
        
        lVals = {*range(0, self.budget + 1)} | {math.inf}
        for ract, ract_sym in self.relevant_robot_actions_sym.items():
            print(f"Computing BR for Robot Act: {ract}")
            
            game_state_action_without_ract = ract_sym.ite(self.manager.plusInfinity(), game_state_action)
            ba_per_act = self.symbolic_min_abstract(game_state_action_without_ract, self.rVars)

            # chop the ADDs into vector of BDD(s), one for each leaf node
            for leaf_val in lVals:
                bdd_state_act = (ba_per_act.bddInterval(leaf_val, leaf_val))
                if not bdd_state_act.isZero():
                    bdd_state_act = bdd_state_act & ract_sym.bddPattern()
                    # remove goal states from br computation; later we add them to +inf br value
                bdd_state_act = bdd_state_act & ~self.dfa_handle.goal_latch.bddPattern()  
                self.vector_of_br[leaf_val] |= bdd_state_act.toADD() & self.monolithic_valid_state_robot_actions
        
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
    

    def create_best_alternate_response_transition_relation(self):
        self.brVars_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.brVars}
        
        for br in self.brVals:
            brConf_cube = self.brVar_map_sym[br]
            
            for prime_br in self.brVals:
                if prime_br <= br:
                    # create the transition cube
                    transition_cube = brConf_cube & self.vector_of_br[prime_br]
                    
                    brConf_cube_str = self.brVar_map[prime_br]
                    for sidx, s in enumerate(brConf_cube_str):
                        if s == '1':
                            self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= transition_cube
                    
                else:
                    state_act_pairs = self.manager.addZero()
                    for i in self.brVals:
                        if i > br:
                            state_act_pairs |= self.vector_of_br[i]
                    
                    transition_cube = brConf_cube & state_act_pairs
                
                    brConf_cube_str = self.brVar_map[br]
                    for sidx, s in enumerate(brConf_cube_str):
                        if s == '1':
                            self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= transition_cube
                    break
        
            # add that from human states, the best-alternate response remains the same
            human_transition_cube = brConf_cube & self.tVar_map_sym['human']
            brConf_cube_str = self.brVar_map[br]
            for sidx, s in enumerate(brConf_cube_str):
                if s == '1':
                    self.brVars_transition_relation[self.brVars[sidx].bddPattern().__str__()] |= human_transition_cube
    

    def create_transition_relation(self):
        # creates game transition relation first and then dfa transition relation
        super().create_transition_relation()

        # now create utility transition relation
        self.create_utility_transition_relation()

        tic = time.time()
        # strategy = self.gou_solve(verbose=False, optimized=False)
        # strategy = self.hybrid_gou_solve(verbose=True)
        # self.TVI_gou_solve(verbose=False, optimized=False)
        strategy = self.pure_bdd_gou_solve(verbose=False)
        toc = time.time()
        print(f"Time to synthesize GOU values: {toc - tic} seconds")

        # if strategy is not None:
        #     self.gou_roll_out_strategy(strategy=strategy, verbose=True)
        # return

        # compute best-alternate response
        self.compute_best_alternate_response(verbose=False)

        # create boolean vars for Best-alternate response values computed
        self.create_all_br_vars_maps()

        tic = time.time()
        self.create_best_alternate_response_transition_relation()
        toc = time.time()
        print(f"Time to create GoBR Transition Relation: {toc - tic} seconds")
    

    def create_goal_nodes_with_utility_values(self, verbose: bool = False) -> ADD:
        """
         Create Graph of Utility's accting nodes with utility values.
        """
        uVars_add = self.manager.plusInfinity()
        # skip u = 0; we reason about u = 0 separately
        for u in range(1, self.budget + 1):
            uVars_add = uVars_add.min(self.uVar_map_sym[f'u{u}'].ite(self.manager.addConst(u), self.manager.plusInfinity()))

        dfa_goal_cube: ADD = self.goal_latch.ite(self.manager.addOne(), self.manager.plusInfinity())
        goal_add: ADD = dfa_goal_cube.times(uVars_add)
        final_goal_add = (self.uVar_map_sym[f'u{0}'] & self.dfa_handle.goal_latch).ite(self.manager.addZero(), goal_add)
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
            self.sys_gou_transition_relation.append(tr_dd & self.tVar_map_sym['robot'])
            self.env_gou_transition_relation.append(tr_dd & self.tVar_map_sym['human'])
    

    def compute_preimage(self, curr_winning_states: ADD) -> ADD:
        """
         Preimage comptuation over the Graph of Utility transition relation.
        """
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        # dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        preimage = dfa_preimage.vectorCompose(self.latches + self.uVars, self.graph_of_utility_tr)

        return preimage
    

    def compute_regret_preimage(self, curr_winning_states: ADD) -> ADD:
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the DFA game state (s, u)
        preimage_subr: ADD = dfa_preimage.vectorCompose(self.latches + self.uVars + self.brVars, self.graph_of_br_tr)
        self.iteration_bookkeeping.append([dfa_preimage.size(), preimage_subr.size()])

        return preimage_subr

    
    def iros23_gou_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        for sval, succ_states in win_state_bucket.items():
            # first evolve over the DFA
            # dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))
            pre_states: BDD = dfa_preimage.vectorCompose(self.latches_bdd + self.uVars_bdd, self.gou_ts_bdd_transition_fun_list)

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
    

    def iros23_gobr_compute_preimage(self, win_state_bucket: Dict[int, BDD], return_bdd: bool = False) -> Union[ADD, Dict[int, BDD]]:
        pre_buckets: Dict[int, BDD] = defaultdict(lambda: self.manager.bddZero())
        bookkeeping_size =  defaultdict(lambda: defaultdict(list))
        for sval, succ_states in win_state_bucket.items():
            # first evolve over the DFA
            # dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_bdd.values()))
            dfa_preimage: BDD = succ_states.vectorCompose(self.qVars_bdd, list(self.dfa_handle.dfa_transition_relation_accp_sink_bdd.values()))
            pre_states: BDD = dfa_preimage.vectorCompose(self.latches_bdd + self.uVars_bdd + self.brVars_bdd, self.gobr_ts_bdd_transition_fun_list)

            if not pre_states.isZero():
                assert pre_buckets[sval] & pre_states == self.manager.bddZero(), "Make sure there are no overlapping states in the pre buckets..."
                pre_buckets[sval] |= pre_states
                bookkeeping_size[sval] = [dfa_preimage.size(), pre_states.size()]

        self.iteration_bookkeeping.append(bookkeeping_size)
        # unions of all predecessors
        if not return_bdd:
            preimage = self.manager.plusInfinity()
            for sval, add_bucket in pre_buckets.items():
                preimage = add_bucket.toADD().ite(self.manager.addConst(sval), preimage)
            
            return preimage
        return pre_buckets
    

    def compute_preimage_optimized(self, curr_winning_states: ADD) -> Tuple[ADD, ADD]:
        """
         Optimized preimage comptuation over the Graph of Utility transition relation.
        """
        # first evolve over the DFA
        dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation.values()))
        # dfa_preimage: ADD = curr_winning_states.vectorCompose(self.qVars, list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))

        # then evolve over the game
        pre_sys = dfa_preimage.vectorCompose(self.latches + self.uVars, self.sys_gou_transition_relation)
        pre_env = dfa_preimage.vectorCompose(self.latches + self.uVars, self.env_gou_transition_relation)

        return pre_sys, pre_env


    def gou_solve(self, verbose: bool = False, optimized: bool = False) -> Optional[ADD]:
        # extende the DFA game TR to construct TR for Graph of Utility that includes uVars
        self.graph_of_utility_tr = list(self.transition_relation.values())
        self.graph_of_utility_tr.extend(list(self.uVars_transition_relation.values()))
        if optimized:
            self.create_gou_sys_env_transition_relations()
        
        goal = self.create_goal_nodes_with_utility_values(verbose=verbose)
        # print("Goal States with Utility values:", goal)
        curr_winning_states =  self.manager.plusInfinity()
        curr_winning_states = curr_winning_states.min(goal)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")
            if optimized:
                pre_sys, pre_env = self.compute_preimage_optimized(curr_winning_states)
                next_winning_states_sys = self.symbolic_min_abstract(pre_sys, variables_to_abstract=self.rVars)
                next_winning_states_env = self.symbolic_min_abstract(pre_env, variables_to_abstract=self.rVars)
                next_winning_states = self.tVar[0].ite(next_winning_states_sys, next_winning_states_env)
            else:
                preimage: ADD = self.compute_preimage(curr_winning_states)
                next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.gou_convert_cube_to_state_ADD(next_winning_states, action=False, verbose=True, print_val=True)
            
            # if curr_winning_states.compare(next_winning_states, 2):
            if next_winning_states.compare(curr_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states != self.manager.plusInfinity():
                    if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Cooperation Strategy Exists!!. The State value is {init_val}")
                    self.cVals = curr_winning_states
                    if optimized:
                        preimage = pre_sys.min(pre_env)
                    return preimage.min(goal) if init_val < math.inf else None
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
    

    def regret_solver(self, verbose: bool = False, optimized: bool = False) -> Union[ADD, None]:
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
        regret_init_latch = self.init_latch & self.brVar_map_sym[math.inf]
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        # iteration bookkeeping
        self.iteration_bookkeeping = []

        while True:
            print(f"**************************Layer: {layer}**************************")
            if optimized:
                preimage = self.manager.plusInfinity()
                # preprocess the preimage to split into "buckets" of ADD with same values
                for reg_val in sorted_reg_vals:
                    curr_winning_states_reg_val = curr_winning_states.bddInterval(reg_val, reg_val).toADD()
                    preimage_reg_val = self.compute_regret_preimage(curr_winning_states_reg_val)
                    preimage = preimage_reg_val.ite(self.manager.addConst(reg_val), preimage)
            else:
                preimage: ADD = self.compute_regret_preimage(curr_winning_states)
            
            next_winning_states = self.compute_min_max_preimage(preimage, valid_human_action_mask=valid_human_action_mask)
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
                self.logger.comp_time['Iterations'] = layer
                self.logger.comp_time['Preimage_size'] = {idx: e for idx, e in enumerate(self.iteration_bookkeeping)}
                if curr_winning_states.restrict(self.dfa_handle.init_latch & regret_init_latch) != self.manager.plusInfinity():
                    if self.dfa_handle.init_latch & regret_init_latch & curr_winning_states == self.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & regret_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!! The State value is {init_val}")
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
        A method that implements the value iteration algorithm For computing regret minimizing strategies. 
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

        # initialize the iteration counter
        layer = 0
        regret_init_latch = self.init_latch & self.brVar_map_sym[math.inf]
        valid_human_action_mask = reduce(lambda x, y: x | y, self.env_action_cube_list)
        self.iteration_bookkeeping = []

        while True:
            print(f"**************************Layer: {layer}**************************")
            win_state_bucket = self.gobr_convert_monolithic_add_to_bdd_buckets(monolithic_add=curr_winning_states, reg_vals=sorted_reg_vals)
            preimage: ADD = self.iros23_gobr_compute_preimage(win_state_bucket)
            
            next_winning_states = self.compute_min_max_preimage(preimage, valid_human_action_mask=valid_human_action_mask)
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
                self.logger.comp_time['action_list'] = list(self.action_map_sym.keys())
                self.logger.comp_time['Iterations'] = layer
                self.logger.comp_time['Preimage_size'] = {idx: e for idx, e in enumerate(self.iteration_bookkeeping)}
                if curr_winning_states.restrict(self.dfa_handle.init_latch & regret_init_latch) != self.manager.plusInfinity():
                    if self.dfa_handle.init_latch & regret_init_latch & curr_winning_states == self.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & regret_init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!! The State value is {init_val}")
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
        regret_init_latch: ADD = self.init_latch & self.brVar_map_sym[math.inf]
        self.iteration_bookkeeping = []

        while True:
            print(f"**************************Layer: {layer}**************************")
            # compute preimage
            vector_preimage: Dict[int, BDD] = self.iros23_gobr_compute_preimage(win_state_bucket=curr_winning_states, return_bdd=True)            
            next_winning_states_opt = self.compute_min_max_preimage_pure_bdd(vector_preimage, debug=False)
            # as GoU Solver - goal/sink states in GoBR do not have outgoing transition. We add them back as preimage will not capture them
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
                init_val = math.inf
                self.logger.comp_time['action_list'] = list(self.action_map_sym.keys())
                self.logger.comp_time['Iterations'] = layer
                self.logger.comp_time['Preimage_size'] = {idx: e for idx, e in enumerate(self.iteration_bookkeeping)}
                for sval, sbdd in curr_winning_states.items():
                    if sbdd & (self.dfa_handle.init_latch & regret_init_latch).bddPattern() != self.manager.bddZero():
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


    def gou_convert_mono_tr_to_action_tr(self):
        # loop throught the transition relation and separate them based on action
        for tr_bdd in self.graph_of_utility_tr:
            self.gou_ts_bdd_transition_fun_list.append(tr_bdd.bddPattern())
    

    def gobr_convert_mono_tr_to_action_tr(self):
        # loop throught the transition relation and separate them based on action
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
        # process reg values - all invalid uvArs conf. map to +inf
        valid_uVars_add = reduce(lambda x, y: x | y, self.uVar_map_sym.values())
        valid_brVars_add = reduce(lambda x, y: x | y, self.brVar_map_sym.values())
        reg_vals_add = valid_uVars_add.ite(reg_vals_add, self.manager.plusInfinity())

        # If uVars is buget + 1, then it is a sink states and hence also maps to +inf regret value
        reg_vals_add = self.uVar_map_sym[f'u{self.budget + 1}'].ite(self.manager.plusInfinity(), reg_vals_add)
        
        # invalid br vals also map to +inf regret value
        reg_vals_add = valid_brVars_add.ite(reg_vals_add, self.manager.plusInfinity())
        rVals = {int(leaf_value) if leaf_value != math.inf else leaf_value for _, leaf_value in reg_vals_add.generate_cubes()} | {0}
        # print(reg_vals)
        print("Processed the Regret Values!")

        goal_add: ADD = self.goal_latch.ite(reg_vals_add, self.manager.plusInfinity())
        # now restrict it to the set of valid box conf.
        goal_add = self.monolithic_relevant_box_preds.ite(goal_add, self.manager.plusInfinity())
        print("Initialized the goal states with regret values!")
        return goal_add, sorted(rVals)
    

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
            preimage: ADD = self.iros23_gou_compute_preimage(win_state_bucket)
            next_winning_states = self.symbolic_min_abstract(preimage, variables_to_abstract=self.rVars)
            
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.gou_convert_cube_to_state_ADD(next_winning_states, action=False, verbose=True, print_val=True)
            
            # if curr_winning_states.compare(next_winning_states, 2):
            if next_winning_states.compare(curr_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states != self.manager.plusInfinity():
                    if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Cooperation Strategy Exists!!. The State value is {init_val}")
                    self.cVals = curr_winning_states
                    return preimage.min(goal) if init_val < math.inf else None
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
            vector_preimage: Dict[int, BDD] = self.iros23_gou_compute_preimage(win_state_bucket=curr_winning_states, return_bdd=True)

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
                init_val = math.inf
                for sval, sbdd in curr_winning_states.items():
                    if sbdd & (self.dfa_handle.init_latch & self.init_latch).bddPattern() != self.manager.bddZero():
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
    

    def gou_roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a strategy on the graph of utility game.
        """
        curr_state_sym = self.init_latch
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]

        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            curr_state_exp: List[str] = self.gou_convert_cube_to_state_ADD(curr_state_sym,
                                                                           state_flag=True,
                                                                           action=False,
                                                                           verbose=False,
                                                                           table_header=False,
                                                                           print_val=False)
            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.cVals).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            if verbose:
                # print(tabulate([(curr_state_exp[0][0][0], opt_sval)], headers=['Current State', 'Optimal State Value']))
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)])) 

            turn = 'robot' if curr_state_exp[0][0][0][0][0] == 'robot' else'human'
            act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return
           
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
    

    def gobr_roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a strategy on the graph of utility game.
        """
        curr_state_sym = self.init_latch & self.dfa_handle.init_latch & self.brVar_map_sym[math.inf]
        rVars_bdd: List[BDD] = [var.bddPattern() for var in self.rVars]

        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            curr_state_exp: List[str] = self.gobr_convert_cube_to_state_ADD(curr_state_sym,
                                                                            state_flag=True,
                                                                            action=False,
                                                                            verbose=False,
                                                                            table_header=False,
                                                                            print_val=False)

            assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
            "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.rVals).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            if verbose:
                # print(tabulate([(curr_state_exp[0][0][0], opt_sval)], headers=['Current State', 'Optimal State Value']))
                print(tabulate([(curr_state_exp[0][0][0], opt_sval)])) 

            turn = 'robot' if curr_state_exp[0][0][0][0][0][0] == 'robot' else'human'

            # for state with optimal state value of 0, the restruct operation returns zero ADD.
            act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(rVars_bdd)
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.action_map.inv[act_cube_string]
            except KeyError:
                print("No action found!!")
                return
           
            # get the next state in the GoU game
            curr_gou_game_state_sym, act_name = self.get_next_state(turn, curr_state_exp[0], act_name, curr_state_sym=curr_state_sym)
            curr_gobr_br_sym = self.get_next_state_br(turn, curr_state_exp, curr_state_sym=curr_state_sym, curr_action_sym=act_cube.toADD())
            curr_game_state_sym = curr_gou_game_state_sym & curr_gobr_br_sym
            
            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            curr_dfa_state: int = curr_state_exp[0][0][0][0][1]
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
    

    def get_next_state(self, turn: str, curr_state_exp: List[str], act_name: str, **kwargs) -> Tuple[ADD, str]:
        # get the next state in the game in explicit form
        curr_game_state = list(curr_state_exp[0][0][0][0])
        curr_utl_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        if turn == 'robot':
            curr_game_state_sym: ADD = self.get_next_state_robot(curr_game_state, act_name, curr_state_utl=curr_utl_state_val, curr_state_sym=curr_state_sym)
        else:
            curr_game_state_sym, act_name = self.get_next_state_human(curr_game_state, act_name, curr_state_utl=curr_utl_state_val)
        
        return curr_game_state_sym, act_name
    

    def get_next_state_br(self, turn: str, curr_state_exp: List[str], curr_action_sym: ADD, **kwargs) -> ADD:
        """
         A function to compute b' values during rollout in Graph of Best-response game. b' = min{b, br(s, a)}
        """
        curr_br_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        
        if turn == 'robot':
            cube = list(self.monolithic_br.restrict(curr_state_sym & curr_action_sym).generate_cubes())
            assert len(cube) == 1, "Make sure there is only one best-alternate response value for the given state-action pair."
            next_br = cube[0][1]
            if next_br <= curr_br_state_val:
                return self.brVar_map_sym[next_br]
        return self.brVar_map_sym[curr_br_state_val]
    

    def get_next_state(self, turn: str, curr_state_exp: List[str], act_name: str, **kwargs) -> Tuple[ADD, str]:
        # get the next state in the game in explicit form
        curr_game_state = list(curr_state_exp[0][0][0][0])
        curr_utl_state_val = curr_state_exp[0][0][0][-1]
        curr_state_sym = kwargs['curr_state_sym']
        if turn == 'robot':
            next_state_dfa_game: ADD = self.get_next_state_robot(curr_game_state, act_name)
        else:
            next_state_dfa_game, act_name = self.get_next_state_human(curr_game_state, act_name)
        
        # update uVal - get the next utility value based on the current state and action
        # get the state cost
        if self.weight.cofactor(curr_state_sym & self.action_map_sym[act_name]).isZero():
            state_cost: int = 0
        else:
            state_cost: int = int(list((self.weight.cofactor(curr_state_sym & self.action_map_sym[act_name])).generate_cubes())[0][1])
        state_utl: int = int(curr_utl_state_val[1:])

        if state_utl + state_cost <= self.budget:
            next_uVar_sym = self.uVar_map_sym[f'u{state_utl + state_cost}']
        else:
            next_uVar_sym = self.uVar_map_sym[f'u{self.budget + 1}']
        
        return next_state_dfa_game & next_uVar_sym, act_name
    

    def gou_convert_cube_to_state_ADD(self,
                                      dd: ADD, state_flag: bool = True,
                                      dfa_flag: bool = True, action: bool = False,
                                      verbose: bool = False, table_header: bool = True, print_val: bool = True) -> None:
        """
         Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] + self.tVar + self.kVars + self.uVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)

        headers = []
        if verbose:
            headers.extend(['state'])
            if action:
                headers.append('action')
            if print_val:
                headers.append('value')
        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.rVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.rVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.rVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.rVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.rVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = []
        for cube, val in cubes:
            state = None
            tConf_exist_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_exist_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_exist_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_exist_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_exist_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = ((self.tVar_map.inv[tConf_exist_str],
                          self.kVar_map.inv[kConf_exist_str],
                          self.pVar_map.inv[rConf_exist_str], box_states),
                          self.dfa_handle.qVar_map.inv[qConf_exist_str], self.uVar_map.inv[uConf_exist_str])
                states_action_pairs.append([
                    (((self.tVar_map.inv[tConf_exist_str],
                       self.kVar_map.inv[kConf_exist_str],
                       self.pVar_map.inv[rConf_exist_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_exist_str], self.uVar_map.inv[uConf_exist_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(oCube_str, self.rVars)]
                except KeyError:
                    continue
            
            if action:
                if print_val:
                    states_bookkeeping.append((state, action_str, val))
                # states_bookkeeping.append((state, action_str, val))
                else:
                    states_bookkeeping.append((state, action_str))
            else:
                if print_val:
                    states_bookkeeping.append((state, val))
                else:
                    states_bookkeeping.append(state)
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))

        return states_action_pairs


    def gobr_convert_cube_to_state_ADD(self,
                                       dd: ADD, state_flag: bool = True,
                                       dfa_flag: bool = True, action: bool = False,
                                       verbose: bool = False, table_header: bool = True, print_val: bool = True) -> None:
        """
        Convert Graph of Best-Response state-action cubes to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] + self.tVar + self.kVars + self.uVars + self.brVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if action:
            relevant_vars.extend(self.rVars) # action vars (rVars)
        
        headers = []
        if verbose:
            headers.extend(['state'])
            if action:
                headers.append('action')
            if print_val:
                headers.append('value')

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.rVars[0]), self.manager.addVariables().index(self.rVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.uVars + self.brVars + self.rVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.uVars + self.brVars + self.rVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.uVars + self.brVars + self.rVars)
        uConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.brVars + self.rVars)
        brConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.qVars + self.uVars + self.rVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.uVars + self.brVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.rVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.brVars + self.rVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.uVars + self.brVars + self.rVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = []
        states_bookkeeping = []
        for cube, val in cubes:
            state = None
            action_str = None
            tConf_cube_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
            kConf_cube_str = cube.existAbstract(kConf_exist_cube).bddPattern().cubeString().replace('-', '')
            qConf_cube_str = cube.existAbstract(qConf_exist_cube).bddPattern().cubeString().replace('-', '')
            uConf_cube_str = cube.existAbstract(uConf_exist_cube).bddPattern().cubeString().replace('-', '')
            brConf_cube_str = cube.existAbstract(brConf_exist_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for e in bConf_exist_cube.values():
                bCube_str.append(cube.existAbstract(e).bddPattern().cubeString().replace('-', ''))
            
            try:
                box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            except KeyError:
                continue
            
            try:
                state = (((self.tVar_map.inv[tConf_cube_str],
                          self.kVar_map.inv[kConf_cube_str],
                          self.pVar_map.inv[rConf_cube_str], box_states),
                          self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), self.brVar_map.inv[brConf_cube_str])
                states_action_pairs.append([
                    ((((self.tVar_map.inv[tConf_cube_str],
                       self.kVar_map.inv[kConf_cube_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_cube_str], self.uVar_map.inv[uConf_cube_str]), self.brVar_map.inv[brConf_cube_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    action_str = self.action_map_sym.inv[self.cube_to_add(oCube_str, self.rVars)]
                except KeyError:
                    continue
            
            if action:
                states_bookkeeping.append((state, action_str, val))
            else:
                states_bookkeeping.append((state, val))
        
        if verbose and table_header:
            print(tabulate(states_bookkeeping, headers=headers))
        elif verbose and not table_header:
            print(tabulate(states_bookkeeping))

        return states_action_pairs