import re
import sys
import math
import warnings
import graphviz as gv

from typing import List, Tuple

from src.explicit_graphs import TwoPlayerGame
from src.explicit_graphs import Ltlf2MonaDFA

from bidict import bidict
from cudd import Cudd, ADD, BDD


from config import *

class SymbolicPartitionedDFA():
    """
    Base Class: 
      A class where we constuct a Symbolic DFA in partitioned form, i.e., we have a set of boolean variables (latches)
      and another set of boolane vairable (prime latches). The latches correspond to the current state in the DFA. 
      The prime latches are used during strategy synthesis.
     
      This class is  different from SymbolicDFA in the sense that I am create two copies of the current state variables. 
      SymbolicDFA create two copis for curr and next state. The PartitionedDFA and ADDPartitionedDFA only create latches
      and not prime latches.

      The intherited class overrides the create_dfa_transition_relation() method to construct the transition relation
      in partitioned form.
    """

    def __init__(self, formula: str, manager: Cudd, latches_map: bidict, game_latches: List[ADD], prime_game_latches: List[ADD], dfa_name: str = 'dfa'):
        self.formula: str = formula
        self.predicate_add_sym_map_lbl = latches_map
        self.game_latches = game_latches
        self.prime_game_latches: List[ADD] = prime_game_latches
        self.dfa_name: str = dfa_name
        self.manager: Cudd = manager
        self.dfa, self.num_of_states = self.formula_to_automaton()
        # create valid transitions of DFA 
        # curr dfa state (q) --- prime game state (ps) ---> to next dfa state (pq)
        self.monolithic_valid_q_ps_pq: ADD = self.manager.addZero()

        # initialize handles for dfa latches, prime latches and maps
        self.qVars: List[ADD] = []
        self.prime_qVars: List[ADD] = []
        # initialize str and sym map
        self.qVar_map = bidict({})
        self.qVar_map_sym = bidict({})
        self.prime_qVar_map_sym = bidict({})
        self.init_latch: ADD = self.manager.addZero()
        self.goal_latch: ADD = self.manager.addZero()        

        # intialize transition relation handle
        self.dfa_transition_relation = {}

        # set the initial and goal states in explicit form
        self.set_init_goal_states()
    

    def formula_to_automaton(self):
        raise NotImplementedError()

    def set_init_goal_states(self):
        raise NotImplementedError()

    def set_init_latch(self):
        raise NotImplementedError()
    
    def set_goal_latch(self):
        raise NotImplementedError()

    
    def create_latches_and_map(self):
        varsize = self.manager.size()
        qVars_size: int = math.ceil(math.log2(self.num_of_states))
        qVars_size = qVars_size + 1 if pow(2, qVars_size) == self.num_of_states else qVars_size
        self.qVars: List[ADD] = [self.manager.addVar(k + varsize, 'q' + str(k)) for k in range(qVars_size)]
        self.create_qVar_map()
        self.qVar_map_sym = bidict({k: self.cube_to_add(v, self.qVars) for k, v in self.qVar_map.items()})

    def create_prime_latches(self):
        """
         We create prime latches for the DFA states. Note that we do not create mapping for prime latches as they are not needed.
        """
        varsize = self.manager.size()
        self.prime_qVars: List[ADD] = [self.manager.addVar(k + varsize, 'pq' + str(k)) for k in range(len(self.qVars))]
        self.prime_qVar_map_sym = bidict({k: self.cube_to_add(v, self.prime_qVars) for k, v in self.qVar_map.items()})
    
    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        """
         Tiny helper function to convert a cube string to ADD representation. Copied as is from Compositional Abstraction Classes like
          FrankaWorldDyanmicRatioTurnBased(), FrankaWorldDyanmicTurnBased() etc.
        """
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add

    def create_qVar_map(self):
        # offseting the value to skip the 0-bit vector
        offset = 1
        for q_idx, q in enumerate(self.dfa._graph.nodes()):
            bit_str = f"{offset + q_idx:0{len(self.qVars)}b}"
            self.qVar_map[q] = bit_str
    

    def create_dfa_transition_relation(self):
        raise NotImplementedError()


class SymbolicPartitionedDFAFromSpot(SymbolicPartitionedDFA):
    """
    This class inherits the SymbolicPartitionedDFA and implements the following:

     1. Calling the appropriate tool (for now only SPOT) for constructing the automaton
     2. Implementing the init and goal state initialization method for LTL formula
     3. Implementing the create_dfa_transition_relation() method for constructing the transition relation from SPOT DFA 
    """
    
    def __init__(self, formula: str, manager: Cudd, latches_map: bidict, game_latches: List[ADD], prime_game_latches: List[ADD]):
        super().__init__(formula=formula, manager=manager, latches_map=latches_map, game_latches=game_latches, prime_game_latches=prime_game_latches)
        self.valid_dfa_edge_formula_size: int = len(self.dfa.get_symbols())


    def formula_to_automaton(self): 
        # Call SPOT, Construct Graph using our PDDLtoSim toolbox, and return the full graph that has the nodes and edges
        two_player_instance = TwoPlayerGame(None, None)
        dfa = two_player_instance.build_LTL_automaton(formula=self.formula, plot=False)
        state = dfa.get_states()
        num_of_states = len(state)

        return dfa, num_of_states

    def set_init_goal_states(self):
        self.init: str = self.dfa.get_initial_states()[0][0]
        self.goal: str = self.dfa.get_accepting_states()[0]
    

    def set_init_latch(self):
        self.init_latch |= self.qVar_map_sym[self.init]
    

    def set_goal_latch(self):
        self.goal_latch |= self.qVar_map_sym[self.goal]
    

    def find_symbols(self, formula: str):
        """
        Find symbols associated with an edge
        """
        regex = re.compile(r"[a-z]+[a-z0-9]*")
        matches = regex.findall(formula)
        symbols = list()
        for match in matches:
            symbols += [match]
        symbols = list(set(symbols))
        symbols.sort()
        return symbols
    

    def in_order_nnf_tree_traversal(self, expression, formula) -> ADD:
        """
        Traverse the edge formula given by Promela a binary tree. This function implements a in-order tree traversal algorithm.
        """
        if hasattr(formula, 'symbol'):
            # get the corresponding boolean expression
            if '!' in formula.name:
                box_loc: str = re.search(r'\d+', formula.name).group()
                return ~self.predicate_add_sym_map_lbl[f'b{box_loc[0]} l{box_loc[1]}']
            else:
                box_loc: str = re.search(r'\d+', formula.name).group()
                return self.predicate_add_sym_map_lbl[f'b{box_loc[0]} l{box_loc[1]}']
        
        expression = self.in_order_nnf_tree_traversal(expression, formula.left)
        if formula.name == 'AND':
            expression = expression & self.in_order_nnf_tree_traversal(expression, formula.right)
        elif formula.name == 'OR':
            expression |= self.in_order_nnf_tree_traversal(expression, formula.right)

        return expression
    

    def get_edge_boolean_formula(self, curr_state, nxt_state) -> ADD:
        """
        Given an edge, extract the string and construct the boolean formula associated with this string 
        """        
        _guard = self.dfa._graph[curr_state][nxt_state][0]['guard']
        _guard_formula = self.dfa._graph[curr_state][nxt_state][0]['guard_formula']

        symbls =  self.find_symbols(_guard_formula)

        # if symbls is empty then create True edge
        if not symbls or 'true' in symbls:
            return self.manager.addOne()
        else:
            if len(symbls) > self.valid_dfa_edge_formula_size:
                return self.manager.addZero()

            elif len(symbls) <= self.valid_dfa_edge_formula_size:
                edgy_formula: ADD = self.in_order_nnf_tree_traversal(expression=self.manager.addZero() , formula=_guard)
                return edgy_formula
    

    def create_dfa_transition_relation(self):
        self.dfa_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.qVars}
        for curr, nxt in self.dfa._graph.edges():
            # get the boolean formula for the corresponding edge 
            dfa_state_cube = self.qVar_map_sym[curr] 
            dfa_state_prime_str: str = self.qVar_map[nxt]
            edge_sym = self.get_edge_boolean_formula(curr_state=curr,
                                                      nxt_state=nxt)
            
            if not isinstance(edge_sym, ADD):
                edge = self.dfa._graph[curr][nxt][0]['guard_formula']
                warnings.warn(f"Error while parsing the LTL Formula. Could not parse edge {edge}")
                sys.exit(-1)
            
            self.monolithic_valid_q_ps_pq |= dfa_state_cube & edge_sym.swapVariables(self.game_latches, self.prime_game_latches) & self.prime_qVar_map_sym[nxt]
            
            # now we add the transition dfa's transition relation
            for sidx, s in enumerate(dfa_state_prime_str):
                if s == '1':
                    self.dfa_transition_relation[self.qVars[sidx].bddPattern().__str__()] |= dfa_state_cube & edge_sym
        


class SymbolicPartitionedDFAFromMona(SymbolicPartitionedDFA):
    """
    This class inherits the SymbolicPartitionedDFA and implements the following:

     1. Calling the approriate Tool (for now Mona) for constructing the automaton
     2. Implementing the init and goal state initialization method LTLf formula
     3. Implementing the create_dfa_transition_relation() method for constructing the transition relation from SPOT DFA 
    """
    
    def __init__(self, formula: str, manager: Cudd, latches_map: bidict, game_latches: List[ADD], prime_game_latches: List[ADD]):
        super().__init__(formula=formula, manager=manager, latches_map=latches_map, game_latches=game_latches, prime_game_latches=prime_game_latches)


    def formula_to_automaton(self): 
        # Call MONA, Construct Graph using our PDDLtoSim toolbox, and return a graph that ONLY has the nodes (no edges)
        dfa = Ltlf2MonaDFA(formula=self.formula)
        num_of_states = dfa.num_of_states

        return dfa, num_of_states

    def set_init_goal_states(self):
        self.init: List[int] = self.dfa.init_state
        self.goal: List[int] = self.dfa.accp_states
    

    def set_init_latch(self):
        for q in self.init:
            self.init_latch |= self.qVar_map_sym[q]
    

    def set_goal_latch(self):
        for q in self.goal:
            self.goal_latch |= self.qVar_map_sym[q]

    def get_ltlf_edge_boolean_formula(self, labels: List, guard: str) -> ADD:
        """
        A function that parse the guard and constructs its correpsonding symbolic edge for symbolic LTLf DFA construction.

        The Atomic Proposition in the formula are of the form: pij where i is the box id and j is the location id.
        """
        expr = self.manager.addOne()
        for idx, value in enumerate(guard):
            if value == "1":
                if isinstance(labels, tuple):
                    cryptic_lbl = labels[idx]
                else:
                    cryptic_lbl = labels
                
                box_loc: str = re.search(r'\d+', str(cryptic_lbl)).group()
                expr &= self.predicate_add_sym_map_lbl[f'b{box_loc[0]} l{box_loc[1]}']
            
            elif value == "0":
                if isinstance(labels, tuple):
                    cryptic_lbl = labels[idx]
                else:
                    cryptic_lbl = labels
                
                box_loc: str = re.search(r'\d+', str(cryptic_lbl)).group()
                expr &= ~self.predicate_add_sym_map_lbl[f'b{box_loc[0]} l{box_loc[1]}']
            else:
                assert value == "X", "Error while constructing symbolic LTLF DFA edge. FIX THIS!!!"
        
        return expr
    

    def create_dfa_transition_relation(self):
        """
         This function parses the Mona DFA output and construct the symbolic TR associated with DFA.
        """
        self.dfa_transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.qVars}
        mona_output: str = self.dfa.mona_dfa

        for line in mona_output.splitlines():
            if line.startswith("State "):
                # extract the original state
                orig_state = self.dfa.get_value(line, r".*State[\s]*(\d+):\s.*", int)
                
                # extract string guard
                guard = self.dfa.get_value(line, r".*:[\s](.*?)[\s]->.*", str)
                
                # convert it into boolean formula
                if self.dfa.task_labels:
                    edge_sym: ADD = self.get_ltlf_edge_boolean_formula(self.dfa.task_labels, guard)
                else:
                    edge_sym: ADD = self.get_ltlf_edge_boolean_formula(self.dfa.task_labels, "X")
                
                dest_state = self.dfa.get_value(line, r".*state[\s]*(\d+)[\s]*.*", int)

                # ignore the superficial state 0
                if orig_state:
                    dfa_state_cube: ADD = self.qVar_map_sym[orig_state] 
                    dfa_state_prime_str: str = self.qVar_map[dest_state]

                    self.monolithic_valid_q_ps_pq |= dfa_state_cube & edge_sym.swapVariables(self.game_latches, self.prime_game_latches) & self.prime_qVar_map_sym[dest_state]

                    # now we add the transition dfa's transition relation
                    for sidx, s in enumerate(dfa_state_prime_str):
                        if s == '1':
                            self.dfa_transition_relation[self.qVars[sidx].bddPattern().__str__()] |= dfa_state_cube & edge_sym
