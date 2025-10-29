"""
 This script contains the SymbolicPartitionedDFAGame class, which extends the FrankaWorldDyanmicRatioTurnBased class to play a game 
 where the objective is specified by a LTL/LTLf formula. 

 SymbolicPartitionedDFAGame uses a symbolic partitioned DFA to represent the automaton corresponding to the LTL/LTLf formula.
 It constructs the transition relation in a partitioned manner (vector of boolean variables), leveraging the symbolic representation for efficiency.
"""

from typing import List, Union
from bidict import bidict

from cudd import Cudd, ADD

from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot
from src.compositional_graphs.test_frankadynamic_ratio_tb import FrankaWorldDyanmicRatioTurnBased


class SymbolicPartitionedDFAGame(FrankaWorldDyanmicRatioTurnBased):
    """
    This class extends the FrankaDynamicRatioTurnBased class to create a game environment where the objective is specified by a LTL/LTLf formula.
    """

    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 ltlf_flag: bool = True):
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
        self.ltlf_flag: bool = ltlf_flag
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(boxes, locs, ratio, init, goal, restricted_human_locs)

        # set up dfa init and goal states
        self.dfa_handle.set_init_latch()
        self.dfa_handle.set_goal_latch()
        
    

    # need to override the create lacthes method to include dfa latches
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
    

    def create_all_prime_boolean_state_vars(self):
        """
         The main method that creates all primed version of the boolean variables for the FrankaDynamic Turn-Based Game.
          1. prime turn variables - tVars
          2. prime ratio variables - kVars
          3. prime predicate variables - pVars
          4. prime box predicate variables - bVars
        """
        offset = self.manager.size()
        self.prime_tVar: List[ADD] = [self.manager.addVar(offset, "pt0")]
        self.prime_kVars: List[ADD] = self.create_prime_ratio_vars()
        self.prime_pVars, self.prime_bVars = self.create_prime_latches()
        
        # create prime DFA latches next
        self.dfa_handle.create_prime_latches()
    

    def create_dfa_latches_and_maps(self):
        """
         Create DFA latches and their symbolic maps based on the provided LTL/LTLf formula.
        """
        # here we only create the variables and maps for the dfa
        if self.ltlf_flag:
            dfa_handle = SymbolicPartitionedDFAFromMona(formula=self.formula,
                                                        manager=self.manager,
                                                        latches_map=self.xVar_map_sym)
        else:
            dfa_handle = SymbolicPartitionedDFAFromSpot(formula=self.formula,
                                                        manager=self.manager,
                                                        latches_map=self.xVar_map_sym)
        
        self.dfa_handle = dfa_handle
        self.dfa_handle.create_latches_and_map()

        # now we create the TR
        self.dfa_handle.create_dfa_transition_relation(verbose=False, plot=False)

        self.dfa_latches: List[ADD] = dfa_handle.latches
        self.dfa_latches_sym_map = dfa_handle.latches_sym_map