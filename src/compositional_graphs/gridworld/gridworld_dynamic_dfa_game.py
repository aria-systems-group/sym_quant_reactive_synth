from bidict import bidict
from typing import List, Union

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMonaNoPrime, SymbolicPartitionedDFAFromSpotNoPrime


class GridWorldDynamicDFAGame(GridWorldDynamicGame):
    
    def __init__(self,
                 rows: int, columns: int,
                 init: List[tuple], goal: tuple,
                 formula: str, ltlf_flag: bool = True,
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
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMonaNoPrime, SymbolicPartitionedDFAFromSpotNoPrime] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(rows=rows, columns=columns, init=init, goal=goal, enable_reordering=False)

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


    def create_all_boolean_state_vars_and_maps(self):
        super().create_all_boolean_state_vars_and_maps()

        # create the dfa state variables and maps
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
                                                                latches_map={'x': self.xVar_map_sym, 'y': self.yVar_map_sym},
                                                                game_latches=None)
        else:
            dfa_handle = SymbolicPartitionedDFAFromSpotNoPrime(formula=self.formula,
                                                                manager=self.manager,
                                                                latches_map={'x': self.xVar_map_sym, 'y': self.yVar_map_sym},
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
        super().create_transition_relation()