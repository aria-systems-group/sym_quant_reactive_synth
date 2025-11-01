"""
 This script contains the SymbolicPartitionedDFAGame class, which extends the FrankaWorldDyanmicRatioTurnBased class to play a game 
 where the objective is specified by a LTL/LTLf formula. 

 SymbolicPartitionedDFAGame uses a symbolic partitioned DFA to represent the automaton corresponding to the LTL/LTLf formula.
 It constructs the transition relation in a partitioned manner (vector of boolean variables), leveraging the symbolic representation for efficiency.
"""
import sys
import math

from functools import reduce
from typing import List, Union

from bidict import bidict

from cudd import Cudd, ADD, BDD

from src.compositional_graphs.symbolic_partitioned_dfa import SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot
from src.compositional_graphs.test_frankadynamic_ratio_tb import FrankaWorldDynamicRatioTurnBased
from src.compositional_graphs.test_frankadynamic_ratio_else_tb import FrankaWorldDynamicRatioTurnBasedElse


class SymbolicPartitionedDFAGame(FrankaWorldDynamicRatioTurnBasedElse):
    """
    This class extends the FrankaDynamicRatioTurnBased class to create a game environment where the objective is specified by a LTL/LTLf formula.
    """

    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
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
        self.prime_qVars: List[ADD] = []
        self.qVar_map: List[ADD] = {} 
        self.qVar_map_sym: List[ADD] = {} 
        self.ltlf_flag: bool = ltlf_flag
        self.dfa_handle: Union[SymbolicPartitionedDFAFromMona, SymbolicPartitionedDFAFromSpot] = None
        self.dfa_latches: List[ADD] = []
        self.dfa_latches_sym_map = bidict({})
        # Game setup, DFA setup all are done in create_all_boolean_state_vars_and_maps() that is called in the super class init
        super().__init__(boxes, locs, ratio, init, goal, restricted_human_locs, enable_reordering=False)

        # set up dfa init and goal states
        self.dfa_handle.set_init_latch()
        self.dfa_handle.set_goal_latch()

        # by default variable reordering is disabled for DFA games - to check for computation time without this optimization
        # however, switching variable ordering make the code faster for sure.
        if enable_reordering:
            self.manager.autodynEnable()
    

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
        self.prime_qVars: List[ADD] = self.dfa_handle.prime_qVars
    

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
        self.dfa_handle.create_dfa_transition_relation()
        self.qVars = dfa_handle.qVars
        self.dfa_latches: List[ADD] = dfa_handle.qVars
        self.qVar_map = dfa_handle.qVar_map
        self.qVar_map_sym = dfa_handle.qVar_map_sym
    

    def convert_cube_to_state_ADD(self, dd: ADD, state_flag: bool = True, dfa_flag: bool = True, robot_action: bool = False, human_action: bool = False) -> None:
        """
         Convert a cube to a state representation. Set the respective flags to True to print respective information. 
         By default DFA and Game state flags are set to True.
         If you want to print the robot action as well, set robot_action to True. 
         If you want to print the human action as well, set human_action to True.
        """
        relevant_vars = [] + self.tVar + self.kVars
        if state_flag:
            relevant_vars.extend(self.latches) # includes pVars and bVars
        if dfa_flag:
            relevant_vars.extend(self.dfa_latches) # includes qVars
        if robot_action:
            relevant_vars.extend(self.oVars) # robot action vars (oVars)
        if human_action:
            relevant_vars.extend(self.iVars) # env action vars (iVars)

        cubes = self.get_all_cubes(dd, relevant_vars=relevant_vars)
        
        # the next vars are l' vars - we ignore them for now. The next ones are robot action and finally human action vars
        start_ovar_idx, end_ovar_idx = self.manager.addVariables().index(self.oVars[0]), self.manager.addVariables().index(self.oVars[-1])
        start_ivar_idx, end_ivar_idx = self.manager.addVariables().index(self.iVars[0]), self.manager.addVariables().index(self.iVars[-1])

        # create turn abstraction cube
        tConf_exist_cube = reduce(lambda a, b: a & b, self.xVars + self.qVars + self.oVars + self.iVars)
        kConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.xVars[len(self.kVars):] + self.qVars + self.oVars + self.iVars)
        qConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.xVars + self.oVars + self.iVars)
        # create existential abstraction cubes
        rConf_exist_cube = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.qVars + self.xVars[len(self.kVars)+len(self.pVars):] + self.oVars + self.iVars) 
        # because ADD is not iterable and cannot be added to a list directly
        bConf_exist_cube = dict({})
        for bidx in range(self.boxes):
            if self.boxes == 1:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.oVars + self.iVars)
            else:
                bConf_exist_cube[bidx] = reduce(lambda a, b: a & b, self.tVar + self.kVars + self.pVars + self.qVars + self.oVars + self.iVars) & reduce(lambda x, y: x & y, self.bVars_cubes[:bidx] + self.bVars_cubes[bidx+1:])
        
        # print the states
        states_action_pairs = [] 
        for cube, val in cubes:
            tConf_exist_str = cube.existAbstract(tConf_exist_cube).bddPattern().cubeString().replace('-', '')
            rConf_cube_str = cube.existAbstract(rConf_exist_cube).bddPattern().cubeString().replace('-', '')
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
                print(f"[(({self.tVar_map.inv[tConf_exist_str]}, {self.kVar_map.inv[kConf_exist_str]}, {self.pVar_map.inv[rConf_cube_str]}, {box_states}), {self.dfa_handle.qVar_map.inv[qConf_exist_str]}), {val}]")
                states_action_pairs.append([
                    (((self.tVar_map.inv[tConf_exist_str],
                       self.kVar_map.inv[kConf_exist_str],
                       self.pVar_map.inv[rConf_cube_str], box_states),
                       self.dfa_handle.qVar_map.inv[qConf_exist_str]), val), None])
            except KeyError:
                continue
            
            # print the robot and human actions as well
            if robot_action:
                oCube_str = cube.bddPattern().cubeString()[start_ovar_idx:end_ovar_idx + 1].replace('-', '')
                try:
                    rAction_str = self.rAction_map_sym.inv[self.cube_to_add(oCube_str, self.oVars)]
                except KeyError:
                    continue
            if human_action:
                iCube_str = cube.bddPattern().cubeString()[start_ivar_idx:end_ivar_idx + 1].replace('-', '')
                try:
                    eAction_str = self.eAction_map_sym.inv[self.cube_to_add(iCube_str, self.iVars)]
                except KeyError:
                    continue
            if robot_action or human_action:    
                action = ", ".join(filter(None, [rAction_str if robot_action else None, eAction_str if human_action else None]))
                print(f"    -- Actions: ({action})")
        
        return states_action_pairs


    def roll_out_strategy(self, strategy: ADD, verbose: bool = False):
        """
         A function to rollout a give strategy
        """
        curr_state_sym = self.init_latch & self.dfa_handle.init_latch
        oVars_bdd: List[BDD] = [var.bddPattern() for var in self.oVars]
        iVars_bdd: List[BDD] = [var.bddPattern() for var in self.iVars]

        while (curr_state_sym & self.dfa_handle.goal_latch).isZero():
            if verbose:
                print("Current State:")
                curr_state_exp: List[str] = self.convert_cube_to_state_ADD(curr_state_sym, state_flag=True, robot_action=False)
                assert len(curr_state_exp) == 1, "Make sure the current state is a singleton set. ..."
                "For rollout, it should be a single intial state."
            
            # first get the optimum state value
            try:
                opt_sval = list((curr_state_sym & self.comp_winning_states).generate_cubes())[0][1]
            except IndexError:
                opt_sval = 0
            
            turn = 'robot' if curr_state_exp[0][0][0][0][0] == 'robot' else'human'

            # get the action to be taken at the current state
            if turn == 'robot':
                act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(oVars_bdd)
            else:
                act_cube: BDD = (strategy.restrict(curr_state_sym)).bddInterval(opt_sval, opt_sval).pickOneMinterm(iVars_bdd)
            
            act_cube_string = act_cube.cubeString().replace('-', '')

            try:
                act_name = self.rAction_map.inv[act_cube_string] if turn == 'robot' else self.eAction_map.inv[act_cube_string]
            except KeyError:
                print("No robot action found!!")
                return

            curr_game_state = list(curr_state_exp[0][0][0][0])
            curr_dfa_state: int = curr_state_exp[0][0][0][1]
           
            # get the next state in the game
            if turn == 'robot':
                curr_game_state_sym: ADD = self.get_next_state_robot(curr_game_state, act_name)
            else:
                curr_game_state_sym, act_name = self.get_next_state_human(curr_game_state, act_name)
            
            # check if you evolved over the DFA 
            # create DFA edge and check if it satisfies any of the dges or not
            for dfa_state_sym in self.qVar_map_sym.values():
                dfa_state_sym = dfa_state_sym.swapVariables(self.qVars, self.prime_qVars)
                dfa_pre: ADD = dfa_state_sym.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))
                edge_exists: bool = not (dfa_pre & (self.qVar_map_sym[curr_dfa_state] & curr_game_state_sym)).isZero()

                if edge_exists:
                    curr_dfa_state: ADD = dfa_state_sym.swapVariables(self.prime_qVars, self.qVars)
                    break
            
            curr_state_sym: ADD = curr_game_state_sym & curr_dfa_state
            
            # printing the action here as the human action is overriden above. This because invalid human moves
            # are converted to hmove noop. So, it is more accurate to print the action after getting the next state.
            if verbose:
                print(f"Robot Action: {act_name}") if turn == 'robot' else print(f"Human Action: {act_name}")


    def solve(self, verbose: bool = False, cooperative_game: bool = False) -> Union[ADD, None]:
        """
        A method that implements the value iteration algorithm For DFA Game. This method compute the optimal cost winning strategy
          for the Sys player (robot) to reach the goal state.
        """
        
        # initialize a weight ADD that assigns cost to each robot state
        self.weight = self.manager.addZero()
        for rConf in self.pVar_map.keys():
            if rConf != f'ready l{self.locs + 1}' and rConf != f'holding l{self.locs + 1}':
                self.weight |= self.tVar_map_sym['robot'] & self.xVar_map_sym[rConf]
        
        # initialize goal state with 0 state value and add it to the winning region
        goal = self.dfa_handle.goal_latch.ite(self.manager.addZero(), self.manager.plusInfinity())
        curr_winning_states =  self.manager.plusInfinity()
        curr_winning_states = curr_winning_states.min(goal)

        # print the initial winning states
        if verbose:
            print("Initial Winning States:")
            # by default generate cubes does not return cubes that point to 0 leaf. 
            # So, we manually convert the 0 leaf to a cube with leaf value 1 here for printing.
            # setting state flag to False as ever Game state with an accepting DFA state is a winning state. 
            # So if state flag was true, it would have printed the entire game 
            self.convert_cube_to_state_ADD(curr_winning_states.bddInterval(0, 0).toADD(), state_flag=False, dfa_flag=True, robot_action=False)
        
        # intialize the iteration counter
        layer = 0

        while True:
            print(f"**************************Layer: {layer}**************************")

            # prime the vars
            curr_winning_states_primed = curr_winning_states.swapVariables(self.latches + self.qVars, self.prime_latches + self.prime_qVars)
            
            # first evolve over the DFA
            dfa_preimage: ADD = curr_winning_states_primed.vectorCompose(self.prime_qVars, list(self.dfa_handle.dfa_transition_relation.values()))

            # then evolve over the game
            preimage = dfa_preimage.vectorCompose(self.prime_latches, list(self.transition_relation.values()))

            # add the action costs associated with the robot actions   
            preimage = preimage + self.weight
            # print("Current Preimage:")
            # self.convert_cube_to_state_ADD(preimage, state_flag=True, robot_action=False, human_action=False)
            # go over all the env actions and preserve the maximum one
            MaxUpre = []
            for env_tr_dd in self.env_action_cube_list:
                # MaxUpre.append(preimage.restrict(env_tr_dd))
                MaxUpre.append(preimage.cofactor(env_tr_dd))
            
            if cooperative_game:
                Upre = reduce(lambda x, y: x.min(y), MaxUpre)
            else:
                Upre = reduce(lambda x, y: x.max(y), MaxUpre)

            # go over all the sys actions and preserve the minimum one
            Minpre = []
            for robot_tr_dd in self.robot_action_cube_list:
                # Minpre.append(Upre.restrict(robot_tr_dd))
                Minpre.append(Upre.cofactor(robot_tr_dd))
            
            next_winning_states = reduce(lambda x, y: x.min(y), Minpre)
            next_winning_states = next_winning_states.min(goal)

            # adding debugging step
            if verbose:
                print("Current Winning States:")
                self.convert_cube_to_state_ADD(next_winning_states, robot_action=False)
            
            if curr_winning_states.compare(next_winning_states, 2):
                print("**************************Reached fixpoint**************************")
                if (self.dfa_handle.init_latch & self.init_latch) & curr_winning_states != self.manager.plusInfinity():
                    if self.init_latch & curr_winning_states == self.manager.addZero():
                        print("Either The Initial State is a Goal State or the human can complete the task for the robot without expending energy!!")
                        init_val: int = 0
                    else:
                        init_val: int = list((self.dfa_handle.init_latch & self.init_latch & curr_winning_states).generate_cubes())[0][1]
                    print(f"A Winning Strategy Exists!!. The State value is {init_val}")
                    self.comp_winning_states = curr_winning_states
                    return preimage if init_val < math.inf else None
                return None

            # update the counter
            layer += 1

            # swap the winning states
            curr_winning_states = next_winning_states
        
    

    def preimage_test(self, From: ADD, latches: List[ADD], prime_latches: List[ADD], ts_action: List[ADD]) -> ADD:
        From = From.swapVariables(latches, prime_latches)
        return From.vectorCompose(prime_latches, ts_action)


    def test_pre_image(self):
        goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['ready l1'] & self.xVar_map_sym['b0 l1'] & self.dfa_handle.goal_latch
        # goal state is b0 and l0 and ready l0
        print('Goal state:', goal_cube)

        # first evolve over the DFA
        dfa_preimage = self.preimage_test(From=goal_cube, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation.values()))
        print('DFA Preimage: ', dfa_preimage)
        self.convert_cube_to_state_ADD(dfa_preimage, human_action=False, robot_action=False)
        
        # then evolve over the game
        dfa_game_preimage = self.preimage_test(From=dfa_preimage, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)
        self.convert_cube_to_state_ADD(dfa_game_preimage, human_action=False, robot_action=False)