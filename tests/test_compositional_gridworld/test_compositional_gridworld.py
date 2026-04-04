'''
 This file tests the Dynamic gridworld implementation. Specifically, it testst fo rthe following:
 1. Compositional Construciton of the Game Abstarction
 2. Synthesis of the strategy using the pure ADD, hybrid, and BDD based approaches
 3. Comparison of the strategies and optimal state values obtained from the different approaches
'''
import re

import os
import unittest

from cudd import Cudd

from .scenarios import *
from .expected_op import *

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors import GridWorldDynamicDoorsGame

SOLVERS = {'ADD': 'solve', 'BDD': 'pure_bdd_solve', 'hybrid': 'hybrid_solve'}


# TODO: add a more thorough test for the abstraction construction in the future.
class TestGridWorld(unittest.TestCase):
    def test_synthesis_no_doors_ADD(self):
        # load a gridworld instance
        for scenario_name, config in SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                config = SCENARIOS[scenario_name]
                gridworld = GridWorldDynamicGame(**config.__dict__)
        
                init_state_exp = gridworld.convert_cube_to_state_ADD(gridworld.init_latch, state_flag=True, action=False, table_header=False, verbose=False)

                self.assertEqual(len(init_state_exp), 1, "Expected only one initial state!")
                
                gridworld.create_transition_relation()

                strategy, opt_sval = gridworld.solve(verbose=False)
                if Expected_op_SCENARIOS[scenario_name].realizable:
                    self.assertIsNotNone(strategy, "Strategy synthesis failed!")
                    self.assertIsNotNone(opt_sval, "Optimal state values synthesis failed!")
                
                    if opt_sval.restrict(gridworld.init_latch) == gridworld.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((gridworld.init_latch & opt_sval).generate_cubes())[0][1]
                    init_val = int(init_val) if init_val != math.inf else math.inf
                
                    self.assertEqual(init_val, Expected_op_SCENARIOS[scenario_name].opt_init_sval, \
                            f"[{scenario_name}]: Optimal state value {init_val} for the initial state does not match expected value {Expected_op_SCENARIOS[scenario_name].opt_init_sval}!")
                    
                    gridworld.roll_out_strategy(strategy=strategy, verbose=False)

                self.assertEqual(gridworld.vi_layers, Expected_op_SCENARIOS[scenario_name].vi_layers, \
                        f"[{scenario_name}]: Number of VI layers {gridworld.vi_layers} does not match expected value {Expected_op_SCENARIOS[scenario_name].vi_layers}!")
    

    def test_synthesis_no_doors_BDD(self):
        # load a gridworld instance
        for scenario_name, config in SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                config = SCENARIOS[scenario_name]
                gridworld = GridWorldDynamicGame(**config.__dict__)
        
                init_state_exp = gridworld.convert_cube_to_state_ADD(gridworld.init_latch, state_flag=True, action=False, table_header=False, verbose=False)

                self.assertEqual(len(init_state_exp), 1, "Expected only one initial state!")
                
                gridworld.create_transition_relation()

                strategy, opt_sval = gridworld.pure_bdd_solve(verbose=False)
                if Expected_op_SCENARIOS[scenario_name].realizable:
                    self.assertIsNotNone(strategy, "Strategy synthesis failed!")
                    self.assertIsNotNone(opt_sval, "Optimal state values synthesis failed!")
                
                    if opt_sval.restrict(gridworld.init_latch) == gridworld.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((gridworld.init_latch & opt_sval).generate_cubes())[0][1]
                    init_val = int(init_val) if init_val != math.inf else math.inf
                
                    self.assertEqual(int(init_val), Expected_op_SCENARIOS[scenario_name].opt_init_sval, \
                            f"[{scenario_name}]: Optimal state value {init_val} for the initial state does not match expected value {Expected_op_SCENARIOS[scenario_name].opt_init_sval}!")
                
                    gridworld.roll_out_strategy(strategy=strategy, verbose=False)
                
                self.assertEqual(gridworld.vi_layers, Expected_op_SCENARIOS[scenario_name].vi_layers, \
                        f"[{scenario_name}]: Number of VI layers {gridworld.vi_layers} does not match expected value {Expected_op_SCENARIOS[scenario_name].vi_layers}!")

    

    def test_synthesis_no_doors_hybrid_solve(self):
        # load a gridworld instance
        for scenario_name, config in SCENARIOS.items():
            with self.subTest(scenario=scenario_name):
                config = SCENARIOS[scenario_name]
                gridworld = GridWorldDynamicGame(**config.__dict__)
        
                init_state_exp = gridworld.convert_cube_to_state_ADD(gridworld.init_latch, state_flag=True, action=False, table_header=False, verbose=False)

                self.assertEqual(len(init_state_exp), 1, "Expected only one initial state!")
                
                gridworld.create_transition_relation()

                strategy, opt_sval = gridworld.hybrid_solve(verbose=False)
                if Expected_op_SCENARIOS[scenario_name].realizable:
                    self.assertIsNotNone(strategy, "Strategy synthesis failed!")
                    self.assertIsNotNone(opt_sval, "Optimal state values synthesis failed!")
                
                    if opt_sval.restrict(gridworld.init_latch) == gridworld.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((gridworld.init_latch & opt_sval).generate_cubes())[0][1]
                    init_val = int(init_val) if init_val != math.inf else math.inf
                
                    self.assertEqual(init_val, Expected_op_SCENARIOS[scenario_name].opt_init_sval, \
                            f"[{scenario_name}]: Optimal state value {init_val} for the initial state does not match expected value {Expected_op_SCENARIOS[scenario_name].opt_init_sval}!")
                
                    gridworld.roll_out_strategy(strategy=strategy, verbose=False)

                self.assertEqual(gridworld.vi_layers, Expected_op_SCENARIOS[scenario_name].vi_layers, \
                        f"[{scenario_name}]: Number of VI layers {gridworld.vi_layers} does not match expected value {Expected_op_SCENARIOS[scenario_name].vi_layers}!")
                    
    

    def test_synthesis_doors_ADD(self):
        # load a gridworld instance
        for scenario_name, config in SCENARIOS_DOOR.items():
            with self.subTest(scenario=scenario_name):
                config = SCENARIOS_DOOR[scenario_name]
                gridworld = GridWorldDynamicDoorsGame(**config.__dict__)
        
                init_state_exp = gridworld.convert_cube_to_state_ADD(gridworld.init_latch, state_flag=True, action=False, table_header=False, verbose=False)

                self.assertEqual(len(init_state_exp), 1, "Expected only one initial state!")
                
                gridworld.create_transition_relation()

                strategy, opt_sval = gridworld.solve(verbose=False)
                # if 'unrealizable' not in scenario_name:
                if Expected_op_SCENARIOS_DOOR[scenario_name].realizable:
                    self.assertIsNotNone(strategy, "Strategy synthesis failed!")
                    self.assertIsNotNone(opt_sval, "Optimal state values synthesis failed!")

                    if opt_sval.restrict(gridworld.init_latch) == gridworld.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((gridworld.init_latch & opt_sval).generate_cubes())[0][1]
                    init_val = int(init_val) if init_val != math.inf else math.inf
                
                    self.assertEqual(init_val, Expected_op_SCENARIOS_DOOR[scenario_name].opt_init_sval, \
                            f"[{scenario_name}]: Optimal state value {init_val} for the initial state does not match expected value {Expected_op_SCENARIOS_DOOR[scenario_name].opt_init_sval}!")

                    gridworld.roll_out_strategy(strategy=strategy, verbose=False)

                self.assertEqual(gridworld.vi_layers, Expected_op_SCENARIOS_DOOR[scenario_name].vi_layers, \
                        f"[{scenario_name}]: Number of VI layers {gridworld.vi_layers} does not match expected value {Expected_op_SCENARIOS_DOOR[scenario_name].vi_layers}!")                    
    

    def test_synthesis_doors_BDD(self):
        # load a gridworld instance
        for scenario_name, config in SCENARIOS_DOOR.items():
            with self.subTest(scenario=scenario_name):
                config = SCENARIOS_DOOR[scenario_name]
                gridworld = GridWorldDynamicDoorsGame(**config.__dict__)
        
                init_state_exp = gridworld.convert_cube_to_state_ADD(gridworld.init_latch, state_flag=True, action=False, table_header=False, verbose=False)

                self.assertEqual(len(init_state_exp), 1, "Expected only one initial state!")
                
                gridworld.create_transition_relation()

                strategy, opt_sval = gridworld.pure_bdd_solve(verbose=False)
                if Expected_op_SCENARIOS_DOOR[scenario_name].realizable:
                    self.assertIsNotNone(strategy, "Strategy synthesis failed!")
                    self.assertIsNotNone(opt_sval, "Optimal state values synthesis failed!")
                
                    if opt_sval.restrict(gridworld.init_latch) == gridworld.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((gridworld.init_latch & opt_sval).generate_cubes())[0][1]
                    init_val = int(init_val) if init_val != math.inf else math.inf
                
                    self.assertEqual(int(init_val), Expected_op_SCENARIOS_DOOR[scenario_name].opt_init_sval, \
                            f"[{scenario_name}]: Optimal state value {init_val} for the initial state does not match expected value {Expected_op_SCENARIOS_DOOR[scenario_name].opt_init_sval}!")

                    gridworld.roll_out_strategy(strategy=strategy, verbose=False)
                
                self.assertEqual(gridworld.vi_layers, Expected_op_SCENARIOS_DOOR[scenario_name].vi_layers, \
                        f"[{scenario_name}]: Number of VI layers {gridworld.vi_layers} does not match expected value {Expected_op_SCENARIOS_DOOR[scenario_name].vi_layers}!")
                    
    

    def test_synthesis_doors_hybrid_solve(self):
        # load a gridworld instance
        for scenario_name, config in SCENARIOS_DOOR.items():
            with self.subTest(scenario=scenario_name):
                config = SCENARIOS_DOOR[scenario_name]
                gridworld = GridWorldDynamicDoorsGame(**config.__dict__)
        
                init_state_exp = gridworld.convert_cube_to_state_ADD(gridworld.init_latch, state_flag=True, action=False, table_header=False, verbose=False)

                self.assertEqual(len(init_state_exp), 1, "Expected only one initial state!")
                
                gridworld.create_transition_relation()

                strategy, opt_sval = gridworld.hybrid_solve(verbose=False)
                if Expected_op_SCENARIOS_DOOR[scenario_name].realizable:
                    self.assertIsNotNone(strategy, "Strategy synthesis failed!")
                    self.assertIsNotNone(opt_sval, "Optimal state values synthesis failed!")

                    if opt_sval.restrict(gridworld.init_latch) == gridworld.manager.addZero():
                        init_val: int = 0
                    else:
                        init_val: int = list((gridworld.init_latch & opt_sval).generate_cubes())[0][1]
                    init_val = int(init_val) if init_val != math.inf else math.inf
                
                    self.assertEqual(init_val, Expected_op_SCENARIOS_DOOR[scenario_name].opt_init_sval, \
                            f"[{scenario_name}]: Optimal state value {init_val} for the initial state does not match expected value {Expected_op_SCENARIOS_DOOR[scenario_name].opt_init_sval}!")

                    gridworld.roll_out_strategy(strategy=strategy, verbose=False)

                self.assertEqual(gridworld.vi_layers, Expected_op_SCENARIOS_DOOR[scenario_name].vi_layers, \
                        f"[{scenario_name}]: Number of VI layers {gridworld.vi_layers} does not match expected value {Expected_op_SCENARIOS_DOOR[scenario_name].vi_layers}!")


if __name__ == "__main__":
    unittest.main()