import os
import sys
import time
import yaml

# import argparse
from config_compositional import *

# Imports with prime latches
from src.compositional_graphs.symbolic_partitioned_dfa_game import SymbolicPartitionedDFAGame
from src.compositional_graphs.test_frankadynamic_ratio_else_tb import FrankaWorldDynamicRatioTurnBasedElse
from src.compositional_graphs.test_frankadynamic_ratio_tb import FrankaWorldDynamicRatioTurnBased

# Imports with no prime latches
from src.compositional_graphs.frankadynamic_ratio_tb_noprime import FrankaWorldDynamicRatioTurnBasedNoPrime
from src.compositional_graphs.frankadynamic_ratio_else_tb_noprime import FrankaWorldDynamicRatioTurnBasedElseNoPrime
from src.compositional_graphs.symbolic_partitioned_dfa_game_noprime import SymbolicPartitionedDFAGameNoPrime
from src.compositional_graphs.symbolic_partitioned_regret_dfa_game_noprime import SymbolicPartitionedRegretDFAGameNoPrime

# regret script imports
from src.compositional_graphs.symbolic_partitioned_regret_dfa_game import SymbolicPartitionedRegretDFAGame


if __name__ == "__main__":
    """
    Main function to run the synthesis process based on parsed arguments.
    """
    # --- Setup Logging ---
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    

    for run in range(ITERATIONS):
        # log setup
        setup_dict = {}
        
        setup_dict['boxes'] = BOXES
        setup_dict['locs'] = LOCS
        setup_dict['ratio'] = RATIO
        setup_dict['init'] = INIT
        setup_dict['goal'] = GOAL
        setup_dict['formula'] = FORMULA
        setup_dict['restricted_human_locs'] = list(HUMAN_LOCS)
        setup_dict['restricted_human_boxes'] = HUMAN_BOXES
        setup_dict['ltlf_flag'] = LTLF_FLAG
        setup_dict['budget'] = BUDGET
        setup_dict['enable_reordering'] = ENABLE_REORDERING
        setup_dict['only_reachable_states'] = ONLY_REACHABLE_STATES
        setup_dict['cooperative_game'] = COOPERATIVE_GAME
        setup_dict['game_type'] = 'regret_game' if REGRET_GAME else 'dfa_game' if DFA_GAME else 'game'
        setup_dict['algorithm'] = ALGORITHM

        print(f"\n=== Starting Run {run + 1}/{ITERATIONS} ===")
        game = None
        
        # 1. Instantiate the correct game class based on arguments
        if REGRET_GAME:
            print("--- Initializing Regret DFA Game ---")
            if NO_PRIME:
                print("--- Initializing No Prime Regret DFA Game ---")
                game = SymbolicPartitionedRegretDFAGameNoPrime(
                    boxes=BOXES, locs=LOCS, ratio=RATIO, init=INIT,
                    goal=GOAL, formula=FORMULA, restricted_human_locs=HUMAN_LOCS,
                    restricted_human_boxes=HUMAN_BOXES, ltlf_flag=LTLF_FLAG,
                    budget=BUDGET, enable_reordering=ENABLE_REORDERING)
            else:
                print("--- Initializing Regret DFA Game with Primes ---")
                game = SymbolicPartitionedRegretDFAGame(
                    boxes=BOXES, locs=LOCS, ratio=RATIO, init=INIT,
                    goal=GOAL, formula=FORMULA, restricted_human_locs=HUMAN_LOCS,
                    restricted_human_boxes=HUMAN_BOXES, ltlf_flag=LTLF_FLAG,
                    budget=BUDGET, enable_reordering=ENABLE_REORDERING,
                    only_reachable_states=ONLY_REACHABLE_STATES)
            setup_dict['goal'] = ''
        elif DFA_GAME:
            print("--- Initializing DFA Game ---")
            if NO_PRIME:
                print("--- Initializing No Prime DFA Game ---")
                game = SymbolicPartitionedDFAGameNoPrime(
                    boxes=BOXES, locs=LOCS, ratio=RATIO, init=INIT,
                    goal=GOAL, formula=FORMULA, restricted_human_locs=HUMAN_LOCS,
                    restricted_human_boxes=HUMAN_BOXES, ltlf_flag=LTLF_FLAG,
                    enable_reordering=ENABLE_REORDERING)
            else:
                print("--- Initializing DFA Game with Primes ---")
                game =SymbolicPartitionedDFAGame(
                    boxes=BOXES, locs=LOCS, ratio=RATIO, init=INIT,
                    goal=GOAL, formula=FORMULA, restricted_human_locs=HUMAN_LOCS,
                    restricted_human_boxes=HUMAN_BOXES, ltlf_flag=LTLF_FLAG,
                    enable_reordering=ENABLE_REORDERING,
                    only_reachable_states=ONLY_REACHABLE_STATES)
            setup_dict['goal'] = ''
            setup_dict['budget'] = ''
        elif GAME:
            print("--- Initializing Standard Game ---")
            if NO_PRIME:
                print("--- Initializing No Prime Standard Game ---")
                game = FrankaWorldDynamicRatioTurnBasedElseNoPrime(
                    boxes=BOXES, locs=LOCS, ratio=RATIO, init=INIT,
                    goal=GOAL, restricted_human_locs=HUMAN_LOCS,
                    restricted_human_boxes=HUMAN_BOXES, enable_reordering=ENABLE_REORDERING)
            else:
                print("--- Initializing Standard Game with Primes ---")
                game = FrankaWorldDynamicRatioTurnBasedElse(
                    boxes=BOXES, locs=LOCS, ratio=RATIO, init=INIT,
                    goal=GOAL, restricted_human_locs=HUMAN_LOCS,
                    restricted_human_boxes=HUMAN_BOXES, enable_reordering=ENABLE_REORDERING,
                    only_reachable_states=ONLY_REACHABLE_STATES)
            setup_dict['formula'] = ''
            setup_dict['budget'] = ''
        else:
            print("Error: No valid game type selected.", file=sys.stderr)
            sys.exit(1)

        # 2. Create the transition relation
        print("\n--- Creating Transition Relation ---")
        tic = time.time()
        game.create_transition_relation()
        toc = time.time()
        print(f"Time to create transition relation: {toc - tic:.4f} seconds")
        comp_time = {'TR_time': toc - tic}

        # 3. Run the appropriate solver
        print(f"\n--- Synthesizing Strategy using '{ALGORITHM}' solver ---")
        strategy = None
        tic = time.time()

        if REGRET_GAME:
            # Regret games have their own solver methods
            if ALGORITHM == 'ADD':
                strategy, _ = game.regret_solver(verbose=False)
            elif ALGORITHM == 'BDD':
                strategy, _ = game.pure_bdd_regret_solver(verbose=False)
            elif ALGORITHM == 'hybrid':
                strategy,_ = game.hybrid_regret_solver(verbose=False)
            else:
                print("Error: No valid ALGORITHM type selected.", file=sys.stderr)
                sys.exit(1)
            
        else:
            if ALGORITHM == 'ADD':
                strategy, _ = game.solve(verbose=False, cooperative_game=COOPERATIVE_GAME)
            elif ALGORITHM == 'BDD':
                strategy, _ = game.pure_bdd_solve(verbose=False, cooperative_game=COOPERATIVE_GAME)
            elif ALGORITHM == 'hybrid':
                strategy, _ = game.hybrid_solve(verbose=False, cooperative_game=COOPERATIVE_GAME)
            else:
                print("Error: No valid ALGORITHM type selected.", file=sys.stderr)
                sys.exit(1)

        toc = time.time()
        print(f"Time to synthesize strategy: {toc - tic:.4f} seconds")
        comp_time['Synth_time'] = toc - tic

        # 4. Roll out the strategy if found
        if strategy is not None:
            print("\n--- Rolling out strategy ---")
            if hasattr(game, 'gobr_roll_out_strategy'):
                game.gobr_roll_out_strategy(strategy=strategy, verbose=True)
            else:
                game.roll_out_strategy(strategy=strategy, verbose=True)
        else:
            print("\n--- No winning strategy found ---")
        game.logger.status = 'REALIZABLE' if strategy is not None else 'UNREALIZABLE'
        
        # game.manager.printInfo()
        # Capture CUDD manager info
        # import io
        # import contextlib
        # info_buffer = io.StringIO()
        # with contextlib.redirect_stdout(info_buffer):
        #    game.manager.printInfo()
        # manager_info = info_buffer.getvalue()

        # 5. Log results
        comp_time.update(game.logger.comp_time)
        game.logger.log(setup_dict=setup_dict, comp_time=comp_time, abs_dict=game.log_game_details())
        # Add manager info to the log
        game.logger.run_data['MemoryInUse'] = game.manager.readMemoryInUse()
        game.logger.dump_results_to_yaml(file_path=os.path.join('logs', 'synthesis_results_2'), iteration=run, add_time_stamp=False)
      
        del game.manager