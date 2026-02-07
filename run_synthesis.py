import os
import sys
import time
import math
import yaml
import json

import argparse

import logging
import logging.config

from typing import Union
from cudd import Cudd, ADD

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


def parse_args():
    """
    Parse command-line arguments for the synthesis script.
    """
    parser = argparse.ArgumentParser(description="Compositional Synthesis Script")

    # Game setup parameters
    parser.add_argument('--boxes', type=int, required=True,
                        help='Number of boxes in the environment.')
    parser.add_argument('--locs', type=int, required=True,
                        help='Number of locations in the environment.')
    parser.add_argument('--budget', type=int,required=True,
                        help='Energy budget for the robot.')
    parser.add_argument('--ratio', type=int, required=True,
                        help='Ratio of human interventions to robot moves.')
    
    # Use nargs='+' to accept multiple values for these list-based arguments
    parser.add_argument('--human_locs', type=int, nargs='+', required=True,
                        help='Locations where the human can intervene.')
    parser.add_argument('--human_boxes', type=int, nargs='+', required=True,
                        help='Boxes that the human can move.')
    
    parser.add_argument('--init', type=str, nargs='+', required=True,
                        help='Initial state configuration.')
    parser.add_argument('--goal', type=str, nargs='+', required=True,
                        help='Goal state configuration.')
    
    parser.add_argument('--formula', type=str, required=True,
                        help='LTLf formula for the task specification.')

    # Boolean flags
    parser.add_argument('--cooperative_game', action='store_true',
                        help='Flag to run a cooperative game.')
    parser.add_argument('--enable_reordering', action='store_true',
                        help='Flag to enable BDD variable reordering.')
    parser.add_argument('--ltlf_flag', action='store_true', default=True,
                        help='Flag to indicate an LTLf synthesis task.')
    parser.add_argument('--only_reachable_states', action='store_true', default=False,
                        help='Flag to consider only reachable states during synthesis.')

    # Game type flags (mutually exclusive)
    game_type = parser.add_mutually_exclusive_group()
    game_type.add_argument('--game', action='store_true', dest='game_type_game',
                           help='Construct a standard game.')
    game_type.add_argument('--dfa_game', action='store_true', dest='game_type_dfa',
                           help='Construct a game with a DFA.')
    game_type.add_argument('--regret_game', action='store_true', dest='game_type_regret',
                           help='Construct a regret-based game.')

    # Algorithm choice
    parser.add_argument('--algorithm', type=str, default='ADD',
                        choices=['ADD', 'BDD', 'hybrid'],
                        help='Algorithm to use for strategy synthesis.')

    return parser.parse_args()


def main(args):
    """
    Main function to run the synthesis process based on parsed arguments.
    """
    game_instance = None
    
    # 1. Instantiate the correct game class based on arguments
    if args.regret_game:
        print("--- Initializing Regret DFA Game ---")
        game_class = SymbolicPartitionedRegretDFAGameNoPrime if args.no_prime else SymbolicPartitionedRegretDFAGame
        game_instance = game_class(
            boxes=args.boxes, locs=args.locs, ratio=args.ratio, init=args.init,
            goal=args.goal, formula=args.formula, restricted_human_locs=args.human_locs,
            restricted_human_boxes=args.human_boxes, ltlf_flag=args.ltlf_flag,
            budget=args.budget, enable_reordering=args.enable_reordering,
            only_reachable_states=args.only_reachable_states
        )
    elif args.dfa_game:
        print("--- Initializing DFA Game ---")
        game_class = SymbolicPartitionedDFAGameNoPrime if args.no_prime else SymbolicPartitionedDFAGame
        game_instance = game_class(
            boxes=args.boxes, locs=args.locs, ratio=args.ratio, init=args.init,
            goal=args.goal, formula=args.formula, restricted_human_locs=args.human_locs,
            restricted_human_boxes=args.human_boxes, ltlf_flag=args.ltlf_flag,
            enable_reordering=args.enable_reordering,
            only_reachable_states=args.only_reachable_states
        )
    elif args.game:
        print("--- Initializing Standard Game ---")
        game_class = FrankaWorldDynamicRatioTurnBasedNoPrime if args.no_prime else FrankaWorldDynamicRatioTurnBased
        game_instance = game_class(
            boxes=args.boxes, locs=args.locs, ratio=args.ratio, init=args.init,
            goal=args.goal, restricted_human_locs=args.human_locs,
            restricted_human_boxes=args.human_boxes, enable_reordering=args.enable_reordering,
            only_reachable_states=args.only_reachable_states
        )
    else:
        print("Error: No valid game type selected.", file=sys.stderr)
        sys.exit(1)

    # 2. Create the transition relation
    print("\n--- Creating Transition Relation ---")
    tic = time.time()
    game_instance.create_transition_relation()
    toc = time.time()
    print(f"Time to create transition relation: {toc - tic:.4f} seconds")

    # 3. Run the appropriate solver
    print(f"\n--- Synthesizing Strategy using '{args.algorithm}' solver ---")
    strategy = None
    tic = time.time()

    if args.regret_game:
        # Regret games have their own solver methods
        strategy, _ = game_instance.regret_solver(verbose=False)
    else:
        if args.algorithm == 'ADD':
            strategy, _ = game_instance.solve(verbose=False, cooperative_game=args.cooperative_game)
        elif args.algorithm == 'BDD':
            strategy, _ = game_instance.pure_bdd_solve(verbose=False, cooperative_game=args.cooperative_game)
        elif args.algorithm == 'hybrid':
            strategy, _ = game_instance.hybrid_solve(verbose=False, cooperative_game=args.cooperative_game)

    toc = time.time()
    print(f"Time to synthesize strategy: {toc - tic:.4f} seconds")

    # 4. Roll out the strategy if found
    if strategy is not None:
        print("\n--- Rolling out strategy ---")
        if hasattr(game_instance, 'gobr_roll_out_strategy'):
            game_instance.gobr_roll_out_strategy(strategy=strategy, verbose=True)
        else:
            game_instance.roll_out_strategy(strategy=strategy, verbose=True)
    else:
        print("\n--- No winning strategy found ---")


if __name__ == "__main__":
    # --- Setup Logging ---
    # Ensure the logs directory exists
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Load logging configuration from YAML
    with open('src/logging_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())
    logging.config.dictConfig(config)

    # Get the specific logger defined in the YAML file
    logger = logging.getLogger('synthesis_logger')
    # --- End Logging Setup ---

    args = parse_args()
    # The parse_args() function will automatically use sys.argv
    args = parse_args()

    # You can now access the parsed arguments like this:
    print("--- Running Synthesis with the following configuration ---")
    print(f"Boxes: {args.boxes}")
    print(f"Locations: {args.locs}")
    print(f"Budget: {args.budget}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Initial State: {args.init}")
    print(f"Cooperative Game: {args.cooperative_game}")
    
    # Example of accessing a game type flag
    if args.game_type_dfa:
        print("Game Type: DFA Game")
    elif args.game_type_regret:
        print("Game Type: Regret Game")
    else:
        print("Game Type: Standard Game")

    # Here, you would call your main synthesis logic, passing the `args` object
    # e.g., main_synthesis(args)
    main(args)
