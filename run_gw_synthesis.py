import os
import sys
import time
import argparse
import yaml

# import argparse
from config_gw_compositional import *

from src.compositional_graphs.gridworld.gridworld_dynamic import GridWorldDynamicGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors import GridWorldDynamicDoorsGame
from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game import GridWorldDynamicDFAGame
from src.compositional_graphs.gridworld.gridworld_dynamic_doors_dfa_game import GridWorldDynamicDoorsDFAGame

# regret imports
from src.compositional_graphs.gridworld.gridworld_dynamic_regret import GridWorldDynamicRegretGame
from src.compositional_graphs.gridworld.gridworld_dynamic_regret_doors import GridWorldDynamicDoorsRegretGame

# No Prime imports
from src.compositional_graphs.gridworld.gridworld_dynamic_no_prime import GridWorldDynamicGameNoPrime
from src.compositional_graphs.gridworld.gridworld_dynamic_doors_no_prime import GridWorldDynamicDoorsGameNoPrime

from src.compositional_graphs.gridworld.gridworld_dynamic_dfa_game_no_prime import GridWorldDynamicDFAGameNoPrime, GridWorldDynamicDoorsDFAGameNoPrime

# regret imports
from src.compositional_graphs.gridworld.gridworld_dynamic_regret_no_prime import GridWorldDynamicRegretGameNoPrime
from src.compositional_graphs.gridworld.gridworld_dynamic_regret_doors_no_prime import GridWorldDynamicDoorsRegretGameNoPrime


if __name__ == "__main__":
    """
    Main function to run the synthesis process based on parsed arguments.
    """
    # --- Setup Logging ---
    # Ensure the logs directory exists
    # if not os.path.exists('./logs'):
    #     os.makedirs('./logs')


    for run in range(ITERATIONS):
        # log setup
        setup_dict = {}
        
        setup_dict['rows'] = ROWS
        setup_dict['columns'] = COLUMNS
        setup_dict['sys_players'] = PLAYERS['sys']
        setup_dict['env_players'] = PLAYERS['env']
        setup_dict['init'] = INIT
        setup_dict['goal'] = GOAL
        setup_dict['formula'] = FORMULA
        setup_dict['restricted_env_locs'] = RESTRICTED_ENV_LOCS
        setup_dict['ltlf_flag'] = LTLF_FLAG
        setup_dict['budget'] = BUDGET
        setup_dict['prime'] = False if NO_PRIME else True
        setup_dict['enable_reordering'] = ENABLE_REORDERING
        setup_dict['cooperative_game'] = COOPERATIVE_GAME
        setup_dict['game_type'] = 'regret_game' if REGRET_GAME else 'dfa_game' if DFA_GAME else 'game'
        setup_dict['algorithm'] = ALGORITHM

        print(f"\n=== Starting Run {run + 1}/{ITERATIONS} ===")
        game = None
        
        # 1. Instantiate the correct game class based on arguments
        if REGRET_GAME:
            print("--- Initializing Regret DFA Game ---")
            if 'door' in GRID.keys():
                if NO_PRIME:
                    print("--- Initializing No Prime Regret Doors DFA Game ---")
                    game = GridWorldDynamicDoorsRegretGameNoPrime(rows=ROWS, columns=COLUMNS,
                                                                  init=INIT, goal=GOAL,
                                                                  formula=FORMULA, budget=BUDGET,
                                                                  grid=GRID,
                                                                  players=PLAYERS,
                                                                  restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                                  camera=CAMERA,
                                                                  ltlf_flag=LTLF_FLAG,
                                                                  cooperative_game=COOPERATIVE_GAME,
                                                                  enable_reordering=ENABLE_REORDERING)
                else:
                    print("--- Initializing Regret Doors DFA Game with Primes ---")
                    game = GridWorldDynamicDoorsRegretGame(rows=ROWS, columns=COLUMNS,
                                                           init=INIT, goal=GOAL,
                                                           formula=FORMULA, budget=BUDGET,
                                                           grid=GRID,
                                                           players=PLAYERS,
                                                           restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                           camera=CAMERA,
                                                           ltlf_flag=LTLF_FLAG,
                                                           cooperative_game=COOPERATIVE_GAME,
                                                           enable_reordering=ENABLE_REORDERING)
            
            else:
                if NO_PRIME:
                    print("--- Initializing No Prime Regret DFA Game ---")
                    game = GridWorldDynamicRegretGameNoPrime(rows=ROWS, columns=COLUMNS,
                                                             init=INIT, goal=GOAL,
                                                             formula=FORMULA, budget=BUDGET,
                                                             grid=GRID,
                                                             players=PLAYERS,
                                                             restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                             camera=CAMERA,
                                                             ltlf_flag=LTLF_FLAG,
                                                             cooperative_game=COOPERATIVE_GAME,
                                                             enable_reordering=ENABLE_REORDERING)
                else:
                    print("--- Initializing Regret DFA Game with Primes ---")
                    game = GridWorldDynamicRegretGame(rows=ROWS, columns=COLUMNS,
                                                      init=INIT, goal=GOAL,
                                                      formula=FORMULA, budget=BUDGET,
                                                      grid=GRID,
                                                      players=PLAYERS,
                                                      restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                      camera=CAMERA,
                                                      ltlf_flag=LTLF_FLAG,
                                                      cooperative_game=COOPERATIVE_GAME,
                                                      enable_reordering=ENABLE_REORDERING)
            setup_dict['goal'] = ''
        elif DFA_GAME:
            print("--- Initializing DFA Game ---")
            if 'door' in GRID.keys():
                if NO_PRIME:
                    print("--- Initializing No Prime Doors DFA Game ---")
                    game = GridWorldDynamicDoorsDFAGameNoPrime(rows=ROWS, columns=COLUMNS,
                                                               formula=FORMULA, 
                                                               init=INIT, goal=GOAL,
                                                               grid=GRID,
                                                               players=PLAYERS,
                                                               restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                               camera=CAMERA,
                                                               cooperative_game=COOPERATIVE_GAME,
                                                               ltlf_flag=LTLF_FLAG,
                                                               enable_reordering=ENABLE_REORDERING)
                else:
                    print("--- Initializing Doors DFA Game with Primes ---")
                    game = GridWorldDynamicDoorsDFAGame(rows=ROWS, columns=COLUMNS,
                                                        formula=FORMULA, 
                                                        init=INIT, goal=GOAL,
                                                        grid=GRID,
                                                        players=PLAYERS,
                                                        restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                        camera=CAMERA,
                                                        cooperative_game=COOPERATIVE_GAME,
                                                        ltlf_flag=LTLF_FLAG,
                                                        enable_reordering=ENABLE_REORDERING)
            else:
                if NO_PRIME:
                    print("--- Initializing No Prime DFA Game ---")
                    game = GridWorldDynamicDFAGameNoPrime(rows=ROWS, columns=COLUMNS,
                                                          init=INIT, goal=GOAL,
                                                          formula=FORMULA, 
                                                          grid=GRID,
                                                          players=PLAYERS,
                                                          restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                          camera=CAMERA,
                                                          ltlf_flag=LTLF_FLAG,
                                                          cooperative_game=COOPERATIVE_GAME,
                                                          enable_reordering=ENABLE_REORDERING)
                else:
                    print("--- Initializing DFA Game with Primes ---")
                    game = GridWorldDynamicDFAGame(rows=ROWS, columns=COLUMNS,
                                                   init=INIT, goal=GOAL,
                                                   formula=FORMULA,
                                                   grid=GRID,
                                                   players=PLAYERS,
                                                   restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                   camera=CAMERA,
                                                   ltlf_flag=LTLF_FLAG,
                                                   cooperative_game=COOPERATIVE_GAME,
                                                   enable_reordering=ENABLE_REORDERING)
            setup_dict['goal'] = ''
            setup_dict['budget'] = ''
        elif GAME:
            print("--- Initializing Standard Game ---")
            if 'door' in GRID.keys():
                if NO_PRIME:
                    print("--- Initializing No Prime Doors Standard Game ---")
                    game = GridWorldDynamicDoorsGameNoPrime(rows=ROWS, columns=COLUMNS,
                                                            init=INIT, goal=GOAL,
                                                            grid=GRID,
                                                            players=PLAYERS,
                                                            restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                            cooperative_game=COOPERATIVE_GAME,
                                                            enable_reordering=ENABLE_REORDERING)
                else:
                    print("--- Initializing Doors Standard Game with Primes ---")
                    game = GridWorldDynamicDoorsGame(rows=ROWS, columns=COLUMNS,
                                                     init=INIT, goal=GOAL,
                                                     grid=GRID,
                                                     players=PLAYERS,
                                                     restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                     cooperative_game=COOPERATIVE_GAME,
                                                     enable_reordering=ENABLE_REORDERING)
            else:
                if NO_PRIME:
                    print("--- Initializing No Prime Standard Game ---")
                    game = GridWorldDynamicGameNoPrime(rows=ROWS, columns=COLUMNS,
                                                    init=INIT, goal=GOAL,
                                                    players=PLAYERS, grid=GRID,
                                                    restricted_env_locs=RESTRICTED_ENV_LOCS,
                                                    cooperative_game =COOPERATIVE_GAME,
                                                    enable_reordering=ENABLE_REORDERING)
                else:
                    print("--- Initializing Standard Game with Primes ---")
                    game =GridWorldDynamicGame(rows=ROWS, columns=COLUMNS, 
                                               init=INIT, goal=GOAL,
                                               players= PLAYERS, grid=GRID,
                                               restricted_env_locs= RESTRICTED_ENV_LOCS,
                                               cooperative_game=COOPERATIVE_GAME,
                                               enable_reordering=ENABLE_REORDERING)
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
                strategy, _ = game.solve(verbose=False)
            elif ALGORITHM == 'BDD':
                strategy, _ = game.pure_bdd_solve(verbose=False)
            elif ALGORITHM == 'hybrid':
                strategy, _ = game.hybrid_solve(verbose=False)
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

        # 5. Log results
        comp_time.update(game.logger.comp_time)
        game.logger.log(setup_dict=setup_dict, comp_time=comp_time, abs_dict=game.log_game_details())
        # Add manager info to the log
        game.logger.run_data['MemoryInUse'] = game.manager.readMemoryInUse()
        import pprint
        pprint.pp(game.logger.run_data)
        if GAME:
            if COOPERATIVE_GAME:
                pass
            else:
                game.logger.dump_results_to_yaml(file_path=os.path.join('recuv_logs/game/no_prime/scenario_3/', f'{ROWS}x{COLUMNS}_{PLAYERS["sys"]}s_{PLAYERS["env"]}e_{ALGORITHM}_game'), iteration=run, add_time_stamp=False)
        elif DFA_GAME:
            if COOPERATIVE_GAME:
                raise NotImplementedError("Logging for cooperative DFA games not implemented yet.")
            else:
                game.logger.dump_results_to_yaml(file_path=os.path.join('recuv_logs/dfa_game/no_prime/scenario_4/', f'{ROWS}x{COLUMNS}_{PLAYERS["sys"]}s_{PLAYERS["env"]}e_{ALGORITHM}_dfa_game'), iteration=run, add_time_stamp=False)
        elif REGRET_GAME:
            raise NotImplementedError("Logging for regret games not implemented yet.")
        del game.manager
