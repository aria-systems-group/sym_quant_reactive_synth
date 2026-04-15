from typing import List, Tuple, Dict

# Iterations
ITERATIONS: int = 1

ROWS: int = 10
COLUMNS: int = 10
INIT: List[Tuple[int, int]] = [(0, 0), (0, 1)]
GOAL: List[Tuple[int, int]] = [(ROWS - 1, COLUMNS - 1)]
GRID: Dict[str, List] = dict({'goal': [*GOAL], 'door': [(1, 2)]})
PLAYERS: Dict[str, int] = {'sys': 1, 'env': 1}
BUDGET: int = 0
FORMULA: str = 'F(goal)'
CAMERA: bool = False
NO_PRIME: bool = True

# Flags
LTLF_FLAG: bool = True
COOPERATIVE_GAME: bool = False
ENABLE_REORDERING: bool = False
RESTRICTED_ENV_LOCS: List[Tuple[int, int]] = []

# Type of game to construct
GAME: bool = True
DFA_GAME: bool = False
REGRET_GAME: bool = False 

assert sum([GAME, DFA_GAME, REGRET_GAME]) == 1, "Exactly one of GAME, DFA_GAME, or REGRET_GAME must be True."

# algorithm to use for strategy synthesis
ALGORITHM: str = 'BDD' # choose from 'ADD', 'BDD', 'hybrid'
