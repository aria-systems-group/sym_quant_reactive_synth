from typing import List

# Iterations
ITERATIONS: int = 10

# Set up the game
BOXES: int = 3
LOCS: int = 18
BUDGET: int = 25
RATIO: int = 1
WEIGHT_FACTOR: int = 1
HUMAN_LOCS: range = range(5, LOCS + 1)
HUMAN_BOXES: List[int] = [2]

if BOXES == 3:
    INIT: List[str] = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l5']
elif BOXES == 4:
    INIT: List[str] = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l5', 'b3 l4']
elif BOXES == 5:
    INIT: List[str] = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l5', 'b3 l4', 'b4 l6']
GOAL: List[List[str]] = [['b0 l1']]
FORMULA: str = 'F(p01)'
NO_PRIME: bool = False

# Flags
LTLF_FLAG: bool = True
COOPERATIVE_GAME: bool = False
ENABLE_REORDERING: bool = False
ONLY_REACHABLE_STATES: bool = False

# Type of game to construct
GAME: bool = False
DFA_GAME: bool = True
REGRET_GAME: bool = False

assert sum([GAME, DFA_GAME, REGRET_GAME]) == 1, "Exactly one of GAME, DFA_GAME, or REGRET_GAME must be True."

# algorithm to use for strategy synthesis
ALGORITHM: str = 'ADD' # choose from 'ADD', 'BDD', 'hybrid'