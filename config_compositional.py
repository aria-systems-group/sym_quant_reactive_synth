from typing import List

# Set up the game
BOXES: int = 3
LOCS: int = 10
BUDGET: int = 25
RATIO: int = 1
HUMAN_LOCS: range = range(1, LOCS + 1)
HUMAN_BOXES: List[int] = [0]
INIT: List[str] = ['ready l3', 'b0 l2', 'b1 l3', 'b2 l6']
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
DFA_GAME: bool = False
REGRET_GAME: bool = True

assert sum([GAME, DFA_GAME, REGRET_GAME]) == 1, "Exactly one of GAME, DFA_GAME, or REGRET_GAME must be True."

# algorithm to use for strategy synthesis
ALGORITHM: str = 'ADD' # choose from 'ADD', 'BDD', 'hybrid'