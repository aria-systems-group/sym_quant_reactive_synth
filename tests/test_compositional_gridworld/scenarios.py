from dataclasses import dataclass, field
from typing import List, Tuple, Dict

@dataclass
class GameConfig:
    rows: int
    columns: int
    init: List[Tuple[int, int]]
    goal: List[Tuple[int, int]]
    grid: Dict[str, List]
    players: Dict[str, int]
    budget: int = 0
    formula: str = None
    ltlf_flag: bool = True
    camera: bool = False
    cooperative_game: bool = False
    enable_reordering: bool = False
    restricted_env_locs: List[Tuple[int, int]] = field(default_factory=list)
    

SCENARIOS = {
    "2x2_simple": GameConfig(rows=2, columns=2, init=[(0, 0), (1, 0)], 
                             goal=[(1, 1)], grid={'wall': [(0, 1)], 'goal': [(1, 1)]},
                             formula='F(goal)', camera=False, budget=4,
                             cooperative_game=False, players={'sys': 1, 'env': 1}),
    "3x3_simple": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0)], 
                             goal=[(2, 2)], grid={'wall': [(0, 1),(2, 1)], 'goal': [(2, 2)]},
                             formula='F(goal) & G!c', camera=False, budget=6,
                             cooperative_game=False, players={'sys': 1, 'env': 1}),
    "3x3_2sys": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0), (1, 0)], 
                           goal=[(2, 2)], grid={'wall': [(0, 1),(2, 1)], 'goal': [(2, 2)]}, 
                           formula='F(goal)', camera=False, budget=8,
                           cooperative_game=False, players={'sys': 2, 'env': 1}),
    "3x3_2sys_2env": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0), (1, 0), (1, 2)], 
                             goal=[(2, 2)], grid={'wall': [(0, 1),(2, 1)], 'goal': [(2, 2)]},
                             formula='F(goal)', camera=False, budget=10,
                             cooperative_game=False, players={'sys': 2, 'env': 2}),
    "2x2_no_wall": GameConfig(rows=2, columns=2, init=[(0, 0), (1, 0)], 
                              goal=[(1, 1)], grid={'goal': [(1, 1)]},
                              formula='F(goal)', camera=False, budget=4,
                              cooperative_game=False, players={'sys': 1, 'env': 1}), 
    "5x5_no_wall_2env": GameConfig(rows=5, columns=5, init=[(0, 0), (4, 0), (0, 4)], 
                              goal=[(4, 4)], grid={'goal': [(4, 4)]},
                              formula='F(goal)', camera=False, budget=14,
                              cooperative_game=False, players={'sys': 1, 'env': 2}),            
}

SCENARIOS_DOOR = {
    "2x2_2sys": GameConfig(rows=3, columns=3, init=[(0, 0), (1, 0), (0, 1)], 
                            goal=[(1, 1)], grid={'goal': [(1, 1)], 'door': [(0, 1)]},
                            formula='F(goal)', camera=False, budget=4,
                            players={'sys': 2, 'env': 1}), 
    "3x3_simple": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0)], 
                             goal=[(2, 2)], grid={'wall': [(0, 1), (2, 1)], 'door': [(1, 1)], 'goal': [(2, 2)]},
                             formula='F(goal)', camera=False, budget=8,
                             cooperative_game=False, players={'sys': 1, 'env': 1}),
    "3x3_complex": GameConfig(rows=3, columns=3,
                              init=[(0, 0), (2, 0)], goal=[(2, 2)],
                              formula='F(goal & X(!goal))', camera=False, budget=10,
                              grid={'wall': [(0, 1), (2, 1)], 'goal': [(2, 2)], 'door': [(1, 1)]},
                              cooperative_game=False, players={'sys': 1, 'env': 1}),
    "3x3_2env": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0), (0, 2)], 
                            goal=[(2, 2)], grid={'wall': [(0, 1), (2, 1)], 'goal': [(2, 2)], 'door': [(1, 1)]},
                            formula='F(goal)', camera=False, budget=10,
                            players={'sys': 1, 'env': 2}),
    "3x3_2sys": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0), (0, 2)], 
                            goal=[(2, 2)], grid={'wall': [(0, 1), (2, 1)], 'goal': [(2, 2)], 'door': [(1, 1)]},
                            formula='F(goal)', camera=False, budget=10,
                            players={'sys': 2, 'env': 1}),
    "3x3_3env_realizable": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0), (0, 2), (2, 2)], 
                            goal=[(2, 2)], grid={'wall': [(0, 1), (2, 1)], 'goal': [(2, 2)], 'door': [(1, 1)]},
                            formula='F(goal)', camera=False, budget=10,
                            players={'sys': 1, 'env': 3}),
    "3x3_3env_unrealizable": GameConfig(rows=3, columns=3, init=[(0, 0), (2, 0), (1, 2), (2, 2)], 
                            goal=[(2, 2)], grid={'wall': [(0, 1), (2, 1)], 'goal': [(2, 2)], 'door': [(1, 1)]},
                            formula='F(goal)', camera=False, budget=10,
                            players={'sys': 1, 'env': 3})
}