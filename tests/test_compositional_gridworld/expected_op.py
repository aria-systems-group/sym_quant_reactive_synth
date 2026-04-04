"""
 This file hold the expected values for the test-cases
"""
import math

from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class ExpectedGameConfig:
    opt_init_sval: int
    vi_layers: int
    realizable: bool = True
    

Expected_op_SCENARIOS = {
    "2x2_simple": ExpectedGameConfig(opt_init_sval=2, vi_layers=4),
    "3x3_simple": ExpectedGameConfig(opt_init_sval=4, vi_layers=8),
    "3x3_2sys": ExpectedGameConfig(opt_init_sval=7, vi_layers=12),
    "3x3_2sys_2env": ExpectedGameConfig(opt_init_sval=7, vi_layers=16),
    "2x2_no_wall": ExpectedGameConfig(opt_init_sval=2, vi_layers=4),
    "5x5_no_wall_2env": ExpectedGameConfig(opt_init_sval=8, vi_layers=24)
}

Expected_op_SCENARIOS_DOOR = {
    "2x2_2sys": ExpectedGameConfig(opt_init_sval=2, vi_layers=6),
    "3x3_simple": ExpectedGameConfig(opt_init_sval=4, vi_layers=8),
    "3x3_complex": ExpectedGameConfig(opt_init_sval=4, vi_layers=8),
    "3x3_2env": ExpectedGameConfig(opt_init_sval=4, vi_layers=15),
    "3x3_2sys": ExpectedGameConfig(opt_init_sval=7, vi_layers=15),
    "3x3_3env_realizable": ExpectedGameConfig(opt_init_sval=4, vi_layers=20),
    "3x3_3env_unrealizable": ExpectedGameConfig(opt_init_sval=math.inf, vi_layers=20, realizable=False)
}



DFAGame_Expected_op_SCENARIOS = {
    "2x2_simple": ExpectedGameConfig(opt_init_sval=2, vi_layers=4),
    "3x3_simple": ExpectedGameConfig(opt_init_sval=math.inf, vi_layers=8, realizable=False),
    "3x3_2sys": ExpectedGameConfig(opt_init_sval=7, vi_layers=12),
    "3x3_2sys_2env": ExpectedGameConfig(opt_init_sval=7, vi_layers=16),
    "2x2_no_wall": ExpectedGameConfig(opt_init_sval=2, vi_layers=4),
    "5x5_no_wall_2env": ExpectedGameConfig(opt_init_sval=8, vi_layers=24)
}

DFAGame_Expected_op_SCENARIOS_DOOR = {
    "2x2_2sys": ExpectedGameConfig(opt_init_sval=2, vi_layers=6),
    "3x3_simple": ExpectedGameConfig(opt_init_sval=4, vi_layers=8),
    "3x3_complex": ExpectedGameConfig(opt_init_sval=5, vi_layers=10),
    "3x3_2env": ExpectedGameConfig(opt_init_sval=4, vi_layers=15),
    "3x3_2sys": ExpectedGameConfig(opt_init_sval=7, vi_layers=15),
    "3x3_3env_realizable": ExpectedGameConfig(opt_init_sval=4, vi_layers=20),
    "3x3_3env_unrealizable": ExpectedGameConfig(opt_init_sval=math.inf, vi_layers=20, realizable=False)
}