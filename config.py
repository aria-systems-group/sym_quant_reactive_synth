import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SIM_CONFIG_PATH = '/root/ws/src/taskit/config/objects.yaml'

EXPLICIT_GRAPH: bool = False  # set this flag to true when you want to construct Explicit graph

GRIDWORLD: bool = False   # Set this flag to true when using gridworld example for graph search 
FRANKAWORLD: bool = False  # Set this flag to true when using manipulator scenarios for graph search
STRATEGY_SYNTHESIS: bool = True  # Set this flag to true when  when using manipulator scenarios for Strategy synthesis
REGRET_SYNTHESIS: bool = False  # Set this flag to true when you want compute regret minimizing strategies
REGRET_HYBRID: bool = False  # Set this flag to true when you contruct graph of utility and best response explicitly

MONOLITHIC_TR: bool = True  # Set this flag to true when using Monolithic TR during regret str synthesis

TWO_PLAYER_GAME: bool = True # Set this flag to true when you want to contruct a two-player game env.
TWO_PLAYER_GAME_BND: bool = False  # Set this flag to true when you want to construct som bounded no. off human interventions.
GAME_ALGORITHM = 'quant-adv' # choose qual for qualitative game, quant-adv for quantitative adversarial game, and quant-coop for cooperative game

HUMAN_INT_BND: int = 3

DIJKSTRAS: bool = False  # set this flag to true when you want to use Dijkstras
ASTAR: bool = False # set this flag to true when you want to use A* algorithm 

USE_LTLF: bool = True # Construct DFA from LTLf

DRAW_EXPLICIT_CAUSAL_GRAPH: bool = False
SIMULATE_STRATEGY: bool = True
GRID_WORLD_SIZE: int = 5
OBSTACLE: bool = False  # flag to load the obstacle gridworld and color the gridworld accordingly
DYNAMIC_VAR_ORDERING: bool = False

##################### Franka Declare supports and top location for valid Human Int. #########################
# SUP_LOC = ['l6', 'l7']   # support for Arch
# TOP_LOC = ['l8']         # top location for Arch
SUP_LOC = []
TOP_LOC = []

######################################################################
#################### FRANKA TABLE TOP FORMULAS #######################
######################################################################

##### FRANKA ARCH CONF #######
# formulas = ['F((p00 & p12 & p21) | (p10 & p01 & p22) | (p10 & p21 & p02) | (p20 & p01 & p12) | (p20 & p11 & p02))']
# formulas = ['F(p00 & p12 & p21) & G(~(p00 & p21) -> ~(p12))']   # this one works for sure

# Testing
# formulas = ['F(p00 & p11 & XF(p10 & p01))'] 

# formulas = ['F(p01 & free & F(p12 & free & F(p23 & free & F(p34 & free))))']
# formulas = ['F(p01 & free & F(p12 & free & F(p23 & free)))']
# formulas = ['(F((p01 & free)) & F((p12 & free)) & F((p23 & free)) & F((p34 & free)))']

# simple one box formula 
# formulas = ['F(p00 & p11 & !p22 & XF(p22))']
# formulas = ['F(p00 & p11 & p22 & p33 & p44)']

### BENCHMARKING formulas
# formulas = ['F(p00 & p11 & p22 & p33 & p44)']  # 5 boxes one
# formulas = ['F(p00 & p11 & p22)']
# formulas = ['F(p06 & F(p07 & F(p06)))']

### TESTING Straight line alignment
# formulas = ['F(p01 & p20 & p42) | F(p36 & p17 & p48)']   # straight line from ICRA22
# formulas = ['F(p01 & p10 & p32) | F(p26 & p38)']   # straight line from ICRA22 - modified
# formulas = ['F(p01 | p17)']   # smaller version of straight line alignment for sanity checking

#### Testing Arch constrcution example
# formulas = ['( F(p01 & p20 & p12) & G(~(p01 & p20) -> ~(p12)) ) | ( F(p36 & p07 & p18) & G(~(p36 & p07) -> ~(p18)) )']
# formulas = ['( F(p01 & p20 & p12) ) | ( F(p36 & p07 & p18) )']

##### TESTING two block regret formulas
# formulas = ['F(p00 & XF(p21)) | F(p18 & p06)']

##### Simple Formula
# formulas = ['F(p01) | F(p17)']

## VARYING BOXES Formulas
formulas = ['F(p01)']  # p00
# formulas = ['F(p01 & p22)']  # p01
# formulas = ['F(p01 & p22)']  # p02
# formulas = ['F(p01 & p22 & p43)']  # p03
# formulas = ['F(p01 & p22 & p43)']  # p04

# VARYING LOCS Formula
# formulas = ['F(p01 & p22)']  # same formula for all problem files


### BENCHMARKING formulas
# formulas = ['F(p00 & p11)']
# formulas = ['F(p01 & XF(p17))']   # tricky complicated formula
# formulas = ['F(p11 & p06 & F(p07 & F(p06)))']   # neat complicated formula
# formulas = ['F(p01)']
# formulas = ['F(p01 | p11)'] 


# formulas = [
#             # 'F((p01 & p20 & free))',
#             'F(p20 & free & F(p01 & free))',
#             # 'F((p12 & free))', 
#             # 'F((p23 & free))',
#             # 'F((p34 & free))'
#             ]

# formulas = ['(F((p01 & free)) & F((p12 & free)) & F((p23 & free)) & F((p34 & free)))',
#             '(F((p12 & free)) & F((p23 & free)) & F((p34 & free)) & F((p01 & free)))', 
#             '(F((p23 & free)) & F((p34 & free)) & F((p01 & free)) & F((p12 & free)))', 
#             '(F((p34 & free)) & F((p01 & free)) & F((p12 & free)) & F((p23 & free)))']


######################################################################
#################### SIMPLE GRIDWORLD FORMULAS #######################
######################################################################


 # 5 by 5 formulas
# formulas = ["F(l21) & F(l5) & F(l25) & F(l1)",
#             # "F(l19) & F(l7) & F(l9) & F(l17)",
#             # "F(l23) & F(l3) & F(l11) & F(l15)",
#             # "F(l16) & F(l24) & F(l2) & F(l10) "
#             ]

# formulas = [#"F(l21) & F(l5) & F(l25) & F(l1)",
#             # "F(l22) & F(l4) & F(l20) & F(l6)",
#             "F(l23) & F(l3) & F(l11) & F(l15)",
#             "F(l16) & F(l24) & F(l2) & F(l10) "
#             ]

# single 12 nested formula
# formulas = ['F(l2) & F(l91) & F(l93) & F(l4) & F(l95) & F(l6) & F(l97) & F(l8) & F(l99) & F(l10) & F(l56) & F(l45)']

# single 10 nested formulas
# formulas = ['F(l4) & F(l95) & F(l6) & F(l97) & F(l8) & F(l99) & F(l10) & F(l12) & F(l56) & F(l45)']
# formulas = [' F(l4)']


# formulas = ['F(l2)',
#             'F(l91)', 
#             'F(l93)',
#             'F(l4)', 
#             'F(l95)',
#             'F(l6)',
#             'F(l97)',
#             'F(l8)',
#             'F(l99)',
#             'F(l10)'
#             ] 




# for 20 by 20 grid world
# formulas = ["F(l191 & F(l110) & F(l200))",
#             # "F(l289 & F(l212) & F(l119))",
#             # "F(l123 & F(l13) & F(l111))",
#             # "F(l165 & F(l324) & F(l32))"
#             ]

# formulas = ['F(l7)', 'F(l13)', 'F(l19)', 'F(l25)']
# formulas = ['F(l7)', 'F(l13)', 'F(l19)']
# formulas = ['F(l13)', 'F(l7)']
# formulas = ['F(l25)', 'F(l2)', 'F(l21)', 'F(l5)']
# formulas = ["F(l21) & F(l5) & F(l25)"]
# formulas = ["F(l25)"]

# for 10 by 10 gridworld - conflicting formulas
# formulas = ['F(l1 & F(l100))', 'F(l100 & F(l1))']

# for 10 by 10 gridworld
# formulas = ["F(l91 & F(l10) & F(l100))",
#             "F(l92 & F(l9) & F(l90))",
#             "F(l93 & F(l8) & F(l80))",
#             "F(l94 & F(l7) & F(l70))",
#             "F(l95 & F(l6) & F(l60))",
#             "F(l96 & F(l5) & F(l50))",
#             "F(l97 & F(l4) & F(l40))",
#             "F(l98 & F(l3) & F(l30))",
#             # "F(l99 & F(l2) & F(l20))", # from here onwards formulas are just for stress testing
#             # "F(l81 & F(l18) & F(l79))",   # 976,562,500
#             # "F(l82 & F(l17) & F(l69))",   # 4,882,812,500
#             # "F(l83 & F(l16) & F(l59))",   # 24,414,062,500   ~10sec for A*
#             # "F(l84 & F(l15) & F(l49))",   # 122,070,312,500  ~20sec for A*
#             # "F(l85 & F(l14) & F(l39))",   # 610,351,562,500  ~80sec for A*   - Dijkstras algorithm broke here 
#             # "F(l86 & F(l13) & F(l29))",   # 3.0517578e+12    ~ 250sec for A*
#             # "F(l87 & F(l12) & F(l19))",
#             # "F(l88 & F(l11) & F(l))",
#             ]


# 5 state formula for 5x5 GW
# formulas = ["F(l21 & F(l5) & F(l25))",
#             "F(l22 & F(l4) & F(l20))",
#             "F(l23 & F(l3) & F(l15))",
#             "F(l24 & F(l2) & F(l10))",
#             "F(l16 & F(l21) & F(l2))",
#             "F(l11 & F(l22) & F(l3))",
#             "F(l6 & F(l23) & F(l4))",
#             "F(l2 & F(l20) & F(l16))",
#             ]



# list of formula
# formulas = [
#     'F(l25)',
#     '!l2 & !l7 U l13',
# #     'F(l25) & F(l15)',
# #     'F(l19 & F(l13))',   # simple Formula w 2 states
# #     # 'F(l13 & (F(l21) & F(l5)))',
# #     # 'F(l6) & F(l2)', 
# #     # 'F(l13 & (F(l21 & (F(l5)))))',
# #     # "F(l21 & (F(l5 & (F(l25 & F(l1))))))",   # traversing the gridworld on the corners
#     # "F(l91 & (F(l10 & (F(l100 & F(l1))))))"   # traversing the gridworld on the corners for 10 x 10 gridworld
# #     # "F(l400)",
# #     # "F(l100 & F(l1))",
# #     # "F(l100 & F(l1 & F(l91)))"
# #     # "F(l381 & (F(l20 & (F(l400 & F(l1))))))",   # traversing the gridworld on the corners for 20 x 20 gridworld
# #     # "F(l381 & (F(l20 & (F(l400)))))",
# #     # "F(l381 & (F(l20)))",
#     ]