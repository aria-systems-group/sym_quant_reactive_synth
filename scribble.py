# testing stuff

from cudd import Cudd, BDD, ADD
 
m = Cudd()

# first create the variables
i0 = m.bddVar(0, 'i0')
o0, o1, o2 = (m.bddVar(i+1,   'o' + str(i)) for i in range(3))
a0_0 = m.bddVar(4,'a0')
b0_0, b0_1, b0_2 = (m.bddVar(i+5,   'b0' + str(i)) for i in range(3))
x0, x1, x2, x3 = (m.bddVar(i+8,   'x' + str(i)) for i in range(4))

# set of acceptins states.
From = ~a0_0 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2)) | ~b0_1 & (b0_2 & (x0 & x1 & x2 & ~x3 | ~x0 & x1 & ~x2 & x3) | ~b0_2 & (x0 & x1 & ~x2 | x1 & ~x2 & x3))) | ~b0_0 & (b0_1 & (b0_2 & (x0 & x1 & ~x2 | x1 & ~x2 & x3) | ~b0_2 & (x0 & x1 & x2 & x3 | ~x0 & x1 & ~x2 & x3))))

# dfa transition fun list
dfa_bdd_transition_fun_list = a0_0 & (b0_0 | (b0_2 | ~b0_1))

# game transition fun list
ts_actions = [None for _ in range(2)]
ts_actions[0] = o0 & (o1 & (o2 & b0_0 & b0_1 & b0_2 & ~x0 & x1 & x2 | ~o2 & (b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & b0_0 & ~b0_1 & b0_2 & x0 & x1 & x2 & ~x3 | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))) | ~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3)))))) | ~i0 & (o0 & (o1 & (o2 & b0_0 & b0_1 & b0_2 & ~x0 & x1 & x2 | ~o2 & (b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & (b0_0 & (~b0_1 & (b0_2 & x0 & x1 & x2 & ~x3 | ~b0_2 & x0 & x1 & ~x2)) | ~b0_0 & b0_1 & b0_2 & x0 & x1 & ~x2) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))) | ~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3)))))))
ts_actions[1] = i0 & (o0 & (o1 & (o2 & b0_0 & b0_1 & b0_2 & x0 & ~x1 & ~x2 | ~o2 & (b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & (b0_0 & ~b0_1 & ~b0_2 & x0 & x1 & ~x2 | ~b0_0 & (b0_1 & (b0_2 & x0 & x1 & ~x2 | ~b0_2 & x0 & x1 & x2 & x3))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))) | ~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3))))))) | (o0 & (o1 & (o2 & b0_0 & b0_1 & b0_2 & x0 & ~x1 & ~x2 | ~o2 & (b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & ~b0_0 & b0_1 & ~b0_2 & x0 & x1 & x2 & x3 | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))) | ~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3)))))))
# ts_actions[2] = i0 & (o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2 & x3)))) | ~o2 & (b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & (b0_0 & (~b0_1 & (b0_2 & x0 & x1 & x2 & ~x3 | ~b0_2 & x0 & x1 & ~x2)) | ~b0_0 & b0_1 & b0_2 & x0 & x1 & ~x2) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))) | ~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3))))))) | (o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2 & x3)))) | ~o2 & (b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & b0_0 & ~b0_1 & b0_2 & x0 & x1 & x2 & ~x3 | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))) | ~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3)))))))
# ts_actions[3] = i0 & (o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2)))) | ~o2 & ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3) | ~o1 & o2 & b0_0 & ~b0_1 & ~b0_2 & x0 & x1 & ~x2) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2))))))) | ~i0 & (o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2)))) | ~o2 & ~b0_0 & b0_1 & ~x0 & x1 & ~x2 & x3) | ~o1 & o2 & ~b0_0 & b0_1 & b0_2 & x0 & x1 & ~x2) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2)))))))
# ts_actions[4] = o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2)))) | ~o2 & b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3) | ~o1 & (o2 & (b0_0 & (~b0_1 & (b0_2 & x0 & x1 & x2 & ~x3 | ~b0_2 & x0 & x1 & ~x2)) | ~b0_0 & (b0_1 & (b0_2 & x0 & x1 & ~x2 | ~b0_2 & x0 & x1 & x2 & x3))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3))))))
# ts_actions[5] = o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & x3 | ~x0 & x1 & x2 & x3)))) | ~o2 & b0_0 & ~b0_1 & ~x0 & x1 & ~x2 & x3) | ~o1 & (~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (~o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & x3))))))
# ts_actions[6] = o0 & (o1 & (o2 & b0_0 & b0_1 & b0_2 & x0 & ~x1 & ~x2 | ~o2 & (b0_0 & ~b0_1 & b0_2 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~b0_2 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & (b0_0 & ~b0_1 & b0_2 & x0 & x1 & x2 & ~x3 | ~b0_0 & (b0_1 & (b0_2 & x0 & x1 & ~x2 | ~b0_2 & x0 & x1 & x2 & x3))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))))) | ~i0 & (o0 & (o1 & (o2 & b0_0 & b0_1 & b0_2 & x0 & ~x1 & ~x2 | ~o2 & (b0_0 & ~b0_1 & b0_2 & ~x0 & x1 & ~x2 & x3 | ~b0_0 & b0_1 & ~b0_2 & ~x0 & x1 & ~x2 & x3)) | ~o1 & (o2 & (b0_0 & (~b0_1 & (b0_2 & x0 & x1 & x2 & ~x3 | ~b0_2 & x0 & x1 & ~x2)) | ~b0_0 & (b0_1 & (b0_2 & x0 & x1 & ~x2 | ~b0_2 & x0 & x1 & x2 & x3))) | ~o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 | ~x0 & x1 & x2 & ~x3)))))) | ~o0 & (o1 & (o2 & (b0_0 & (b0_1 & (b0_2 & (x0 & ~x1 & ~x2 & ~x3 | ~x0 & x1 & x2)))))))

#### T0 cubes
# T0_cubes = [[0, 0, 0, 1, 2, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2], [0, 0, 1, 0, 2, 1, 1, 1, 0, 1, 1, 2], [0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1], 
# [0, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2], [0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 2, 1, 1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2],
#  [0, 1, 0, 1, 2, 0, 1, 1, 1, 1, 0, 2], [0, 1, 0, 1, 2, 1, 0, 0, 1, 1, 0, 2], [0, 1, 0, 1, 2, 1, 0, 1, 1, 1, 1, 0], [0, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 1], 
# [0, 1, 1, 0, 2, 1, 0, 2, 0, 1, 0, 1], [0, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2], [1, 0, 0, 1, 2, 1, 1, 1, 0, 1, 1, 1], [1, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2], 
# [1, 0, 1, 0, 2, 1, 1, 1, 0, 1, 1, 2], [1, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1], [1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2], [1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0], 
# [1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 1, 0], [1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2], [1, 1, 0, 1, 2, 1, 0, 1, 1, 1, 1, 0], [1, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 1], 
# [1, 1, 1, 0, 2, 1, 0, 2, 0, 1, 0, 1], [1, 1, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2]]

# t1_subset_action = 

#### T1 cube
# [[0, 0, 0, 1, 2, 1, 1, 1, 0, 1, 1, 1], [0, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2], [0, 0, 1, 0, 2, 1, 1, 1, 0, 1, 1, 2], [0, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1], 
# [0, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2], [0, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0], [0, 1, 0, 0, 2, 1, 1, 1, 0, 1, 1, 0], [0, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2], 
#  [0, 1, 0, 1, 2, 0, 1, 0, 1, 1, 1, 1], [0, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 1], [0, 1, 1, 0, 2, 1, 0, 2, 0, 1, 0, 1], [0, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 2], 
#  [1, 0, 0, 1, 2, 1, 1, 1, 0, 1, 1, 1], [1, 0, 0, 1, 2, 1, 1, 1, 1, 0, 0, 2], [1, 0, 1, 0, 2, 1, 1, 1, 0, 1, 1, 2], [1, 0, 1, 0, 2, 1, 1, 1, 1, 0, 0, 1], 
#  [1, 0, 1, 1, 2, 1, 1, 1, 0, 1, 1, 2], [1, 0, 1, 1, 2, 1, 1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 2, 1, 1, 1, 0, 1, 1, 0], [1, 1, 0, 0, 2, 1, 1, 1, 1, 0, 0, 2], 
#  [1, 1, 0, 1, 2, 0, 1, 0, 1, 1, 1, 1], [1, 1, 0, 1, 2, 0, 1, 1, 1, 1, 0, 2], [1, 1, 0, 1, 2, 1, 0, 0, 1, 1, 0, 2], [1, 1, 1, 0, 2, 0, 1, 2, 0, 1, 0, 1], 
#  [1, 1, 1, 0, 2, 1, 0, 2, 0, 1, 0, 1], [1, 1, 1, 1, 2, 1, 1, 1, 1, 0, 0, 2]]


ts_actions_exist = [None for _ in range(2)]
ts_actions_exist[0] = ts_actions[0].existAbstract(b0_0)
ts_actions_exist[1] = ts_actions[1].existAbstract(b0_1)

# perform vector to ensure everything is setup correctly.
mod_win_state: BDD = From.vectorCompose([a0_0], [dfa_bdd_transition_fun_list])
# pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0, b0_1, b0_2, x0, x1, x2, x3], ts_actions)
# pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0], [ts_actions[0]])
pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0, b0_1], ts_actions_exist)
# pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0], ts_actions)
# pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0], ts_actions)
# pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0], ts_actions)
# pre_prod_state: BDD = mod_win_state.vectorCompose([b0_0], ts_actions)

# reversed_vars_boi = list(reversed([b0_0, b0_1, b0_2, x0, x1, x2, x3]))
# reversed_ts_boi = list(reversed(ts_actions))
# trial: BDD = mod_win_state.vectorCompose(reversed_vars_boi,reversed_ts_boi)

# print(trial == pre_prod_state)


game_dfa_win_states = mod_win_state
game_dfa_win_states = game_dfa_win_states.compose(ts_actions_exist[0], m.bddVariables().index(b0_0))
game_dfa_win_states = game_dfa_win_states.compose(ts_actions_exist[1], m.bddVariables().index(b0_1))

# game_dfa_win_states = game_dfa_win_states.compose(ts_actions[0], m.bddVariables().index(b0_0)) & game_dfa_win_states.compose(ts_actions[1], m.bddVariables().index(b0_1)) 



# first constraint and then compose
# tmp_f = game_dfa_win_states.constrain(ts_actions[0] | ts_actions[1])
# game_dfa_win_states: BDD = game_dfa_win_states.compose(ts_actions[0], m.bddVariables().index(b0_0))
# game_dfa_win_states: BDD = tmp_f.compose(ts_actions[0], m.bddVariables().index(b0_0))
# tmp_f = game_dfa_win_states.constrain(ts_actions[1])
# game_dfa_win_states: BDD = game_dfa_win_states.compose(ts_actions[1], m.bddVariables().index(b0_1))

# testing = mod_win_state
# testing: BDD = testing.compose(ts_actions[1], m.bddVariables().index(b0_1))
# testing: BDD = testing.compose(ts_actions[0], m.bddVariables().index(b0_0))

print(game_dfa_win_states == pre_prod_state)

# # do the same thing using compose operation
# game_dfa_win_states = mod_win_state
# for var, bdd_func in zip([b0_0, b0_1, b0_2, x0, x1, x2, x3], ts_actions):
#     index = m.bddVariables().index(var)
#     game_dfa_win_states: BDD = game_dfa_win_states.compose(bdd_func, index)



# print(pre_prod_state)
# print("**************************************************************************************")
# print(game_dfa_win_states)


# auxilary variables
# i0,i1,i2,i3 = (m.bddVar(i, 'i' + str(i)) for i in range(4))
# x0,x1,x2,x3 = (m.bddVar(i+4,   'x' + str(i)) for i in range(4))
# y0,y1,y2,y3 = (m.bddVar(i+8, 'y' + str(i)) for i in range(4))


# f = x0 | ~x1 & x3
# print(f)
# # x0 | ~(x1 | ~x3) 

# # print(f.swapVariables([x0,x1,x2,x3],[y0,y1,y2,y3]))
# # y0 | ~(y1 | ~y3)
# # print(f.swapVariables([x0,x1,x2,x3],[y3,y2,y1,y0]))
# # y0 & (y3 | ~y2) | ~y0 & y3

# h = y2 & y3 & i1
# g = y1 | y3

# f1 = f.compose(h, 5)
# print(f1)

# # print(f1.compose(g,4))
# f2 = f1.compose(g, 4)
# print(f2)

# # now try the other way around
# # print(f1.compose(g,4))
# f3 = f.compose(g, 4)
# print(f3)

# f4 = f3.compose(h, 5)
# print(f4 == f2)

# k = ~y0
# print(f.compose(k,2))

# vector compose
# print(f.vectorCompose([x0, x1], [g, h]))


