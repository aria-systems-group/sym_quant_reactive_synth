from functools import reduce
from typing import List, Tuple, Union

from cudd import ADD, BDD

from src.compositional_graphs.symbolic_partitioned_dfa_game  import SymbolicPartitionedDFAGame
from src.compositional_graphs.symbolic_partitioned_dfa_game_noprime  import SymbolicPartitionedDFAGameNoPrime


class FrankaWorldArchAria(SymbolicPartitionedDFAGame):
    def __init__(self, 
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 restricted_human_boxes: List[int],
                 ltlf_flag: bool = True,
                 weight_factor: int = 1,
                 enable_reordering: bool = False,
                 only_reachable_states: bool = False):
        super().__init__(boxes, locs,
                         ratio, init,
                         goal, formula,
                         restricted_human_locs, restricted_human_boxes,
                         ltlf_flag, weight_factor,
                         enable_reordering=False, only_reachable_states=False)
        if enable_reordering:
            self.manager.autodynEnable()
    
    
    def post_process_transition_relation(self):
        super().post_process_transition_relation()
        debug: bool = False

        # now lets rmeove human moves that move blocks to the right
        mono_invalid_hmoves = self.manager.addZero()
        possible_hmoves: ADD =  reduce(lambda x, y: x | y, self.env_action_cube_list)
        kval_cube = reduce(lambda x, y: x | y, [self.kVar_map_sym[f'k{i}'] for i in range(0, self.ratio)])
        for hloc in self.human_locs:
            for hbox in self.human_boxes:
                # the box can only to its left, thats from 1 to locs
                valid_moves = [self.action_map_sym.get(f'hmove b{hbox} l{valid_loc}', self.manager.addZero()) for valid_loc in range(1, hloc + 1)]
                valid_moves_cube: ADD = reduce(lambda x, y: x | y, valid_moves) | self.action_map_sym['hmove noop'] | self.hmove_not_b[hbox]

                for tr_key, tr_dd in self.transition_relation.items():
                    if tr_key == 't0':
                        continue
                    mono_invalid_hmoves |= tr_dd & self.tVar_map_sym['human'] & kval_cube & self.xVar_map_sym[f'b{hbox} l{hloc}'] & (~valid_moves_cube & possible_hmoves)
                    self.transition_relation[tr_key] = (self.tVar_map_sym['human'] & kval_cube & self.xVar_map_sym[f'b{hbox} l{hloc}'] & (~valid_moves_cube & possible_hmoves)).ite(self.manager.addZero(), tr_dd)
        
        # brute force that we try to improve later
        for pred, pred_cube_str in self.pVar_map.items():
            care_edge = mono_invalid_hmoves & self.xVar_map_sym[pred]
            if care_edge.isZero():
                continue
            pred_clause_prime_string = ''
            if 'in-transit' in pred:
                to_box = pred.split(' ')[1] # get the box
                pred_clause_prime_string = self.xVar_map[f'to-obj {to_box}']
                print(f"Adding invalid edges From '{pred}' to 'to-obj {to_box}':")
            elif 'in-transfer' in pred:
                to_loc = pred.split(' ')[1] # get the location
                pred_clause_prime_string = self.xVar_map[f'holding {to_loc}']
                print(f"Adding invalid edges From '{pred}' to 'holding {to_loc}':")
            elif 'holding' in pred or 'ready' in pred:
                pred_clause_prime_string = pred_cube_str
                print(f"Adding invalid edges From '{pred}' to '{pred}':")
            else:
                continue

            for sidx, s in enumerate(pred_clause_prime_string):
                if s == '1':
                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= care_edge
            
            # debugging 
            # self.convert_cube_to_state_ADD(care_edge, action=True, verbose=True)

        for bidx in range(self.boxes):
            for pred, box_cube_Str in self.bVars_map[bidx].items():    
                # box remains in the same location if human does not move it/ or chooses an invalid edge
                care_edge = mono_invalid_hmoves & self.xVar_map_sym[pred]
                if care_edge.isZero():
                    continue
                for sidx, s in enumerate(self.xVar_map[pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[bidx][sidx].bddPattern().__str__()] |= care_edge
                
                print(f"Adding invalid edges From '{pred}' to '{pred}':")
                # self.convert_cube_to_state_ADD(care_edge, action=True, verbose=True)

        # add the invalid edges to k0
        for sidx, s in enumerate(self.kVar_map['k0']):
            if s == '1':
                self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= mono_invalid_hmoves
        
        if debug:
            print("Valid edges:")
            self.convert_cube_to_state_ADD(mono_invalid_hmoves, action=True, verbose=True)
    

    def check_valid_human_move(self, curr_state: List[str], action: str):
        box_idx = 3
        action = super().check_valid_human_move(curr_state=curr_state, action=action)
        if action.startswith('hmove noop'):
            return 'hmove noop'
        else:
            assert action.startswith('hmove'), "Unknown human action. Cannot proceed!!"
            b_idx = action.split(' ')[1]
            hmove_to_loc = action.split(' ')[2]
            split_str = curr_state[box_idx].split(', ')
            curr_box__loc = split_str[int(b_idx[1:])].split(' ')[1]
            
            # if the current box is to the LEFT of the location that human is trying to move it to, then its an invalid move
            if int(curr_box__loc[1:]) < int(hmove_to_loc[1:]):
                return 'hmove noop'
            else:
                return action
    
    def test_pre_image(self):
        # goal_cube = self.tVar_map_sym['robot'] & self.xVar_map_sym['ready l3'] & self.xVar_map_sym['b0 l1'] & self.kVar_map_sym['k1'] & self.dfa_handle.goal_latch
        goal_cube = self.xVar_map_sym['b0 l1'] & self.dfa_handle.goal_latch
        # goal state is b0 and l0 and ready l0
        print('Goal state:', goal_cube)

        # first evolve over the DFA
        # dfa_preimage = self.preimage_test(From=goal_cube, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation.values()))
        dfa_preimage = self.preimage_test(From=goal_cube, latches=self.qVars, prime_latches=self.prime_qVars, ts_action=list(self.dfa_handle.dfa_transition_relation_accp_sink.values()))
        print('DFA Preimage: ', dfa_preimage)
        # self.convert_cube_to_state_ADD(dfa_preimage, action=False, verbose=False)
        
        # then evolve over the game
        dfa_game_preimage = self.preimage_test(From=dfa_preimage, latches=self.latches, prime_latches=self.prime_latches, ts_action=list(self.transition_relation.values()))
        print('DFA Game Preimage: ', dfa_game_preimage)
        self.convert_cube_to_state_ADD(dfa_game_preimage, action=False, verbose=True)



class FrankaWorldArchAriaNoPrime(SymbolicPartitionedDFAGameNoPrime):
    def __init__(self,
                 boxes: int, locs: int,
                 ratio: int, init: tuple,
                 goal: tuple, formula: str,
                 restricted_human_locs: List[int],
                 restricted_human_boxes: List[int],
                 ltlf_flag: bool = True,
                 weight_factor: int = 1,
                 enable_reordering: bool = False):
        super().__init__(boxes, locs,
                         ratio, init,
                         goal, formula,
                         restricted_human_locs, restricted_human_boxes,
                         ltlf_flag, weight_factor,
                         enable_reordering=False)
        if enable_reordering:
            self.manager.autodynEnable()
    
    
    def post_process_transition_relation(self):
        super().post_process_transition_relation()
        debug: bool = False

        # now lets rmeove human moves that move blocks to the right
        mono_invalid_hmoves = self.manager.addZero()
        possible_hmoves: ADD =  reduce(lambda x, y: x | y, self.env_action_cube_list)
        kval_cube = reduce(lambda x, y: x | y, [self.kVar_map_sym[f'k{i}'] for i in range(0, self.ratio)])
        for hloc in self.human_locs:
            for hbox in self.human_boxes:
                # the box can only to its left, thats from 1 to locs
                valid_moves = [self.action_map_sym.get(f'hmove b{hbox} l{valid_loc}', self.manager.addZero()) for valid_loc in range(1, hloc + 1)]
                valid_moves_cube: ADD = reduce(lambda x, y: x | y, valid_moves) | self.action_map_sym['hmove noop'] | self.hmove_not_b[hbox]

                for tr_key, tr_dd in self.transition_relation.items():
                    if tr_key == 't0':
                        continue
                    mono_invalid_hmoves |= tr_dd & self.tVar_map_sym['human'] & kval_cube & self.xVar_map_sym[f'b{hbox} l{hloc}'] & (~valid_moves_cube & possible_hmoves)
                    self.transition_relation[tr_key] = (self.tVar_map_sym['human'] & kval_cube & self.xVar_map_sym[f'b{hbox} l{hloc}'] & (~valid_moves_cube & possible_hmoves)).ite(self.manager.addZero(), tr_dd)
        
        # brute force that we try to improve later
        for pred, pred_cube_str in self.pVar_map.items():
            care_edge = mono_invalid_hmoves & self.xVar_map_sym[pred]
            if care_edge.isZero():
                continue
            pred_clause_prime_string = ''
            if 'in-transit' in pred:
                to_box = pred.split(' ')[1] # get the box
                pred_clause_prime_string = self.xVar_map[f'to-obj {to_box}']
                print(f"Adding invalid edges From '{pred}' to 'to-obj {to_box}':")
            elif 'in-transfer' in pred:
                to_loc = pred.split(' ')[1] # get the location
                pred_clause_prime_string = self.xVar_map[f'holding {to_loc}']
                print(f"Adding invalid edges From '{pred}' to 'holding {to_loc}':")
            elif 'holding' in pred or 'ready' in pred:
                pred_clause_prime_string = pred_cube_str
                print(f"Adding invalid edges From '{pred}' to '{pred}':")
            else:
                continue

            for sidx, s in enumerate(pred_clause_prime_string):
                if s == '1':
                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= care_edge
            
            # debugging 
            # self.convert_cube_to_state_ADD(care_edge, action=True, verbose=True)

        for bidx in range(self.boxes):
            for pred, box_cube_Str in self.bVars_map[bidx].items():    
                # box remains in the same location if human does not move it/ or chooses an invalid edge
                care_edge = mono_invalid_hmoves & self.xVar_map_sym[pred]
                if care_edge.isZero():
                    continue
                for sidx, s in enumerate(self.xVar_map[pred]):
                    if s == '1':
                        self.transition_relation[self.bVars[bidx][sidx].bddPattern().__str__()] |= care_edge
                
                print(f"Adding invalid edges From '{pred}' to '{pred}':")
                # self.convert_cube_to_state_ADD(care_edge, action=True, verbose=True)

        # add the invalid edges to k0
        for sidx, s in enumerate(self.kVar_map['k0']):
            if s == '1':
                self.transition_relation[self.kVars[sidx].bddPattern().__str__()] |= mono_invalid_hmoves
        
        if debug:
            print("Valid edges:")
            self.convert_cube_to_state_ADD(mono_invalid_hmoves, action=True, verbose=True)
    
    def check_valid_human_move(self, curr_state: List[str], action: str):
        box_idx = 3
        action = super().check_valid_human_move(curr_state=curr_state, action=action)
        if action.startswith('hmove noop'):
            return 'hmove noop'
        else:
            assert action.startswith('hmove'), "Unknown human action. Cannot proceed!!"
            b_idx = action.split(' ')[1]
            hmove_to_loc = action.split(' ')[2]
            split_str = curr_state[box_idx].split(', ')
            curr_box__loc = split_str[int(b_idx[1:])].split(' ')[1]
            
            # if the current box is to the LEFT of the location that human is trying to move it to, then its an invalid move
            if int(curr_box__loc[1:]) < int(hmove_to_loc[1:]):
                return 'hmove noop'
            else:
                return action