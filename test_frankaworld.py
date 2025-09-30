import sys
import math

from functools import reduce
from itertools import product
from typing import List, Tuple

from bidict import bidict
from cudd import Cudd, ADD, BDD

class FrankaWorld():

    def __init__(self, boxes: int, locs: int):
        self.boxes: int = boxes
        self.locs: int = locs
        self.misc_preds = ['ready', 'to-obj', 'holding']
        self.robot_actions: List[str] = ['transit', 'transfer', 'grasp', 'release']
        self.manager: Cudd = Cudd()

        self.pVars, self.bVars = self.create_latches()
        self.xVars: List[ADD] = self.pVars + [var for box_adds in self.bVars for var in box_adds] 
        self.oVars: List[ADD] = self.create_output_vars()
        self.latches: List[ADD] = self.xVars # in future we will primed version of these as well.
        self.latches_bdd: List[BDD] = [var.bddPattern() for var in self.latches]

        # book keeping
        self.holding_preds = self.to_obj_preds = self.ready_preds = set({})

        self.xVar_map = dict()
        self.rAction_map = dict()
        
        # maps needs for lookup of the states corresponding to cubes
        self.pVar_map = bidict({})
        self.bVars_map = {b: bidict({}) for b in range(self.boxes)}

        self.create_xVar_map()
        self.create_rAction_map()

        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}

        # need these cubes for printing states from cubes
        self.pVars_cube: ADD = reduce(lambda a, b: a & b, self.pVars)
        self.bVars_cubes: List[List[ADD]] = [reduce(lambda a, b: a & b, box_adds) for box_adds in self.bVars]
        self.all_bVars_cube: ADD = reduce(lambda a, b: a & b, self.bVars_cubes)

    def create_latches(self) -> Tuple[List[ADD], List[ADD], List[ADD]]:
        """
         For ready; to-obj; and holding we create a dedicated set of latches
         For every on b predicate we will create a set of latch for all boxes and for all locs. 
        """
        # create holding, ready, to-obj vars
        pVars = self.create_ready_holding_to_obj_vars()
        
        loc_var_size = math.ceil(math.log2(self.locs + 1))  # the +1 is for end-effector (ee) location
        # create an additional boolean var to skip the 0-vector latch
        loc_var_size = loc_var_size + 1 if pow(2, loc_var_size) == self.locs + 1 else loc_var_size 
        
        bVars: List[List[ADD]] = []
        for b in range(self.boxes):
            varsize = self.manager.size()
            bVars.append([self.manager.addVar(j + varsize, f'b{b}{j}') for j in range(loc_var_size)])

        return pVars, bVars

    def create_ready_holding_to_obj_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        # num. of preds = ready x |locs| + to-obj x |boxes| + holding x |locs| + 1 (to account for l0 being end effector loc) + grasp + release
        num_of_preds = 2*self.locs + self.boxes + 2
        vars_size: int = math.ceil(math.log2(num_of_preds))
        Vars: List[ADD] = [self.manager.addVar(k + varsize, 'p' + str(k)) for k in range(vars_size)]
        return Vars
    

    def create_xVar_map(self):
        # for misc preds ready and holding we create all locs.
        offset = 0
        for pidx, pred in enumerate(['ready', 'holding']):
            # +1 to include end-effector location
            for loc in range(1, self.locs + 1):
                bit_str = f"{offset + loc:0{len(self.pVars)}b}"
                self.xVar_map[pred + ' l' + str(loc)] = bit_str
                self.ready_preds.add('ready l' + str(loc)) if pred == 'ready' else self.holding_preds.add('holding l' + str(loc))
                self.pVar_map[pred + ' l' + str(loc)] = bit_str
            offset = self.locs
        
        offset = 2*(self.locs) + 1 # 1 for the offset from the 0-vector
        # for misc pred to-obj we create all boxes
        for b in range(self.boxes):
            bit_str = f"{b + offset:0{len(self.pVars)}b}"
            self.xVar_map['to-obj b' + str(b)] = bit_str
            self.to_obj_preds.add('to-obj b' + str(b))
            self.pVar_map['to-obj b' + str(b)] = bit_str

        # for each boxes we create |locs| boolean vars
        for b in range(self.boxes):
            for l in range(self.locs + 1):
                bit_str = f"{l + 1:0{len(self.bVars[b])}b}"
                self.xVar_map['b' + str(b) + ' l' + str(l)] = bit_str
                self.bVars_map[b]['b' + str(b) + ' l' + str(l)] = bit_str

    def create_output_vars(self) -> List[ADD]:
        """
         Num of robot actions = transit x |boxes| +  transfer x |locs| + grasp + release
        """
        varsize = self.manager.size()
        num_of_rActions = self.boxes + self.locs + 2
        oVars_size = math.ceil(math.log2(num_of_rActions))
        oVars: List[ADD] =  [self.manager.addVar(r + varsize , 'o' + str(r)) for r in range(oVars_size)]
        return oVars


    def create_rAction_map(self):
        for ract in self.robot_actions:
            if ract == 'transit':
                for b in range(self.boxes):
                    act_str = f'{ract} b{b}'
                    rbit_str = f"{b + 1:0{len(self.oVars)}b}"
                    self.rAction_map[act_str] = rbit_str
            elif ract == 'transfer':
                for l in range(1, self.locs + 1):
                    act_str = f'{ract} l{l}'
                    rbit_str = f"{self.boxes + l:0{len(self.oVars)}b}"
                    self.rAction_map[act_str] = rbit_str
            # else:
        rbit_str = f"{self.boxes + self.locs + 1:0{len(self.oVars)}b}"
        self.rAction_map['grasp'] = rbit_str
        rbit_str = f"{self.boxes + self.locs + 2:0{len(self.oVars)}b}"
        self.rAction_map['release'] = rbit_str
    
    
    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add

    def create_only_b_at_ee_cube(self, curr_box: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.cube_to_add(self.xVar_map['b' + str(ob) + ' l0'], self.bVars[ob])
        return bConf_cube
    

    def create_only_b_at_l_cube(self, curr_box: int, curr_loc: int, bConf_cube: ADD) -> ADD:
        # need to add that other boxes are not at end-effector location
        for ob in range(self.boxes):
            if ob == curr_box:
                continue
            bConf_cube &= ~self.cube_to_add(self.xVar_map['b' + str(ob) + f' {curr_loc}'], self.bVars[ob])
        return bConf_cube
    

    def create_ee_empty_cube(self) -> ADD:
        # cube that implies that end-effector location empty
        bConf_cube = self.manager.addOne()
        for b in range(self.boxes):
            bConf_cube &= ~self.cube_to_add(self.xVar_map['b' + str(b) + ' l0'], self.bVars[b])
        return bConf_cube
    

        
    
    def create_transition_relation(self) -> None:
        # transit actions
        ee_empty_cube: ADD = self.create_ee_empty_cube()
        for b in range(self.boxes):
            act_str = f'transit b{b}'
            # for rConf, rCube_str in self.xVar_map.items():
            #     if not rConf.startswith('ready'):
            #         continue 
            
            # as l0 is reserved for end-effector location
            for l in range(1, self.locs + 1):
                rConf_cube = self.cube_to_add(self.xVar_map[f'ready l{l}'], self.pVars)
                box_pred: str = 'b' + str(b) + ' l' + str(l)
                bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b])
                act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                # need to enforce that the end-effector is empty
                state_constraint_cube = ee_empty_cube

                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(l), bConf_cube=bConf_cube)

                # next state clauses - (to-obj b0); box location does not change
                pred_clause_prime_string = self.xVar_map['to-obj b' + str(b)]
                box_clause_prime_string = self.xVar_map[box_pred]
                
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & state_constraint_cube & act_cube
        
        # grasp action
        for b in range(self.boxes):
            act_str = 'grasp'
            rConf_cube = self.cube_to_add(self.xVar_map[f'to-obj b{b}'], self.pVars)
            
            # as l0 is reserved for end-effector location
            for l in range(1, self.locs + 1):
                box_pred = 'b' + str(b) + ' l' + str(l)
                bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b])
                act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                # need to enforce that the end-effector is empty
                state_constraint_cube = ee_empty_cube
                # need to enforce that only one box is at loc l
                bConf_cube = self.create_only_b_at_l_cube(curr_box=b, curr_loc='l' + str(l), bConf_cube=bConf_cube)

                # next state clauses - (holding l) ; box location does not change
                box_pred_prime = 'b' + str(b) + ' l0' # box is now at end-effector location
                pred_clause_prime_string = self.xVar_map['holding l' + str(l)]
                box_clause_prime_string = self.xVar_map[box_pred_prime]
                
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & state_constraint_cube & bConf_cube & act_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & state_constraint_cube & bConf_cube & act_cube
        
        # release action
        act_str = 'release'
        for rConf, rCube_str in self.xVar_map.items():
            if not rConf.startswith('holding'):
                continue
            loc = rConf.split(' ')[1]
            
            rConf_cube = self.cube_to_add(rCube_str, self.pVars)
            
            for b in range(self.boxes):
                box_pred = 'b' + str(b) + ' l0'
                bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b]) # box is at end-effector location
                act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                bConf_cube = self.create_only_b_at_ee_cube(b, bConf_cube)

                # next state clauses - (ready l) ; box location does not change
                box_pred_prime = 'b' + str(b) + f' {loc}' # box is now at location loc
                pred_clause_prime_string = self.xVar_map['ready ' + loc]
                box_clause_prime_string = self.xVar_map[box_pred_prime]
                # need to enforce that the end-effector is empty - we enforce it from transit action, so maybe we don;t need it here. Check this!!!
                
                for sidx, s in enumerate(pred_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
                
                for sidx, s in enumerate(box_clause_prime_string):
                    if s == '1':
                        self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube

        # transfer actions
        for b in range(self.boxes):
            act_str = 'transfer'
            # for all holding confs.
            for rConf_from, rCube_from_str in self.xVar_map.items():
                if not rConf_from.startswith('holding'):
                    continue
                from_loc = rConf_from.split(' ')[1]

                for rConf_to, rCube_to_str in self.xVar_map.items():
                    if not rConf_to.startswith('holding') or rConf_to == rConf_from:
                        continue
                    to_loc = rConf_to.split(' ')[1]

                    act_str = f'transfer {to_loc}'
                    box_pred = f'b{b} l0'
                    rConf_cube = self.cube_to_add(rCube_from_str, self.pVars)
                    bConf_cube = self.cube_to_add(self.xVar_map[box_pred], self.bVars[b])  # box at current location from_loc
                    act_cube = self.cube_to_add(self.rAction_map[act_str], self.oVars)
                    bConf_cube = self.create_only_b_at_ee_cube(b, bConf_cube)

                    # next state clauses - (holding to_loc) ; box location does not change
                    assert rConf_to == 'holding ' + to_loc, "Make sure the to_loc is correct. Fix this!!!"
                    pred_clause_prime_string = self.xVar_map['holding ' + to_loc]
                    box_clause_prime_string = self.xVar_map[box_pred] 

                    for sidx, s in enumerate(pred_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
                
                    for sidx, s in enumerate(box_clause_prime_string):
                        if s == '1':
                            self.transition_relation[self.bVars[b][sidx].bddPattern().__str__()] |= rConf_cube & bConf_cube & act_cube
    
    def convert_cube_to_state(self, dd: BDD) -> None:
        """
         Convert a cube to a state representation
        """
        cubes = []
        for c in dd.generate_cubes():
            _amb_var = []
            var_list = []
            for _idx, var in enumerate(c):
                if self.manager.addVar(_idx) not in self.latches:
                    continue

                if var == 2:
                    _amb_var.append([self.manager.addVar(_idx), ~self.manager.addVar(_idx)])
                elif var == 0:
                    var_list.append(~self.manager.addVar(_idx))
                elif var == 1:
                    var_list.append(self.manager.addVar(_idx))
                else:
                    print("CUDD ERRROR, A variable is assigned an unaccounted integret assignment. FIX THIS!!")
                    sys.exit(-1)
            
            # check if it is not full defined
            if len(_amb_var) != 0:
                cart_prod = list(product(*_amb_var))
                for _ele in cart_prod:
                    var_list.extend(_ele)
                    cubes.append(reduce(lambda a, b: a & b, var_list))
                    var_list = list(set(var_list) - set(_ele))
            else:
                cubes.append(reduce(lambda a, b: a & b, var_list))
        

        # print the states
        for cube in cubes:
            rConf_cube_str = cube.existAbstract(self.all_bVars_cube).bddPattern().cubeString().replace('-', '')
            bCube_str = []
            for b in range(self.boxes):
                all_but_b_cube = reduce(lambda a, b: a & b, self.bVars_cubes[:b] + self.bVars_cubes[b+1:])
                bCube_str.append(cube.existAbstract(all_but_b_cube & self.pVars_cube).bddPattern().cubeString().replace('-', ''))
            
            # you could have invalid states as well. We ksip over such cubes
            invalid_state = False
            for bidx, e in enumerate(bCube_str):
                if e not in self.bVars_map[bidx].inv:
                    invalid_state = True
                    break
            if invalid_state:
                continue
            box_states = ", ".join(self.bVars_map[bidx].inv[e] for bidx, e in enumerate(bCube_str))
            print(f"({self.pVar_map.inv[rConf_cube_str]}, {box_states})")
            
    

    def test_pre_image(self):
        # convert transition relation to latches bdd

        tr_bdd = [dd.bddPattern() for dd in self.transition_relation.values()]

        # goal state is b0 and l0 and ready l0
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l1'], self.pVars) 
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l0'], self.bVars[0]) & self.cube_to_add(self.xVar_map['holding l1'], self.pVars) 
        # goal_cube = self.cube_to_add(self.xVar_map['holding l1'], self.pVars)
        # goal_cube = self.cube_to_add(self.xVar_map['b0 l1'], self.bVars[0]) & self.cube_to_add(self.xVar_map['to-obj b0'], self.pVars)
        goal_cube = self.cube_to_add(self.xVar_map['b0 l2'], self.bVars[0]) & self.cube_to_add(self.xVar_map['ready l2'], self.pVars)

        From = goal_cube.bddPattern()

        pre = From.vectorCompose(self.latches_bdd, tr_bdd)

        print(pre)
        self.convert_cube_to_state(pre)
    

    



def simple_franka_world():
    """
     Build TR manually and test things out.
    """
    
    def add_to_tr(tr: dict, cube: ADD, prime_cube_str: str, var_list: List[ADD]):
        for sidx, s in enumerate(prime_cube_str):
            if s == '1':
                tr[var_list[sidx].bddPattern().__str__()] |= cube
    
    m = Cudd()
    p0, p1 = m.addVar(0, 'p0'), m.addVar(1, 'p1')
    b0 = m.addVar(2, 'b0')
    l0, l1 = m.addVar(3, 'l0'), m.addVar(4, 'l1')
    o0, o1, o2 = m.addVar(5, 'o0'), m.addVar(6, 'o1'), m.addVar(7, 'o2')

    # ready - 01; to-obj - 10; holding - 11 and so on for boxes and locs. Note l0 is reserved for end-effector
    xVar_map = {'ready': '01', 'to-obj': '10', 'holding': '11', 'b0': '1', 'else': '01', 'l0': '10', 'l1': '11'}
    xVar_map_sym = {'ready': ~p0 & p1, 'to-obj': p0 & ~p1, 'holding': p0 & p1, 'b0': b0, 'else': ~l0 & l1, 'l0': l0 & ~l1, 'l1': l0 & l1}

    # robot action map - you can not transfer to end-effector location.
    rAction_map = {'transit0': '000', 'transfer1': '001', 'tranfer2': '010', 'grasp': '011' , 'release': '100'}
    rAction_map_sym = {'transit0': ~o0 & ~o1 & ~o2, 'transfer1': ~o0 & ~o1 & o2, 'tranfer2': ~o0 & o1 & ~o2, 'grasp': ~o0 & o1 & o2 , 'release': o0 & ~o1 & ~o2}

    # build smple transition relation
    transition_relation = {var.bddPattern().__str__(): m.addZero() for var in [p0, p1, b0, l0, l1]}

    # transit0 - (ready else) (on b0 l0) -> (to-obj b0) (on b0 l0) | (ready l0) (on b0 l0))
    pred_clause = xVar_map_sym['ready']
    box_clause = xVar_map_sym['b0']
    else_clause = xVar_map_sym['else']
    loc_clause = xVar_map_sym['l0']
    act_clause = rAction_map_sym['transit0']
    
    # next state clauses - (to-obj b0) & (on-b0-l0) | (ready l0) (on b0 l0))
    pred_clause_prime_string = xVar_map['to-obj']
    box_clause_prime_string = xVar_map['b0']
    loc_clause_prime_string = xVar_map['l0']

    add_to_tr(tr=transition_relation,
              cube=pred_clause & box_clause & else_clause & loc_clause & act_clause,
              prime_cube_str=pred_clause_prime_string + box_clause_prime_string + loc_clause_prime_string,
              var_list=[p0, p1, b0, l0, l1])

    pred_clause_prime_string = xVar_map['ready']
    add_to_tr(tr=transition_relation,
              cube=pred_clause & box_clause & else_clause & loc_clause & act_clause,
              prime_cube_str=pred_clause_prime_string + box_clause_prime_string + loc_clause_prime_string,
              var_list=[p0, p1, b0, l0, l1])


            




if __name__ == "__main__":
    boxes = 5
    locs = 20
    fw = FrankaWorld(boxes, locs)

    print('xVars map:')
    for k, v in fw.xVar_map.items():
        print(f"{k} : {v}")
    
    print('rAction map:')
    for k, v in fw.rAction_map.items():
        print(f"{k} : {v}")

    # simple_franka_world()
    fw.create_transition_relation()
    fw.test_pre_image()

    # print('Transition Relation:')
    # for k, v in fw.transition_relation.items():
    #     print(f"{k} : {v}")