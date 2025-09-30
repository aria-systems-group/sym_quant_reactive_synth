import math

from typing import List, Tuple

from cudd import Cudd, ADD, BDD

class FrankaWorld():

    def __init__(self, boxes: int, locs: int):
        self.boxes: int = boxes
        self.locs: int = locs
        self.misc_preds = ['ready', 'to-obj', 'holding']
        self.robot_actions: List[str] = ['transit', 'transfer', 'grasp', 'release']
        self.manager: Cudd = Cudd()

        self.pVars, self.bVars, self.lVars = self.create_latches()
        self.xVars = self.pVars + self.bVars + self.lVars
        self.oVars = self.create_output_vars()
        self.latches = self.xVars # in future we will primed version of these as well.

        self.xVar_map = dict()
        self.rAction_map = dict()

        self.create_xVar_map()
        self.create_rAction_map()

        self.transition_relation = {var.bddPattern().__str__(): self.manager.addZero() for var in self.latches}


    def create_latches(self) -> Tuple[List[ADD], List[ADD], List[ADD]]:
        """
         For ready; to-obj; and holding we create a dedicated set of latches
         For every on b predicate we will create a set of latch for all boxes and for all locs. 
        """
        box_var_size = math.ceil(math.log2(self.boxes))
        # create an additional boolean var to skip the 0-vector latch 
        box_var_size = box_var_size + 1 if pow(2, box_var_size) == self.boxes else box_var_size 
        
        loc_var_size = math.ceil(math.log2(self.locs + 1))  # the +1 is for end-effector (ee) location
        # create an additional boolean var to skip the 0-vector latch
        loc_var_size = loc_var_size + 1 if pow(2, loc_var_size) == self.locs else loc_var_size 
        
        # create holding, ready, to-obj vars
        pVars = self.create_ready_holding_to_obj_vars()
        
        # create dedicated vars for boxes
        varsize = self.manager.size()
        bVars: List[ADD] = [self.manager.addVar(i + varsize, 'b' + str(i)) for i in range(box_var_size)]
        
        varsize = self.manager.size()
        lVars: List[ADD] = [self.manager.addVar(j + varsize, 'l' + str(j)) for j in range(loc_var_size)]

        return pVars, bVars, lVars

    def create_ready_holding_to_obj_vars(self) -> List[ADD]:
        varsize = self.manager.size()
        vars_size: int = math.ceil(math.log2(3))
        Vars: List[ADD] = [self.manager.addVar(k + varsize, 'p' + str(k)) for k in range(vars_size)]
        return Vars
    

    def create_xVar_map(self):
        # for misc preds
        for pidx, pred in enumerate(self.misc_preds):
            bit_str = f"{pidx + 1:0{len(self.pVars)}b}"
            self.xVar_map[pred] = bit_str
        
        # for boxes 
        for b in range(self.boxes):
            bit_str = f"{b + 1:0{len(self.bVars)}b}"
            self.xVar_map['b' + str(b)] = bit_str
        
        # for locs; +1 is for end-effector (ee) location ; l0 is reserved for end effector
        for l in range(self.locs + 1):
            bit_str = f"{l + 1:0{len(self.lVars)}b}"
            self.xVar_map['l' + str(l)] = bit_str


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
        for ridx, ract in enumerate(self.robot_actions):
            if ract == 'transit':
                for b in range(self.boxes):
                    act_str = ract + str(b)
                    rbit_str = f"{ridx + b + 1:0{len(self.oVars)}b}"
                    self.rAction_map[act_str] = rbit_str
            elif ract == 'transfer':
                for l in range(self.locs):
                    act_str = ract + str(l)
                    rbit_str = f"{ridx + self.boxes + l + 1:0{len(self.oVars)}b}"
                    self.rAction_map[act_str] = rbit_str
            else:
                rbit_str = f"{ridx + self.boxes + self.locs + 1:0{len(self.oVars)}b}"
                self.rAction_map[ract] = rbit_str
    
    
    def cube_to_add(self, cube: str, vars_list: List) -> ADD:
        assert len(cube) == len(vars_list), "Make sure the length of the cube is the same as the number of latches"
        add = self.manager.addOne()
        for idx, val in enumerate(cube):
            add &= vars_list[idx] if val == '1' else ~vars_list[idx]
        return add
    
    def create_transition_relation(self) -> None:
        # we start with creating transitions for transit
        for b in range(self.boxes):
            # current state clauses
            pred_clause = self.cube_to_add(self.xVar_map['ready'], self.pVars)
            box_clause = self.cube_to_add(self.xVar_map['b' + str(b)], self.bVars)
            act_clause = self.cube_to_add(self.rAction_map['transit' + str(b)], self.oVars)

            # next state clauses - (to-obj b0) & (on-b0-<some-loc>)
            pred_clause_prime_string = self.xVar_map['to-obj']
            box_clause_prime_string = self.xVar_map['b' + str(b)]

            for sidx, s in enumerate(pred_clause_prime_string):
                if s == '1':
                    self.transition_relation[self.pVars[sidx].bddPattern().__str__()] |= pred_clause & box_clause & act_clause
            
            for sidx, s in enumerate(box_clause_prime_string):
                if s == '1':
                    self.transition_relation[self.bVars[sidx].bddPattern().__str__()] |= pred_clause & box_clause & act_clause
    



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
    # boxes = 1
    # locs = 2
    # fw = FrankaWorld(boxes, locs)

    # print('Hi Mom!')
    # print('xVars map:')
    # for k, v in fw.xVar_map.items():
    #     print(f"{k} : {v}")
    
    # print('rAction map:')
    # for k, v in fw.rAction_map.items():
    #     print(f"{k} : {v}")

    simple_franka_world()