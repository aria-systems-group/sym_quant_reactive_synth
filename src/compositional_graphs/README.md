# ABOUT

This folder contains code for the following: Compositional Construction and Pure ADD based Strategy Synthesis:

## Compositional Construction

1. Construct Transition System (T) compositionally - FrankaWorld
2. Construct Game (G) compositionally - Gridworld and FrankaDynamic\*

#### Frankaworld

This script implements a simple Single Manipulator robot operating *without* human in the workspace. The atomic actions for the robot are: 

1. `transit` - Move to box while the end-effector is empty.
2. `grasp` - Grasp a box if you are at the box's location.
3. `transfer` - Move a box to a another location.
4. `release` - Release the box that is at end-effector.

#### FrankaworldDynamic - Concurrent Game 

This script inherits the Frankaworld() class and augments the class with human moves (`hmove b# b#`). This script implements a Single Manipulator robot (Sys player) operating with **one** human (Env player) in the workspace (*hence dynamic*). The atomic actions for both the players are: 

Sys player:

1. `transit` - Move to box while the end-effector is empty.
2. `grasp` - Grasp a box if you are at the box's location.
3. `transfer` - Move a box to a another location.
4. `release` - Release the box that is at end-effector location.

Env player:

1. `hmove b# l#` - Move box `b#` to `l#`
2. `hmove noop` - No Human Move

> NOTE 1: FrankaworldDynamic code is a work in progress. This script *tries* to implement a concurrent game (same as our IROS 23; ICRA 22 game) where from a give state, the robot **commits** to an action (say `transit b0`) and the human chooses to either move or not not-move. Thus, after both the player choose their respective actions, the game evolves to the next Sys player state. This script tries to construct the game compositionally. However, the evolution of the game under transit action for robot and human is incorrect.


#### FrankaworldDynamic_tb - Turn-Based Game

This script implements a Manipulator robot (Sys player) and a human (Env player) operating in a shared workspace as a **pure turn-based (hence tb) game**.  Here Sys player moves first, then the Env player moves. The game start in Sys player state where the Sys **choose** an action. The game immediately evolves to the Env player state and stays there until (i) human moves a box (`hmove b# l#`) or (ii) robot finishes its execution (either `transit l#` or `transfer l#`). If human moves an object then the game evolves to robot state and the robot `rConf` is updated to its original status (as if the robot is still in its original status).

This abstraction is very permissive in the sense that, after every robot moves, we evolve to a human state. If the human chooses to move something, we evolve to a robot state and let it re-evaluate its choices. Thus, this abstraction allow for reactive behaviors as the robot does not *commit* to its action and can chose another action.

**The problem: With great expressiveness comes great complexity.** The permissive nature of the game make the existence of a winning strategy for robot impossible. Intuitively, say if the robot commits to action `transit b0`, the human could possibly move either `b0` (or any other box `b1`) to another location and hence ensure that the robot will never reach its intended destination state in the game. In real world, this does not make sense as the robot will eventually finish its action but as the game updates for every human move, the robot could theoretically be stuck in a loop in the game. **To mitigate this we introduce bounded human moves per robot move**. 

Abstraction Details:

Each state in the game tuples consists of: `(Player Token; State Predicates)` and is represented symbolically as follows:

1. `Player Token`: player token represented by `tVar` (`t`). `t` being high (`1`) corresponds to Sys player and low (`0`) corresponds Env player, respectively. 
2. `State Predicate`: Can be further categorized as `(rConf, bConf)` where `rConf` is short for robot configuration and `bConf` is short for box configuration.
	2.1 `rConf`: The set of valid robot conf. are `ready l#`, `to-obj b#`, and `holding l#`. `rConf` is represented using `pVars`  (`p`) 
	2.2 `rConf`: Another set of valid robot conf. is used to represent robot's status of executing an action. We assume that `grasp` and `release` are instantaneous and thus under these actions the robot does **not** evolve to an intermediate state. We introduce `in-transit l# b#` to denote robot executing `transit b#` action from `ready l#` conf. Similarly, we introduce `in-transfer from_l# to_l#` to denote robot executing `transfer to_l#` from `holding from_l#` conf. 
	2.3 `bConf`: The set of valid box conf. are `b# l#`, interpreted as `b#` at `l#`. *NOTE: we reserve l0 for end-effector location and |l| + 1 to be the else location*
	2.3.1 We create a dedicated set of boolean variables (`bVars`) for each box as this encoding is more precise, i.e., `b0 l#` is combination of `b0` boolean variables and `b1 l#` is a combination of `b1` boolean variables.
3. Actions: We use `iVars` (`i`) to denote human action and `oVars` (`o`) to denote robot actions, respectively. Intuition: we consider the game as a boolean circuit where the input values (value of `i`s) are chosen by the Env player and the circuit needs to ensure that output satisfies some specification by choosing values of `o`  accordingly.

#### FrankaworldDynamic_ratio_tb - K Human Moves per Robot move

This script implements a Manipulator robot (Sys player) and a human (Env player) operating in a shared workspace as a **pure turn-based (hence tb) game** where for every robot move there are **at-most K human moves**. Intuitively, this ratio of K human move for every (1) robot move corresponds to the fact that robot moves are (finite) durative in nature and that after finite human moves the robot will eventually reach its intended destination. Note that, after every human move the robot's conf. is updated to be the origial robot's conf.

This abstraction is still somewhat permissive in the sense that, after every robot moves, we evolve to a human state. Additionally, there is variable that keeps track of human move. After k consecutive human moves, human can not intervene any more. If the human chooses not to intervene then the variable is immediately set of zero in the next game state.

Abstraction Details:

Each state in the game tuples consists of: `(Player Token; Hmoves Token; State Predicates)` and is represented symbolically as follows:

1. `Player Token`: Same as above
2. `State Predicates` Same as above
2. `Hmoves Token`: This token represents how many times did the human intervene consecutively. We represent this using `kVars` (`k`) where `k0` corresponds to "human has not moved" yet and `km` corresponds to "human has moved m" times.


#### FrankaworldDynamic_ratio_else_tb - Robot in intermediate state when human moves

This script inherits from the base class FrankaWorldDyanmicRatioTurnBased() override the game update rules as follows: After every human move, the robot transit to else conf. rather that the original conf. This capture the robot in motion status more accurately and further makes the game encoding for compact. 

The Abstraction details are same above. Only the game semantics have changed.


## Pure ADD based Strategy Synthesis

1. `test_gridworld.py` - implements code for constructing a gridworld (with only one)
