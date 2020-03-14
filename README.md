# Control Freak
A solution for solving Classic Control problems by combining Neural Network and Evolution Strategies.<br/>
While these tasks are usually solved by Reinforcement Learning algorithms, here we solve them using Evolution Strategy.
Created a policy neural network for choosing the next action to play.
The weights of the neural network are trained using Evolution strategies.

## Classic Control Problems
Control theory problems from the classic Reinforcement Learning literature.<br/>
The problems are simulated using a framework built by OpenAI called [Gym](http://gym.openai.com/).<br/>
For each task, the environment provides an initial state from a distribution (so each game might be a little different), accepts actions, and given an action provides the next state and a reward.
Each states and actions simulates real physics behavior.<br/>
Example of states values are location, velocity, angle of joint, angular velocity and momentum, actions are usually the force that should be acted on an object.<br/>
The goal is to build a policy model that given a state returns the action that will lead us to the highest reward.<br/>

***
## Tasks definitions
We solved 3 different tasks using the same training code.

### Cartpole 
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart.<br/>
The pendulum starts upright, and the goal is to prevent it from falling over.<br/>
A reward of +1 is provided for every timestep that the pole remains upright.<br/>
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.<br/>
 
### Acrobot
Acrobot is a 2-link pendulum with only the second joint actuated. Initially, both links point downwards. The goal is to swing the end-effector at a height at least the length of one link above the base. Both links can swing freely and can pass by each other, i.e., they don't collide when they have the same angle.<br/>
 
### MountainCar
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.<br/>
 
***
## Results
The goal in all the tasks is to get the highest score<br/>

**1. Cartpole-v0**<br/>
Solved Requirements - average reward of 195<br/>
Our Score - average reward of 200 <br/>
Solved!<br/>


**2. Acrobot-v1**<br/>
Solving – no pecified threshold, only leaderboard [here](https://github.com/openai/gym/wiki/Leaderboard#acrobot-v1)<br/>
Our Score - average reward of -80.73<br/>
10th place in the leaderboard!<br/>


**3. MountainCar-v0**<br/>
Solving  - average reward of -110.0<br/>
Our Score - average reward of -106.26<br/>
Solved!<br/>

