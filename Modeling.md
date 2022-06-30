## Conception


### Learning algorithm
- Infinite amount of states with finite amount of actions -> Deep Q Learning


## State 
Relevant environment information for the agent

__some ideas__
- Ship position (only X since it can't move on Y)
- Invaders position (as an array, or only the closest)
- Distance between ship and closest invader
- Angle between ship and closest invader
- Bullet state
- //vector (speed+direction) of the closes invader
- //distance from border limits
- speed

__todo__

- Implement the best state representation (can try multiple ideas)
- Implement the best learning algorithm (DeepQL,Q-Learning, maybe others?) -> make it work
- Hypertuning of the learning algorithm (probably DeepQL, change the neural networks etc.) -> optimisation for best performances
- Project architecture for efficient testing
- Draw graphs to show performance

- Switch from Pytorch  -> TensorFlow (for deep learning)


**DeepQ-L**
- What to feed to our neural networks ? (how to represent states in a smart and efficient way)
- Model the buffer experience replay
- Model the neural networks (inputs, layers and outputs)
