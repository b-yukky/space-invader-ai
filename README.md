# IAT-project Space Invader

This project was realized within the scope of a school module.
We had to implement a reinforcement learning algorithm to teach an agent playing the Space Invader game.

## Setup

1. Get the source code by cloning this github locally
```bash
git clone https://github.com/b-yukky/space-invader-ai.git
cd space-invader-ai
```

2. Install the dependancies in a virtual env
```bash
python -m venv env
./env/Scripts/activate
```
```bash
pip3 install -r requirements.txt
```

You can run the best agent by running the game
```bash
python3 run_game.py
```

Best agent has been trained with weights "policy_weights_2022-07-03_13h21_best"
 (16 hours)

You can switch trained weights by changing in **run_game.py**
```
weights = torch.load("./training/policy_weights_2022-07-03_13h21_best")
```

You can start a training with
```
python3 train_dqn_agent.py
```

