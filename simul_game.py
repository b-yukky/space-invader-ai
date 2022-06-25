from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.random_agent import RandomAgent

def format_state(state):
    player_x = round(state[0][0])
    player_y = round(state[0][1])
    invaders_x = [round(x) for x in state[1][0]]
    invaders_y = [round(y) for y in state[1][1]]
    bullet_x, bullet_y = round(state[2][0]), round(state[2][1])
    print(f'Player: [{player_x},{player_y}] | Inv: [{invaders_x},{invaders_y}] | Bullet: {state[3]} [{bullet_x}, {bullet_y}]')

def main():

    game = SpaceInvaders(display=False)
    #controller = KeyboardController()
    controller = RandomAgent(game.na)
 
    state = game.reset()
    while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)
        format_state(state)

if __name__ == '__main__' :
    main()
