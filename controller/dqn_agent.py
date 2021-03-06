import copy
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time
import datetime
import pandas as pd
import pygame

from game import SpaceInvaders
from classes import epsilon_profile
class DQNAgent():
    """ 
    Deep-Q Learning agent.
    """
    TEST_FREQUENCY = 5
    
    def __init__(self, game: SpaceInvaders, qnetwork: nn.Module, eps_profile: epsilon_profile.EpsilonProfile, gamma: float, alpha: float, replay_memory_size: int = 1000, batch_size: int = 32, target_update_freq: int = 100, tau: float = 1., final_exploration_episode : int = 500):
        
        self.env = game
        
        self.na = game.na

        self.date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%M")
        
        self.policy_net = qnetwork
        self.target_net = copy.deepcopy(qnetwork)
        
        self.best_score = 15
        
        # Learning parameters
        self.alpha = alpha
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.minibatch_size = batch_size
        self.target_update_frequency = target_update_freq
        self.tau = tau
        
        # Epsilon profile
        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial
        self.init_epsilon = self.eps_profile.initial
        self.final_epsilon = self.eps_profile.final
        self.final_exploration_episode = final_exploration_episode
    
        # Optimization criteria (Mean Squared Error Loss) -> Minimiser erreur quadratique
        self.criterion = nn.MSELoss()

        # Adam algorithm for gradient descent optimization method
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        
        self.log = {
            "episode": [],
            "test_success_ratio": [],
            "epsilon": [],
            "time": [],
            "train_score": [],
            "train_mean_steps": [],
            "test_score": [],
            "test_mean_steps": []
        }
    
    def init_replay_memory(self, env: SpaceInvaders):
        """Cette m??thode initialise le buffer d'exp??rience replay.
        :param env: Environment
        :type env: SpaceInvaders
        """
        # Replay memory pour s, a, r, terminal, and sn
        self.Ds = np.zeros([self.replay_memory_size, len(env.get_state())], dtype=np.float32)
        self.Da = np.zeros([self.replay_memory_size, env.na], dtype=np.float32)
        self.Dr = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dt = np.zeros([self.replay_memory_size], dtype=np.float32)
        self.Dsn = np.zeros([self.replay_memory_size, len(env.get_state())], dtype=np.float32)

        self.d = 0     # counter for storing in D
        self.ds = 0    # total number of steps

    def learn(self, env, n_episodes, max_steps):
        """Cette m??thode ex??cute l'algorithme de q-learning. 
        Il n'y a pas besoin de la modifier. Simplement la comprendre et faire le parall??le avec le cours.
        :param env: L'environnement 
        :type env: SpaceInvaders
        :param num_episodes: Le nombre d'??pisode
        :type num_episodes: int
        :param max_num_steps: Le nombre maximum d'??tape par ??pisode
        :type max_num_steps: int
        """
        self.init_replay_memory(env)
        self.init_log(n_episodes, max_steps)

        # Initialisation des stats d'apprentissage
        sum_rewards = np.zeros(n_episodes)
        len_episode = np.zeros(n_episodes)
        n_steps = np.zeros(n_episodes) + max_steps

        self.start_time = time.time()

        # Execute N episodes
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = env.reset()
            # Execute K steps
            for step in range(max_steps):
                # Selectionne une action
                pygame.event.get()
                action = self.select_action(state)

                # Echantillonne l'??tat suivant et la r??compense
                next_state, reward, terminal = env.step(action)

                # Stocke les donn??es d'apprentissage
                sum_rewards[episode] += reward
                len_episode[episode] += 1

                # Mets ?? jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state, terminal)

                if terminal:
                    n_steps[episode] = step + 1  # number of steps taken
                    break

                state = next_state

            self.epsilon = max(self.final_epsilon, self.epsilon - (1. / self.final_exploration_episode))
            # self.epsilon = max(self.epsilon - self.eps_profile.dec_step, self.eps_profile.final)

            # Mets ?? jour le r??seau cible, en copiant tous les weights et biases dans DQN
            if n_episodes % self.target_update_frequency == 0:
                if (self.tau < 1.):
                    # Mets ?? jour le r??seau de neurones cible en lissant ses param??tres avec ceux de policy_net
                    self.soft_update(self.tau)
                else:
                    # Copie le r??seau de neurones courant dans le r??seau cible
                    self.hard_update()

            n_ckpt = 10
            n_test_runs = 4

            if episode % DQNAgent.TEST_FREQUENCY == DQNAgent.TEST_FREQUENCY - 1:   
                test_score, test_extra_steps = self.run_tests(env, n_test_runs, max_steps)
                print('Episode: %5d/%5d, Test success ratio: %.2f, Epsilon: %.2f, Time: %.1f'
                      % (episode + 1, n_episodes, (np.sum(test_extra_steps) / n_test_runs)/(max_steps), self.epsilon, time.time() - self.start_time))
                print('train score: %.1f, mean steps: %.1f, test score: %.1f, test extra steps: %.1f'
                      % (np.mean(sum_rewards[episode-(n_ckpt-1):episode+1]), np.mean(len_episode[episode-(n_ckpt-1):episode+1]), test_score, np.mean(test_extra_steps)))

                self.log["episode"].append(episode+1)
                self.log["test_success_ratio"].append((np.sum(test_extra_steps) / n_test_runs)/(max_steps))
                self.log["epsilon"].append(self.epsilon)
                self.log["time"].append(time.time() - self.start_time)
                self.log["train_score"].append(np.mean(sum_rewards[episode-(n_ckpt-1):episode+1]))
                self.log["train_mean_steps"].append(np.mean(len_episode[episode-(n_ckpt-1):episode+1]))
                self.log["test_score"].append(test_score)
                self.log["test_mean_steps"].append(np.mean(test_extra_steps))
            
        test_score, test_extra_steps = self.run_tests(env, n_test_runs, max_steps)
        # for k in range(n_test_runs):
        #     print(test_extra_steps[k])
        print('Final test score: %.1f' % test_score)
        print('Final test success ratio: %.2f' % (np.sum(test_extra_steps) / n_test_runs))
        self.export_weight()
        self.export_log()
        
    def updateQ(self, state, action, reward, next_state, terminal):
        """ Cette m??thode utilise une transition pour mettre ?? jour la fonction de valeur Q de l'agent. 
        Une transition est d??finie comme un tuple (??tat, action r??compense, ??tat suivant).
        :param state: L'??tat origine
        :param action: L'action
        :param reward: La r??compense per??ue
        :param next_state: L'??tat suivant
        """
        # Ajoute les ??l??ments dans le buffer d'exp??rience
        #print(state, reward, next_state, terminal)
        self.Ds[self.d], self.Dr[self.d], self.Dsn[self.d], self.Dt[self.d] = state, reward, next_state, terminal

        # since Da[d,:] is a one-hot vector
        self.Da[self.d] = 0
        self.Da[self.d, action] = 1

        # since D is a circular buffer
        self.d = (self.d + 1) % self.replay_memory_size
        self.ds = self.ds + 1
 
        # Commence l'apprentissage quand le buffer est plein
        if self.ds >= self.replay_memory_size:

            self.optimizer.zero_grad()

            # S??lectionne des indices al??atoires dans le buffer
            c = np.random.choice(self.replay_memory_size, self.minibatch_size)

            # R??cup??re les batch de donn??es associ??s
            x_batch, a_batch, r_batch, y_batch, t_batch = torch.from_numpy(self.Ds[c]), torch.from_numpy(
                self.Da[c]), torch.from_numpy(self.Dr[c]),  torch.from_numpy(self.Dsn[c]), torch.from_numpy(self.Dt[c])

            # Calcul de la valeur courante 
            current_value = self.policy_net(x_batch).gather(1, a_batch.max(1).indices.unsqueeze(1)).squeeze(1)

            # Calcul de la valeur cible
            target_value = self.target_net(y_batch).max(1).values * self.gamma * (1. - t_batch) + r_batch

            # La fonction 'detach' arr??te la r??tropopagation du gradient ?? 
            # travers la partie du graphe concern??e (ici target network)
            loss = self.criterion(current_value, target_value.detach())

            loss.backward()
            self.optimizer.step()

    def select_greedy_action(self, state):
        """
        Cette m??thode retourne l'action gourmande.
        :param state: L'??tat courant
        :return: L'action gourmande
        """
        q = self.policy_net(torch.FloatTensor(state).unsqueeze(0))
        a = np.random.choice(np.where(q[0] == q[0].max())[0])
        #print(q)
        return a
    
    def select_action(self, state : 'Tuple[int, int]'):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.na)      # random action
        else:
            return self.select_greedy_action(state)

    def hard_update(self):
        """ Cette fonction copie le r??seau de neurones 
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def init_log(self, n_episodes, max_steps):
        with open(f'./training/params_{self.date}.txt', "w") as f:
            f.write(f"N episodes: {n_episodes}\n")
            f.write(f"Max steps: {max_steps}\n")
            f.write(f"Alpha: {self.alpha}\n")
            f.write(f"Gamma: {self.gamma}\n")
            f.write(f"Replay memory size: {self.replay_memory_size}\n")
            f.write(f"Tau: {self.tau}\n")

    def export_log(self):
        df = pd.DataFrame(self.log)
        df.to_csv(f'./training/logs_{self.date}', sep=',', encoding='utf-8')
    
    def export_weight(self):
        try:
            print(self.policy_net.state_dict())
            trained_time = str(datetime.timedelta(seconds=(time.time() - self.start_time)))
            torch.save(self.policy_net.state_dict(), f"./training/policy_weights_{self.date}")
            with open(f'./training/params_{self.date}.txt', "a") as f:
                f.write(f"Training time: {trained_time}")
        except Exception as e:
            print("Error %s" %e)
    
    def soft_update(self, tau):
        """ Cette fonction fait mise ?? jour glissante du r??seau de neurones cible 
        """
        for target_param, local_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0 - tau) * target_param.data)

    def run_tests(self, env, n_runs, max_steps):
        test_score = 0.
        extra_steps = np.zeros((n_runs))
        for k in range(n_runs):
            s = env.reset()
            print("Run %s" %str(k+1))
            for t in range(max_steps):
                pygame.event.get()
                q = self.policy_net(torch.FloatTensor(s).unsqueeze(0))
                # greedy action with random tie break
                a = np.random.choice(np.where(q[0] == q[0].max())[0])
                sn, r, terminal = env.step(a)
                #print(f"state {sn}, action {a}, total_score {test_score}")
                test_score += r
                if terminal:
                    break
                s = sn
            extra_steps[k] = t
        if test_score > self.best_score:
            self.best_score = test_score
            torch.save(self.policy_net.state_dict(), f"./training/policy_weights_{self.date}_best")
        print("Avg score : %s" %(test_score/n_runs))
        return test_score / n_runs, extra_steps
