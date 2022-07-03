import torch
from torch import nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, n_inputs: int, na: int):
        """À MODIFIER QUAND NÉCESSAIRE.
        Ce constructeur crée une instance de réseau de neurones de type Multi Layer Perceptron (MLP).
        L'architecture choisie doit être choisie de façon à capter toute la complexité du problème
        sans pour autant devenir intraitable (trop de paramètres d'apprentissages). 

        :param na: Le nombre d'actions 
        :type na: int
        """
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 18),
            nn.ReLU(),
            nn.Linear(18, 10),
            nn.ReLU(),
            nn.Linear(10, na),
        )

    def forward(self, x):
        """Cette fonction réalise un passage dans le réseau de neurones. 

        :param x: L'état
        :return: Le vecteur de valeurs d'actions (une valeur par action)
        """
        x = self.flatten(x)
        qvalues = self.layers(x)
        return qvalues

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
