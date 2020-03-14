import gym
import numpy as np
import TrainEs
import sys
import os
from Fitness import Fitness

render = False


class GameFitness(Fitness):
    """
    A generic class for fitness evaluation in gym environment.
    Calculates the fitness for an individual.
    The fitness is calculated by creating a NN, initiating its weights from the individual values, and then running an episode of the game using the NN.
    The running process:
    a.	It receives a random state from the environment
    b.	Selects an action by feeding it to the NN
    c.	Receives a new state and a reward from the environment
    d.	Repeats a-c until the end of the episode.
    The fitness is the sum of the rewards in an episode (or the average sum of rewards if running several episodes).
    """
    def __init__(self, game_name, num_layers, layers_output_sizes_but_last, episodes_per_fitness=1):
        np.random.seed(123)
        self._game_name = game_name
        self.env = gym.make(game_name)
        self.episodes_per_fitness = episodes_per_fitness

        # Define Network
        self.inputSize = self.env.observation_space.shape[0] if type(self.env.observation_space) == gym.spaces.box.Box \
            else self.env.observation_space.n
        self.action_size = self.env.action_space.shape[0] if type(self.env.action_space) == gym.spaces.box.Box \
            else self.env.action_space.n
        self.num_layers = num_layers
        if num_layers - 1 != len(layers_output_sizes_but_last):
            print("the layers_output_sizes_but_last should have 1 less items then num_layers")
            exit()
        self.layers_output_sizes = layers_output_sizes_but_last + [self.action_size]
        self._num_features = (self.inputSize + 1) * self.layers_output_sizes[0] + \
                             sum([(self.layers_output_sizes[i-1] + 1) * self.layers_output_sizes[i] for i in range(1, num_layers)])
        print("state_size:" + str(self.inputSize))
        print("action_size:" + str(self.action_size))

        # Initialize the policy network
        self.policy = PolicyNetwork(self.inputSize, self.num_layers, self.layers_output_sizes)

    def num_features(self):
        return self._num_features

    def game_name(self):
        return self._game_name

    # Playing one round of the game
    def evaluate_task(self, individual):
        nn_weights = self.policy.get_policy_weights_from_individual(individual)
        scores = []
        for i in range(self.episodes_per_fitness):
            state = self.env.reset()
            episode_rewards = 0
            for step in range(self.env._max_episode_steps):
                action = self.policy.run_nn_get_action(state, nn_weights)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                episode_rewards += reward

                if render:
                    self.env.render()
                if done:
                    break
            scores.append(episode_rewards)
        return sum(scores) / float(len(scores)),


class PolicyNetwork:
    def __init__(self, state_size, num_layers, layers_output_sizes):
        self.input_size = state_size
        self.num_layers = num_layers
        self.layers_output_sizes = layers_output_sizes

    def run_nn_get_action(self, input_state, weights, is_continuous=False, continuous_arg=-1):
        x = input_state[np.newaxis, :]
        for i in range(self.num_layers):
            x = np.tanh(x.dot(weights[i*2]) + weights[i*2 + 1])
        if is_continuous:
            return continuous_arg * x[0]
        else:
            return np.argmax(x, axis=1)[0]


    def get_policy_weights_from_individual(self, individual):
        weights = []
        start_index = 0
        for i in range(len(self.layers_output_sizes)):
            layer_input_size = self.input_size if i == 0 else self.layers_output_sizes[i - 1]
            layer_output_size = self.layers_output_sizes[i]

            # Assigning W
            weight_size = layer_input_size * layer_output_size
            weights.append(np.array(individual[start_index: start_index + weight_size]).reshape(
                (layer_input_size, layer_output_size)))
            start_index += weight_size

            # Assigning b
            weight_size = layer_output_size
            weights.append(np.array(individual[start_index: start_index + weight_size]).reshape((1, layer_output_size)))
            start_index += weight_size
        return weights


def get_list_from_file(file_name):
    text = open(file_name).read()
    text = text.replace("[", "").replace("]", "")
    return [float(i.strip()) for i in text.split(",")]

def train_model(game_name = 'MountainCar-v0', episodes_per_fitness = 1):
    if len(sys.argv) == 2:
        game_name = sys.argv[1]
    fitness = GameFitness(game_name, num_layers=2, layers_output_sizes_but_last=[12], episodes_per_fitness=episodes_per_fitness)

    trainer = TrainEs.TrainES(fitness)
    print(trainer.train(maximize_fitness=True))


def test_model(log_dir="", game_name = 'MountainCar-v0', episodes=5, render_val = True):
    global render
    render = render_val
    if len(sys.argv) == 3:
        game_name = sys.argv[1]
        log_dir = sys.argv[2]
    fitness = GameFitness(game_name, num_layers=2, layers_output_sizes_but_last=[12])
    individual = get_list_from_file(os.path.join(log_dir, "final_model.txt"))
    print("Test results")
    scores = []
    for i in range(episodes):
        score = fitness.evaluate_task(individual)[0]
        scores.append(score)
    print(scores)
    avg_score = sum(scores)/episodes
    print("avg score over " + str(episodes) + " episodes: " + str(avg_score))
    print("min score: " + str(min(scores)))
    print("max score: " + str(max(scores)))

