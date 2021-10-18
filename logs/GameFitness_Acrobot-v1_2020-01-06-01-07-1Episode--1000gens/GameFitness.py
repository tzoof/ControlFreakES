import gym
import numpy as np
import TrainEs
import sys
import os

render = False

class GameFitness():
    def __init__(self, game_name, num_layers, layers_output_sizes_but_last):
        np.random.seed(123)
        self.game_name = game_name
        self.env = gym.make(game_name)

        # Define Network
        if type(self.env.observation_space) == gym.spaces.box.Box:
            self.inputSize = self.env.observation_space.shape[0]
        else:
            self.inputSize = self.env.observation_space.n

        if type(self.env.action_space) == gym.spaces.box.Box:
            self.action_size = self.env.action_space.shape[0]
        else:
            self.action_size = self.env.action_space.n

        self.num_layers = num_layers
        if num_layers - 1 != len(layers_output_sizes_but_last):
            print("the layers_output_sizes_but_last should have 1 less items then num_layers")
            exit()
        self.layers_output_sizes = layers_output_sizes_but_last + [self.action_size]
        self.num_features = (self.inputSize + 1) * self.layers_output_sizes[0] + \
                            sum([(self.layers_output_sizes[i-1] + 1) * self.layers_output_sizes[i] for i in range(1, num_layers)])
        print("state_size:" + str(self.inputSize))
        print("action_size:" + str(self.action_size))

        # Initialize the policy network
        self.policy = PolicyNetwork(self.inputSize, self.num_layers, self.layers_output_sizes, self.game_name)

    # Playing one round of the game
    def evaluate_task(self, individual: object) -> object:
        nn_weights = self.policy.get_policy_weights_from_individual(individual)
        state = self.env.reset()
        rewards = 0
        for step in range(self.env._max_episode_steps):
            action = self.policy.run_nn_get_action(state, nn_weights)
            # actions_distribution = self.sess.run(self.policy.actions_distribution, {self.policy.state: state})
            # action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            rewards += reward

            if render:
                self.env.render()
            if done:
                break
        return rewards,


class PolicyNetwork:
    def __init__(self, state_size, num_layers, layers_output_sizes, game_name):
        self.input_size = state_size
        self.num_layers = num_layers
        self.layers_output_sizes = layers_output_sizes
        self.game_name = game_name

    def run_nn_get_action(self, input_state, weights, is_continuous=False, continuous_arg=-1):
        x = input_state[np.newaxis, :]
        for i in range(self.num_layers):
            x = np.tanh(x.dot(weights[i*2]) + weights[i*2 + 1])
        if is_continuous:
            return continuous_arg * x[0]
        else:
            #if 'MountainCar' in self.game_name and input_state[0] > -0.1: return 0
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

def train_model(game_name = 'MountainCar-v0'):
    if len(sys.argv) == 2:
        game_name = sys.argv[1]
    fitness = GameFitness(game_name, num_layers=2, layers_output_sizes_but_last=[12])
    trainer = TrainEs.TrainES(fitness)
    print(trainer.train(maximize_fitness=True))


def test_model(log_dir="", game_name = 'MountainCar-v0'):
    global render
    render = True
    if len(sys.argv) == 3:
        game_name = sys.argv[1]
        log_dir = sys.argv[2]
    fitness = GameFitness(game_name, num_layers=2, layers_output_sizes_but_last=[12])
    individual = get_list_from_file(os.path.join(log_dir, "final_model.txt"))
    print("Test results")
    for i in range(5):
        print(fitness.evaluate_task(individual))

if __name__ == "__main__":
    #test_model(log_dir="D:\Back2School\Semester3\ES\Code\EvolutionaryAlgorithms\Project\RL\PolicyGradients_ES\logs\GameFitness_Acrobot-v1_2020-01-05-13-55-ReturningToDefaultParams",
    #               game_name="Acrobot-v1")
    #test_model(log_dir="D:\Back2School\Semester3\ES\Code\EvolutionaryAlgorithms\Project\RL\PolicyGradients_ES\logs\GameFitness_CartPole-v1_2020-01-05-18-08-ReturningToDefaultParams",
    #               game_name="CartPole-v1")
    train_model("Acrobot-v1")
    sys.exit()

