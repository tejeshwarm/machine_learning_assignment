import numpy as np
import random
from collections import defaultdict

# TicTacToe Environment
class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)  # 0 = empty, 1 = X, -1 = O
        self.current_player = 1  # Start with player X

    def reset(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.board.flatten()

    def is_winner(self, player):
        # Check rows, columns, and diagonals for a win
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return True
        return False

    def is_draw(self):
        return np.all(self.board != 0)

    def step(self, action):
        if self.board[action] != 0:
            raise ValueError("Invalid action! Cell already taken.")
        
        self.board[action] = self.current_player
        
        if self.is_winner(self.current_player):
            return self.board.flatten(), 1, True  # Reward for winning
        elif self.is_draw():
            return self.board.flatten(), 0, True  # Reward for draw
        else:
            self.current_player *= -1  # Switch players
            return self.board.flatten(), 0, False  # No reward and game continues

    def available_actions(self):
        return np.argwhere(self.board.flatten() == 0).flatten()

    def print_board(self):
        print("\n".join([" | ".join([[" ", "X", "O"][int(cell)] for cell in row]) for row in self.board]))
        print()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.q_table = defaultdict(lambda: np.zeros(9))  # 3x3 board flattened
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay

    def choose_action(self, state):
        available_actions = np.where(state == 0)[0]  # Filter only available actions
        if np.random.rand() < self.exploration_rate:
            return random.choice(available_actions)  # Explore: choose random action from available
        else:
            # Choose best action among available actions
            q_values = self.q_table[tuple(state)]
            best_actions = [a for a in available_actions if q_values[a] == np.max(q_values[available_actions])]
            return random.choice(best_actions)  # Randomly pick among best available actions to add exploration

    def learn(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[tuple(next_state)])
        td_target = reward + self.discount_factor * self.q_table[tuple(next_state)][best_next_action]
        td_delta = td_target - self.q_table[tuple(state)][action]
        self.q_table[tuple(state)][action] += self.learning_rate * td_delta

    def update_exploration_rate(self):
        self.exploration_rate *= self.exploration_decay

# Training the agent
def train_agent(agent, episodes=10000):
    env = TicTacToe()
    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            try:
                next_state, reward, done = env.step((action // 3, action % 3))
            except ValueError:
                # Skip this action and try again
                continue

            agent.learn(state, action, reward, next_state)
            state = next_state

        agent.update_exploration_rate()

    print("Training complete!")

# Function to play the game against the agent
def play_game(agent):
    env = TicTacToe()
    print("Welcome to Tic-Tac-Toe! You are X and the AI is O.")

    while True:
        state = env.reset()
        done = False

        while not done:
            env.print_board()

            # Player move
            while True:
                try:
                    player_move = int(input("Enter your move (0-8): "))  # 0-8 for the board
                    if player_move < 0 or player_move > 8:
                        print("Invalid move! Please enter a number between 0 and 8.")
                        continue
                    next_state, reward, done = env.step((player_move // 3, player_move % 3))
                    break
                except ValueError:
                    print("Invalid input! Please enter a valid move.")
                except Exception as e:
                    print(e)

            if done:
                break

            # AI move
            action = agent.choose_action(state)
            next_state, reward, done = env.step((action // 3, action % 3))

            state = next_state

        env.print_board()
        if env.is_winner(1):
            print("Congratulations! You win!")
        elif env.is_winner(-1):
            print("AI wins! Better luck next time.")
        else:
            print("It's a draw!")

        if input("Do you want to play again? (yes/no): ").lower() != 'yes':
            break

# Run training and play the game
if __name__ == "__main__":
    agent = QLearningAgent()
    print("Training the agent...")
    train_agent(agent, episodes=10000)  # Train the agent for 10,000 games
    play_game(agent)
