#!/usr/bin/env python3
"""
Baseline evaluation script for Balatro Gym environments.
Tests random agent, simple heuristic agent, and establishes performance metrics.
"""

import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle
from datetime import datetime
import json

from balatro_gym.env import EightCardDrawEnv

class RandomAgent:
    """Random baseline agent"""
    
    def get_action(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        """Select random valid action"""
        valid_actions = np.where(action_mask)[0]
        return np.random.choice(valid_actions)

class SimpleHeuristicAgent:
    """Simple heuristic agent for 8-card draw poker"""
    
    def get_action(self, obs: Dict[str, np.ndarray], action_mask: np.ndarray) -> int:
        """Select action based on simple poker heuristics"""
        valid_actions = np.where(action_mask)[0]
        phase = obs['phase'].item()
        
        if phase == 0:  # Discard phase
            return self._discard_decision(obs, valid_actions)
        else:  # Select-five phase
            return self._select_decision(obs, valid_actions)
    
    def _discard_decision(self, obs: Dict[str, np.ndarray], valid_actions: List[int]) -> int:
        """Heuristic discard decision - keep pairs and high cards"""
        cards = obs['cards']  # 8x52 one-hot encoding
        
        # Convert to card indices
        card_indices = []
        for i in range(8):
            for j in range(52):
                if cards[i, j] == 1:
                    card_indices.append(j)
                    break
        
        # Count ranks (0-12 for 2-A)
        rank_counts = defaultdict(int)
        for card_idx in card_indices:
            rank = card_idx % 13
            rank_counts[rank] += 1
        
        # Keep pairs and high cards (J, Q, K, A = ranks 9, 10, 11, 12)
        keep_mask = 0
        for i, card_idx in enumerate(card_indices):
            rank = card_idx % 13
            if rank_counts[rank] >= 2 or rank >= 9:  # Keep pairs or high cards
                keep_mask |= (1 << i)
        
        # Discard mask is inverse of keep mask
        discard_mask = (255 - keep_mask) & 255
        
        # Find valid discard action
        if discard_mask in valid_actions:
            return discard_mask
        
        # Fallback to conservative discard (keep all)
        return 0
    
    def _select_decision(self, obs: Dict[str, np.ndarray], valid_actions: List[int]) -> int:
        """Select best 5-card combination"""
        # For simplicity, just take first valid action
        # In practice, this would evaluate poker hand strength
        return valid_actions[0]

def evaluate_agent(agent, env: EightCardDrawEnv, n_episodes: int = 100) -> Dict:
    """Evaluate agent performance"""
    rewards = []
    episode_lengths = []
    wins = 0
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            action_mask = obs['action_mask']
            action = agent.get_action(obs, action_mask)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        episode_lengths.append(steps)
        if total_reward > 0.5:  # Threshold for "good" performance
            wins += 1
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'min_reward': np.min(rewards),
        'max_reward': np.max(rewards),
        'mean_episode_length': np.mean(episode_lengths),
        'win_rate': wins / n_episodes,
        'rewards': rewards
    }

def generate_trajectories(agent, env: EightCardDrawEnv, n_episodes: int = 50) -> List[Dict]:
    """Generate trajectory data for imitation learning"""
    trajectories = []
    
    for episode in range(n_episodes):
        trajectory = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'infos': []
        }
        
        obs, info = env.reset()
        
        while True:
            action_mask = obs['action_mask']
            action = agent.get_action(obs, action_mask)
            
            # Store step data
            trajectory['observations'].append(obs.copy())
            trajectory['actions'].append(action)
            trajectory['infos'].append({'action_mask': action_mask.copy()})
            
            obs, reward, terminated, truncated, info = env.step(action)
            trajectory['rewards'].append(reward)
            
            if terminated or truncated:
                break
        
        trajectories.append(trajectory)
    
    return trajectories

def main():
    """Run baseline evaluation and generate trajectory data"""
    print("Starting Baseline Evaluation for Balatro Gym")
    print("=" * 50)
    
    # Create environment
    env = EightCardDrawEnv()
    
    # Create agents
    random_agent = RandomAgent()
    heuristic_agent = SimpleHeuristicAgent()
    
    # Evaluation parameters
    n_eval_episodes = 200
    n_traj_episodes = 100
    
    print(f"Evaluating agents over {n_eval_episodes} episodes...")
    
    # Evaluate random agent
    print("Evaluating Random Agent...")
    random_results = evaluate_agent(random_agent, env, n_eval_episodes)
    
    # Evaluate heuristic agent
    print("Evaluating Heuristic Agent...")
    heuristic_results = evaluate_agent(heuristic_agent, env, n_eval_episodes)
    
    # Print results
    print("\nBASELINE RESULTS")
    print("=" * 50)
    print(f"Random Agent:")
    print(f"  Mean Reward: {random_results['mean_reward']:.4f} ± {random_results['std_reward']:.4f}")
    print(f"  Win Rate: {random_results['win_rate']:.2%}")
    print(f"  Avg Episode Length: {random_results['mean_episode_length']:.1f}")
    
    print(f"\nHeuristic Agent:")
    print(f"  Mean Reward: {heuristic_results['mean_reward']:.4f} ± {heuristic_results['std_reward']:.4f}")
    print(f"  Win Rate: {heuristic_results['win_rate']:.2%}")
    print(f"  Avg Episode Length: {heuristic_results['mean_episode_length']:.1f}")
    
    improvement = (heuristic_results['mean_reward'] - random_results['mean_reward']) / random_results['mean_reward'] * 100
    print(f"\nHeuristic Improvement: {improvement:+.1f}%")
    
    # Generate trajectory data
    print(f"\nGenerating {n_traj_episodes} trajectory samples...")
    trajectories = generate_trajectories(heuristic_agent, env, n_traj_episodes)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save evaluation results
    results = {
        'timestamp': timestamp,
        'environment': 'EightCardDraw-v0',
        'random_agent': random_results,
        'heuristic_agent': heuristic_results,
        'improvement_pct': improvement
    }
    
    with open(f'baseline_results_{timestamp}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save trajectory data
    with open(f'baseline_trajectories_{timestamp}.pkl', 'wb') as f:
        pickle.dump(trajectories, f)
    
    print(f"\nResults saved:")
    print(f"  - baseline_results_{timestamp}.json")
    print(f"  - baseline_trajectories_{timestamp}.pkl")
    
    # Create reward distribution plot
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(random_results['rewards'], alpha=0.7, label='Random', bins=20)
    plt.hist(heuristic_results['rewards'], alpha=0.7, label='Heuristic', bins=20)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    episodes = range(len(heuristic_results['rewards']))
    plt.plot(episodes, heuristic_results['rewards'], alpha=0.7, label='Heuristic')
    plt.axhline(y=heuristic_results['mean_reward'], color='r', linestyle='--', label='Mean')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Over Episodes')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'baseline_analysis_{timestamp}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"  - baseline_analysis_{timestamp}.png")
    
    print("\nBaseline evaluation complete!")
    print("\nNext steps:")
    print("1. Use trajectory data for imitation learning")
    print("2. Set target performance above heuristic baseline")
    print("3. Begin RL training with PPO/DQN")

if __name__ == "__main__":
    main()