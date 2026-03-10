# -*- coding: utf-8 -*-
"""
Reinforcement Learning
======================

Pekiştirmeli öğrenme stratejisi - ödül bazlı öğrenme.
"""

from typing import Dict, Any, List, Optional, Tuple
import logging
import time
import random
import math
from collections import defaultdict, deque

class ReinforcementLearning:
    """
    Pekiştirmeli öğrenme stratejisi.
    
    Özellikler:
    - Q-Learning
    - Action selection (epsilon-greedy, UCB)
    - Reward tracking
    - Policy improvement
    - Experience replay
    """
    
    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95, 
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        self.logger = logging.getLogger("ReinforcementLearning")
        self.is_initialized = False
        
        # RL parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = 0.01
        
        # Q-table
        self.q_table: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
        
        # Experience replay buffer
        self.experience_buffer: deque = deque(maxlen=10000)
        
        # Action space
        self.actions: List[str] = [
            "search_web", "analyze_content", "store_knowledge", 
            "make_inference", "ask_question", "explore_topic",
            "synthesize_information", "validate_fact"
        ]
        
        # Reward tracking
        self.reward_history: List[float] = []
        self.cumulative_reward = 0.0
        
        # Statistics
        self.stats = {
            "episodes": 0,
            "total_steps": 0,
            "average_reward": 0.0,
            "exploration_rate": epsilon,
            "q_table_size": 0
        }
    
    async def initialize(self) -> bool:
        """Reinforcement Learning'i başlat"""
        try:
            self.logger.info("🎮 Reinforcement Learning başlatılıyor...")
            self.is_initialized = True
            self.logger.info("✅ Reinforcement Learning başarıyla başlatıldı")
            return True
        except Exception as e:
            self.logger.error(f"Reinforcement Learning başlatma hatası: {e}")
            return False
    
    async def select_action(self, state: str, exploration_mode: bool = True) -> str:
        """Durum için en iyi aksiyonu seç"""
        try:
            if exploration_mode and random.random() < self.epsilon:
                # Exploration: rastgele aksiyon
                action = random.choice(self.actions)
                self.logger.debug(f"🎲 Exploration action: {action}")
                return action
            else:
                # Exploitation: en iyi Q-value'ya sahip aksiyon
                state_q_values = self.q_table[state]
                
                if not state_q_values:
                    # Yeni durum, rastgele başla
                    action = random.choice(self.actions)
                    self.logger.debug(f"🆕 New state, random action: {action}")
                    return action
                
                # En yüksek Q-value'ya sahip aksiyonu seç
                best_action = max(state_q_values.items(), key=lambda x: x[1])[0]
                self.logger.debug(f"🎯 Best action: {best_action} (Q={state_q_values[best_action]:.3f})")
                return best_action
                
        except Exception as e:
            self.logger.error(f"Action selection hatası: {e}")
            return random.choice(self.actions)
    
    async def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Q-value güncelle"""
        try:
            # Current Q-value
            current_q = self.q_table[state][action]
            
            # Next state'in maximum Q-value'su
            next_q_values = self.q_table[next_state]
            max_next_q = max(next_q_values.values()) if next_q_values else 0.0
            
            # Q-learning update rule
            new_q = current_q + self.learning_rate * (
                reward + self.discount_factor * max_next_q - current_q
            )
            
            # Update Q-table
            self.q_table[state][action] = new_q
            
            # Experience buffer'a ekle
            experience = {
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "timestamp": time.time()
            }
            self.experience_buffer.append(experience)
            
            # Reward tracking
            self.reward_history.append(reward)
            self.cumulative_reward += reward
            
            # Epsilon decay
            if self.epsilon > self.min_epsilon:
                self.epsilon *= self.epsilon_decay
            
            # Stats güncelle
            self.stats["total_steps"] += 1
            self.stats["exploration_rate"] = self.epsilon
            self.stats["q_table_size"] = len(self.q_table)
            
            if self.reward_history:
                self.stats["average_reward"] = sum(self.reward_history) / len(self.reward_history)
            
            self.logger.debug(f"📈 Q-value güncellendi: {state}->{action} = {new_q:.3f} (reward: {reward})")
            
        except Exception as e:
            self.logger.error(f"Q-value update hatası: {e}")
    
    async def experience_replay(self, batch_size: int = 32):
        """Experience replay ile batch öğrenme"""
        try:
            if len(self.experience_buffer) < batch_size:
                return
            
            # Random batch seç
            batch = random.sample(list(self.experience_buffer), batch_size)
            
            # Batch'teki her experience için Q-value güncelle
            for experience in batch:
                await self.update_q_value(
                    experience["state"],
                    experience["action"], 
                    experience["reward"],
                    experience["next_state"]
                )
            
            self.logger.info(f"🔄 Experience replay tamamlandı: {batch_size} experience")
            
        except Exception as e:
            self.logger.error(f"Experience replay hatası: {e}")
    
    async def get_policy(self, state: str) -> Dict[str, float]:
        """Durum için policy (action probabilities) al"""
        try:
            state_q_values = self.q_table[state]
            
            if not state_q_values:
                # Uniform distribution
                prob = 1.0 / len(self.actions)
                return {action: prob for action in self.actions}
            
            # Softmax policy
            temperature = 1.0
            exp_values = {}
            for action in self.actions:
                q_value = state_q_values.get(action, 0.0)
                exp_values[action] = math.exp(q_value / temperature)
            
            total_exp = sum(exp_values.values())
            policy = {action: exp_val / total_exp for action, exp_val in exp_values.items()}
            
            return policy
            
        except Exception as e:
            self.logger.error(f"Policy hesaplama hatası: {e}")
            # Fallback: uniform distribution
            prob = 1.0 / len(self.actions)
            return {action: prob for action in self.actions}
    
    async def evaluate_performance(self, window_size: int = 100) -> Dict[str, float]:
        """Son N adımdaki performansı değerlendir"""
        try:
            if len(self.reward_history) < window_size:
                recent_rewards = self.reward_history
            else:
                recent_rewards = self.reward_history[-window_size:]
            
            if not recent_rewards:
                return {"average_reward": 0.0, "reward_variance": 0.0, "trend": 0.0}
            
            # Average reward
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            
            # Reward variance
            variance = sum((r - avg_reward) ** 2 for r in recent_rewards) / len(recent_rewards)
            
            # Trend (son yarı vs ilk yarı)
            if len(recent_rewards) >= 10:
                mid_point = len(recent_rewards) // 2
                first_half_avg = sum(recent_rewards[:mid_point]) / mid_point
                second_half_avg = sum(recent_rewards[mid_point:]) / (len(recent_rewards) - mid_point)
                trend = second_half_avg - first_half_avg
            else:
                trend = 0.0
            
            return {
                "average_reward": avg_reward,
                "reward_variance": variance,
                "trend": trend,
                "window_size": len(recent_rewards)
            }
            
        except Exception as e:
            self.logger.error(f"Performance evaluation hatası: {e}")
            return {"average_reward": 0.0, "reward_variance": 0.0, "trend": 0.0}
    
    def get_status(self) -> Dict[str, Any]:
        """Reinforcement Learning durumunu al"""
        return {
            "initialized": self.is_initialized,
            "q_table_states": len(self.q_table),
            "total_experiences": len(self.experience_buffer),
            "cumulative_reward": self.cumulative_reward,
            "current_epsilon": self.epsilon,
            "stats": dict(self.stats)
        }
    
    async def shutdown(self) -> bool:
        """Reinforcement Learning'i kapat"""
        try:
            self.logger.info(f"💾 {len(self.q_table)} state ile Reinforcement Learning kapatılıyor")
            
            self.is_initialized = False
            self.logger.info("🎮 Reinforcement Learning kapatıldı")
            return True
            
        except Exception as e:
            self.logger.error(f"Reinforcement Learning kapatma hatası: {e}")
            return False
