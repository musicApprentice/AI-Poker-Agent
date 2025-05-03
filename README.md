
Q-Learning Poker Agent - Training Summary
=========================================

Overview:
---------
This project implements a Q-learning based Poker AI agent (with some manual detection) trained through staged opponent play and self-play. The agent learns decision-making strategies in heads-up Texas Hold'em using state abstraction, opponent modeling, and TD-learning.

Training Stages:

Altogether, we ran more than 10,000 games on this particular model (not counting the over 200,000 games we played for outdated q-tables)
----------------
1. **Against AllInPlayer**:
   - Trained on approximately 3500 games.
   - Designed an algorithm to exploit players who were overly aggressive 
   - If we detect behavior that is consistently aggressive we play conservatively 
   - Converged to 50-55% win rate; however, because our exploitative algorithm is conservative we fold early on weak hands and play strong and medium hands aggressively as to outsmart a naive algorithm that always raises or calls 
   - As a result, we consistently defeat AllInPlayer in when we run testperf.py as we minimize our losses and maximize our rewards

2. **Against RaisedPlayer**:
   - Trained on approximately 500 games.
   - Consistently strong performance with convergence to 97-99% win rate

2. **Against RandomPlayer**:
   - Trained on approximately 1500 games 
   - Consistently strong performance with convergence to 97-99% win rate

3. **Self-Play (vs Frozen Agent v1)**:
   - Trained over 2000+ games.
   - Final win rate converged around 50–53%.

4. **Self-Play (vs Frozen Agent v2)**:
   - Trained over 2000+ games on our model from (3).
   - Final win rate converged to around 40-45%.

Key Features:
-------------
- Eligibility Traces (λ = 0.8): Enables multi-step TD error propagation.
- Reward Shaping: +2 reward when opponent folds after a raise.
- Opponent Profiling: Tracks action frequencies and switches between exploitative and q-learning models
- Training Assertions: Ensures every visited (state, action) pair is updated.
- Adaptive Exploration: Learns using ε-greedy policy with state abstractions.
- Clever Abstraction: Hands, opponent behavior and rounds are abstracted to form abstract states for q-learning
- Alternating Opponents: Supports training against different strategies.

Files:
------
- `qlearning_player.py`: Main training agent with Q-learning + eligibility traces.
- `train_q_agent.py`: Script to train vs other players like AllIn or Random players.
- `train_q_agent_self.py`: Script to train via self-play.
- `allin_player.py`, `randomplayer.py`: Simple opponent implementations.
- `q_table01.pkl`: Main learned Q-table (periodically saved).
- `q_table_frozen.pkl`: Frozen snapshot of Q-table used in self-play.

