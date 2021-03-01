# Alpha Zero

Sources:

* [Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm](https://arxiv.org/pdf/1712.01815.pdf)
* Mastering the Game of Go without Human Knowledge 


## Notes

* The MCTS search outputs probabilities π of playing each move. . These search probabilities usually select **much stronger moves than the raw move probabilities p **of the neural network fθ(s); MCTS may therefore be viewed as a **powerful policy improvement operator**
* Self-play with search – using the improved MCTS-based policy to select each move, then using the game winner z as a sample of the value – may be viewed as **a powerful policy evaluation operator**.

=> Role of evaluation to review

Reinforcement learning Policy iteration 20, 21 is a classic algorithm that generates a sequence of improving policies, by alternating between **policy evaluation** – estimating the value function of the current policy – and **policy improvement** – using the current value function to generate a better policy. A simple approach to policy evaluation is to estimate the value function from the outcomes of sampled trajectories 35,36. A simple approach to policy improvement is to select actions greedily with respect to the value function 20. In large state spaces, approximations are necessary to evaluate each policy and to represent its improvement 22,23.



* MCTS may be viewed as a self-play algorithm that, given neural network parameters θ and a root position s, computes a vector of search probabilities recommending moves to play, π = $α_θ (s)$, proportional to the exponentiated visit count for each move, $π_a ∝ N (s, a)^{1/τ} $, where τ is a temperature parameter.

* L2 weight regularization???

* Figure 3 page 8 - empiricial Evaluation

* contributions of architecture and algorithm
* Using a residual network was more accurate, achieved lower error, and improved performance in AlphaGo by over 600 Elo. 
* Combining policy and value together into a single network **slightly reduced the move prediction accuracy, but reduced the value error** and boosted playing performance in AlphaGo by around another 600 Elo

* MCTS search parameters were selected by Gaussian process optimisation 68, so as to optimise self-play performance of AlphaGo Zero using a neural network trained in a preliminary run (p23)
*
* Neural network parameters are optimised by stochastic gradient descent with momentum and **learning rate annealing**. The learning rate is annealed according to the standard schedule in Extended Data Table 3
* The momentum parameter is set to 0.9. The cross-entropy and mean-squared error losses are weighted equally (this is reasonable because rewards are unit scaled, r ∈ {−1, +1}) and the L2 regularisation parameter is set to c = 10−4.
* temperature $\tau$ for exploration?? set close to 0 during evaluation
* Additional exploration is achieved by **adding Dirichlet noise** to the prior probabilities in the root node $s_0$ , specifically $P(s, a) = (1 − ε)p_a + εη_a$ , where η ∼ Dir(0.03) and ε = 0.25; this noise ensures that all moves may be tried, but the search may still overrule bad moves.
* resignation parameters? To measure false positives, we disable resignation in 10% of self-play games and play until termination.
* Each node s in the search tree contains edges (s, a) for all legal actions a ∈ A(s). Each edge stores a set of statistics, {N(s,a),W(s,a),Q(s,a),P(s,a)},
where N (s, a) is the visit count, W (s, a) is the total action-value, Q(s, a) is the mean action-value, and P (s, a) is the prior probability of selecting that edge.

The algorithm proceeds by iterating over three phases (a–c in Figure 2), and then selects a move to play (d in Figure 2).

* **Select** 
 * The first in-tree phase of each simulation begins at the root node of the
search tree, s0, and finishes when the simulation reaches a leaf node sL at time-step L. At each of these time-steps, t < L, an action is selected according to the statistics in the search tree, at =argmax􏰀Q(st,a)+U(st,a)􏰁,usingavariantofthePUCTalgorithm24, a
􏰃􏰂b N(s,b) U(s,a)=cpuctP(s,a) 1+N(s,a) where cpuct is a constant determining the level of exploration; this search control strategy initially prefers actions with high prior probability and low visit count, but asympotically prefers actions with high action-value.

* **Expand and evaluate** 
 * The leaf node sL is added to a queue for neural network evaluation, (di(p), v) = fθ(di(sL)), where di is a dihedral reflection or rotation selected uniformly at random from i ∈ [1..8].
 
 * Positions in the queue are evaluated by the neural network using a mini-batch size of 8; the search thread is locked until evaluation completes. The leaf node is expanded and each edge (sL, a) is initialised to {N(sL,a) = 0,W(sL,a) = 0,Q(sL,a) = 0,P(sL,a) = pa}; the value v is then backed up.
 
* **Backup**. 
 * The edge statistics are updated in a backward pass through each step t ≤ L. The visit counts are incremented, N(st, at) = N(st, at)+1, and the action-value is updated
to the mean value, W(st,at) = W(st,at) + v,Q(st,at) = W(st,at). 
 * We use virtual loss to ensure N(st,at) each thread evaluates different nodes 69.


* **Play**. At the end of the search AlphaGo Zero selects a move a to play in the root position s0, proportional to its exponentiated visit count, π(a|s0) = N(s0, a)1/τ / 􏰂b N(s0, b)1/τ , where τ is a temperature parameter that controls the level of exploration. 
 * The search tree is reused at subsequent time-steps: the child node corresponding to the played action becomes the new root node; the subtree below this child is retained along with all its statistics, while the remainder of the tree is discarded. AlphaGo Zero resigns if its root value and best child value are lower than a threshold value $v_resign$.

* During an MCTS simulation, we repeatedly simulate play from a root node (s) representing the current board state. The first thing we do is query the neural network for the prior probability vector (p) of potential actions from s.
Instead of using p as-is, AlphaZero adds noise according to the Dirichlet distribution with parameter ɑ, aka Dir(ɑ) lphaZero uses a truncated policy, backing up values directly from the neural network predictions instead of playing out the game for each simulation


## Questions

* input features: only white and black stone
* From the board state only, how do we know the current player?
* Could TD-learning be used?
* Dirichlet distribution [blog](https://towardsdatascience.com/dirichlet-distribution-a82ab942a879)
* Compared to the MCTS in AlphaGo Fan and AlphaGo Lee, the principal differences are that AlphaGo Zero does not use any rollouts; it uses a single neural network instead of separate policy and value networks; leaf nodes are always expanded, rather than using dynamic expansion; each search thread simply waits for the neural network evaluation, rather than performing evaluation and backup asynchronously; and there is no tree policy.
* History features Xt, Yt are necessary because Go is not fully observable solely from the current stones, as repetitions are forbidden; similarly, the colour feature C is necessary because the komi is not observable.
* asynchronous multi-thread?
* residual networks (the authors tried a variety of architectures including networks with and without residual networks, and with and without parameter sharing for the value and policy networks. Their best architecture used residual networks and shared the parameters for the value and policy networks.)
* board state + history
* Why the MTCS process stabilizes the training so well?
* raw network vs. network + MCTS! 
* is network architecture game-dependent???
* By keeping old tree branches, don't we mix old and new policies?
* What is learnt faster? z or $\pi_{\theta}$?




* TRAINING TARGET _ USING Q s. Z
Based on these descriptions, z seems superior, but there is one large drawback of using z: each game has only a single result, and that single result can be heavily influenced by randomness. For example: imagine a position early in the game where the network has made the correct move, but later ends up choosing a suboptimal move and losing the game. In this case z will be -1 and training will incorrectly associate a low score with the position.
With enough training data, one would hope that these mistakes get overshadowed by correct play. Unfortunately, it is impossible to completely eradicate the mistakes, because the network explores during self-play due to its probabilistic policy. We theorize that this is one of the reasons dropping the temperature after 30 moves was so important to AlphaZero. Otherwise, randomness in move choice towards the end of game play could compromise the accuracy of z.
Unfortunately, q is not a perfect solution. It can suffer from something called the “horizon effect”. This can happen when the simulations return a positive result, but there is a killer response that is just beyond the search horizon, i.e. not visited during the 800 simulation

Our experimentation shows that we can achieve better results by using both q and z together. One way to combine q and z is to average them for each example position and use that average to train the network. This seems to give the benefits of both: z helps counteract q’s horizon effect and q helps counteract z’s randomness. Another promising approach is to begin by training against z, but linearly falloff to q over a number of generations


* when using INT8 you must first generate a calibration file to tell the inference engine what scale factors to apply to your layer activations when using 8-bit approximated math. This calibration is done by feeding a sample of your data into Nvidia’s calibration library.

One downside of using INT8 is that it can be lossy and imprecise in certain situations. While we didn’t observe serious precision issues during the early parts of training, as learning progressed we would observe the quality of inference start to degrade, particularly on our value output. This initially led us to use INT8 only during the very early stages of training.
Serendipitously, we were able to virtually eliminate our INT8 precision problem when we began experimenting with increasing the number of convolutional filters in our head networks, an idea we got from Leela Chess. Below is a chart of our value output’s mean average error with 32 filters in the value head, vs. the AZ default of 1:


[](https://deepmind.com/research/case-studies/alphago-the-story-so-far)


## Hyper-Parameters List

|Hyperparameters|Oracle|Alpha0|
|---|---|---|
|puct|3-4|Alpha0|
|Dirichlet $\alpha$|1|for chess, shogi, and Go: 0.3, 0.15, and 0.03|
|temperature $\tau$|1 (no effect)|see paper...|
|Learning Rate| [Cyclical](https://arxiv.org/abs/1803.09820)|with a fixed learning rate which they periodically tweak|

## Tweakings

|Hyperparameters|Oracle|Alpha0|
|---|---|---|
|8-bit quantization|||
|Position deduplication|Oracle|Alpha0|
Rather than asking our network to average the data on its own, we experimented with performing de-duplication and averaging of the data prior to presenting it to the network. In theory, this creates less work for the network, as it does not have to learn this average itself. Also, de-duplication allows us to present more unique positions to a network each training cycle.

After game generation, training is then performed so that our model can learn from recently generated data to create even more refined gameplay examples in the next cycle. We found that using 2 epochs of training per window sample provided a good bump in learning over single epoch training, without bottlenecking our synchronous training cycle for too long.

Slow Window
In AlphaZero, the authors used a sliding window of size 500,000 games, from which they sampled their training data uniformly. In our first implementation, we used a sliding training window composed of 20 generations of data, which amounts to 143360 games. During our experiments, we noticed that at model 21, there would be a large drop in training error, and a noticeable bump in evaluation performance, just as the amount of available data exceeded the training window size and old data started to get expunged. This seemed to imply that older, less refined data, could be holding back learning.
To counteract this, we implemented a slowly increasing sampling window, where the size of the window would start off small, and then slowly increase as the model generation count increased. This allowed us to quickly phase out very early data before settling to our fixed window size. We began with a window size of 4, so that by model 5, the first (and worst) generation of data was phased out. We then increased the history size by one every two models, until we reached our full 20 model history size at generation 35.





## Description

* Instead of a handcrafted evaluation function and move ordering heuristics, AlphaZero utilises a deep neural network $(p, v) = f_θ(s)$ with parameters θ.

* This neural network is structured with:
 * INPUT: the board position s
 * OUTPUTS: a vector of move probabilities p with components $p_a = Pr(a|s)$ for each action a, $p_{a_1}, p_{a_2}, p_{a_3}$... and a scalar value v estimating the expected outcome z of the current player, from position s, v ≈ E[z|s].

* AlphaZero **learns these move probabilities and value estimates entirely from self- play**; 

* These are **then used to guide its search**.

* Instead of an alpha-beta search with domain-specific enhancements, AlphaZero uses a **general- purpose Monte-Carlo tree search (MCTS) algorithm**. Each search consists of a series of simulated games of self-play that traverse a tree from root $s_{root}$ to leaf. 

* Each simulation proceeds by selecting in each state s a move a **with low visit count**, **high move probability** and **high value** (averaged over the leaf states of simulations that selected a from s) according to the current neural network $f_θ$. 

* The search returns a vector $\pi$ representing a probability distribution over moves, either proportionally or greedily with respect to the visit counts at the root state.

* The parameters θ of the deep neural network in AlphaZero are trained by self-play reinforcement learning, starting from randomly initialised parameters θ. Games are played by selecting moves for both players by MCTS, $a_t ∼ \pi_t$. 

* At the end of the game, the terminal position $s_T$ is scored according to the rules of the game to compute the game outcome z: −1 for a loss, 0 for a draw, and +1 for a win. 

* The neural network parameters θ are updated so as to minimise the error between the predicted outcome $v_t$ and the game outcome z, and to maximise the similarity of the policy vector $p_t$ to the search probabilities $\pi_t$. 

Specifically, the parameters $\theta$ are adjusted by gradient descent on a loss function l that sums over mean-squared error and cross-entropy losses respectively,

$$l - (z - v)^2 - \pi^Tlogp + c||\theta||^2$$

where c is a parameter controlling the level of **L2 weight regularisation**. The updated parameters are used in subsequent games of self-play.


## Implementation Details

* The neural network takes as input the state and outputs both an expert policy (equivalent to an actor) and an expert critic (equivalent to a critic). In the Alpha Zero paper, both the policy and the critic share the same network, so the same weights $\theta$

![alpha_0_algo](Assets/alpha_0_algo.png)





## Monte Carlo Tree Search (MCTS) and Alpha-Beta Search

### Alpha-Beta Search

Wikipedia: Alpha–beta pruning is a search algorithm that seeks to decrease the number of nodes that are evaluated by the **minimax algorithm** in its search tree. It is an adversarial search algorithm used commonly for machine playing of two-player games (Tic-tac-toe, Chess, Go, etc.). It stops evaluating a move when at least one possibility has been found that proves the move to be worse than a previously examined move. Such moves need not be evaluated further. When applied to a standard minimax tree, it returns the same move as minimax would, but prunes away branches that cannot possibly influence the final decision.


For at least four decades the strongest computer chess programs have used alpha-beta search (18, 23). AlphaZero uses a markedly different approach that averages over the position evaluations within a subtree, rather than computing the minimax evaluation of that subtree. However, chess programs using traditional MCTS were much weaker than alpha-beta search programs, (4, 24); while alpha-beta programs based on neural networks have previously been un- able to compete with faster, handcrafted evaluation functions.
AlphaZero evaluates positions using non-linear function approximation based on a deep neural network, rather than the linear function approximation used in typical chess programs. This provides a much more powerful representation, but may also introduce spurious approximation errors. MCTS averages over these approximation errors, which therefore tend to cancel out when evaluating a large subtree. In contrast, alpha-beta search computes an explicit minimax, which propagates the biggest approximation errors to the root of the subtree. Using MCTS may allow AlphaZero to effectively combine its neural network representations with a powerful, domain-independent search.
