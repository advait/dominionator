We begin by solving simplified versions of the game and gradually increase the complexity.

## Simplified Dominion
1. Single player
2. Value function is how quickly we can end the game (number of plys)
3. Limited card set (TBD)

## Big Picture
Our system will consist of the following components:
1. State representation of the game at any moment
2. A set of all possible valid actions from any given state:
   - Each action is a function from S_n to S_(n+1)
4. An NN that takes in as input a state S and outputs Q_est and Policy_est:
   - Q_est is the estimated **Q value** from this state (number of plys when the game is over)
   - Policy_est is a probability distribution over all moves (in logit space). Game logic is implemented
     such that invalid moves are omitted (i.e. logits set to -Inf)
5. A Monte Carlo Tree Search system such that:
   - We can model a game as a tree such that each state is a node and each action is a directed
     edge between two states
   - As we perform more iterations, we explore more subtrees, accumulating average win rate for each
     action as we go
   - After a threshold number of iterations, we simply play the action that has the lowest Q Value.
     Here we keep track of the following:
     - S_i (the input state)
     - Q_est (the estimated Q Value of S_i based on our MCTS estimation)
     - Policy (the visit counts (logits) for each possible action)
6. For a given game, we keep track of the sequence of actions played until the game ends. Once the
   game ends, we can train a new network to predict the the final Q Value and Policy *for each move
   in the game's move sequence given each S_i*.

## State Representation
The game state is composed of the following:
1. Ply (the zero-indexed number)
2. Win conditions (whether the game is over)
3. The player's hand
4. The player's draw pile
5. The player's discard pile
6. The kingdom cards (buyable)
7. Actions/Buys/Gold available
8. Whether we're in a specific card's sub-state (trash, discard, etc.)

### Card Embeddings
A card in Dominion is composed of the following *sub-components*:
1. Cost to acquire
2. +n Cards
3. +n Actions
4. +n Buys
5. +n Gold
6. +n Trash
7. +n Victory Points

Complex features such as Sentry (trash, discard, or reorder two cards) or Throne Room (play an
action card twice) are omitted for now.

Instead of implementing an independent embedding for each card, we create embeddings for each of
the sub-components of a card and sum them to product a card embedding. Note that there will be a
separate embedding for each value of `n` (e.g. `+1 Card` and `+2 Card` will be separate embeddings).

### Win Conditions
The game is over when either:
1. Province pile is empty
2. Three separate piles are empty

Once the game is over, the **Q value** of the state is the ply count. Our objective is to minimize
the Q value. Rather than use the raw ply count, we'll use the *inverse* of the ply count scaled
to the range [-1, 1]. This inverse has the following properties:
1. Q Value of 1 is the best possible outcome (win on the 0th ply)
2. Q Value of -1 is the worst possible outcome (game ends after MAX_PLYS)
3. Q Value of 0 is the average outcome (some value between 0 plys and MAX_PLYS)

### Player's Hand
We would like to implement the network as a transformers variant so that we are able to learn card
interactions via cross-attention.

Rather than a traditional sequence or image transformer which annotates inputs with position
embeddings, our model simply omits the postion embeddings and considers the whole set of cards
together.

In order to keep track of card counts, we will use *count embeddings* that simply represent the
number of each card within our hand.

### Draw Pile
The draw pile's random ordering creates a Partially Observable Markov Decision Process (POMDP).
We'll handle this using Information Set Monte Carlo Tree Search (IS-MCTS).

Each MCTS state reprsents an *information set* which is simply the set of all cards (unordered)
within the draw pile. This information set can be translated to a *probability distribution over
cards* which indicates the likelihood of drawing any one card.

When we expand an MCTS node, we sample multiple possible deck configurations to product *multiple
resulting states*. We perform NN Q_est and Policy estimates for *all such samples* and output an
aggregation over all of these estimates.

### Discard Pile
The discard pile is tracked similarly to the Player's Hand.

### Kingdom Cards
Kingdom cards must also be annotated with availability count embeddings.

We should consider whether we want separate embeddings for the player's hand vs. the kingdom.

### Actions/Buys/Gold available
The actions, buys, and gold available are tracked three separate slots each with a count embedding.

### Card Sub-States
After special action cards are played (e.g. Chapel - Trash up to 4 cards from your hand), the state
must indicate that we're in a special card-specific sub-state where the only actions available are
trashing a specific card or *ending the special sub-state*.

## NN Architecture
The NN is a function that takes in a state and outputs a Q_est and Policy_est.

We'll use a transformer-based architecture that discretizes the state into a set of "tokens".
Because the ordering of cards is irrelevant, we'll use a set transformer.

The key blocks of information include:
1. Pile contents (one token per card in: hand, draw, discard, kingdom)
2. Pile summaries (one token per pile: hand, draw, discard, kingdom)
3. Scalar values (one token per scalar value: ply, buys, actions, gold, victory points)

### Pile contents
Each pile contains a set of (cards, count) tuples. We want to encode this counts both as discrete
embeddings as well as continuous logprob values representing the probability of drawing that card.

1. The card's embedding (learned)
2. The pile type content embedding (learned, hand/draw/discard/kingdom)
3. The discrete count embedding (learned, 0-12, overflow for 13+)
4. The continuous count features (log(n + eps), n / total, log(n / total + eps))

The continuous count features will be projected into the embedding space with a linear layer.

### Pile summaries
For each pile, we will have a single token consisteng of:
1. The pile's type embedding (learned)
2. The pile's total discrete count embedding (learned, 0-12, overflow for 13+)
3. The pile's total continuous count features (log(total + eps))

The continuous count features will be projected into the embedding space with a linear layer.

### Scalar values
Similar to the pile summaries, each scalar value will receive its own embedding/token consisting of:
1. The scalar type embedding (learned)
2. The discrete scalar value embedding (learned, 0-12, overflow for 13+)
3. The continuous scalar value features (log(scalar value + eps))

The continuous scalar value features will be projected into the embedding space with a linear layer.

### Output heads
Each transformer layer will accumulate knowledge within each token via attention. To transform
this knowledge into a final output, we will use two separate output heads. Each output head will
receive a concatenation of all tokens.

1. Q_est: An MLP that takes in the concatenation of all tokens and outputs a single scalar value
   representing the estimated number of plys until game completion. The MLP structure will be:
   - Input: Concatenated token embeddings [n_tokens * embedding_dim]
   - Hidden layers: 2-3 layers with ReLU activation
   - Output: Single scalar with no activation function

2. Policy_est: An MLP that takes in the concatenation of all tokens and outputs a vector of logits
   corresponding to all possible actions. The MLP structure will be:
   - Input: Concatenated token embeddings [n_tokens * embedding_dim]
   - Hidden layers: 2-3 layers with ReLU activation
   - Output: Vector of logits [n_actions] with no activation function
   - Invalid moves will be masked by setting their logits to -Inf before softmax and temperature
