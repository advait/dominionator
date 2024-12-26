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
4. An NN that takes in as input a state S and outputs Q_est and Policy:
   - Q_est is the estimated **Q value** from this state (number of plys when the game is over)
   - Policy is a probability distribution over all moves (in logit space). Game logic is implemented
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
the Q value.

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
