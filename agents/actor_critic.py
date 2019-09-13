import random
import copy
import numpy as np
from keras import layers, models, optimizers, regularizers, initializers
from keras import backend as K
from collections import namedtuple, deque

#algorithm parameters
actor_layers = [64, 32] #dense layer sizes in actor network
critic_layers = [64, 32, 16] #dense layer sizes in critic network

use_bn = False #use batch normalization in each dense layer, between kernel and activation
acti = "softsign" #activation for most layers (excluding final layers such as the actor's raw actions)
gamma = 0.99  #reward discount factor

actor_l2 = 0.0 #l2 regularization weight in the actor network (if 0, then none applied)
critic_l2 = 0.0 #same as above, but in critic network

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size) # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


#### ACTOR ####
class Actor:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high, lr):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
            lr (float): Learning rate
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.lr = lr

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = states
        
        for l in actor_layers:
            l2 = regularizers.l2(actor_l2) if actor_l2 > 0.0 else None
            net = layers.Dense(units=l, kernel_regularizer = l2)(net)
            if use_bn: net = layers.BatchNormalization()(net)
            net = layers.Activation(acti)(net)

        # Add final output layer with sigmoid activation
        #initializer from https://arxiv.org/pdf/1509.02971.pdf section 7: experiment details
        initializer = initializers.RandomUniform(-1e-3, 1e-3)
        net = layers.Dense(units = self.action_size,
                           kernel_initializer = initializer)(net)
        raw_actions = layers.Activation("sigmoid", name="raw_actions")(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=self.lr)
        #optimizer = optimizers.SGD(lr=self.lr)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

#### CRITIC ####
class Critic:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, lr):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            lr (float): Learning rate
        """
        self.state_size = state_size
        self.action_size = action_size

        self.lr = lr

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        net = layers.Concatenate()([states, actions])
        
        for l in critic_layers:
            l2 = regularizers.l2(critic_l2) if critic_l2 > 0.0 else None
            net = layers.Dense(units=l, kernel_regularizer = l2)(net)
            if use_bn: net = layers.BatchNormalization()(net)
            net = layers.Activation(acti)(net)

        # Add final output layer to prduce action values (Q values)
        #initializer from https://arxiv.org/pdf/1509.02971.pdf section 7: experiment details
        initializer = initializers.RandomUniform(-1e-3, 1e-3)
        Q_values = layers.Dense(units=1,
                                kernel_initializer = initializer,
                                name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr = self.lr)
        #optimizer = optimizers.SGD(lr=self.lr)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

#### AGENT ####
class ActorCritic():
    def __init__(self, task, actor_lr = 0.001, critic_lr = 0.01, noise_std = 0.01):
        """
        Params
        ======
            actor_lr (int): Actor learning rate (alpha)
            critic_lr (int): Critic learning rate (alpha)
            noise_std (float): Noise standard deviation (epsilon)
        """
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low
        
        # Actor (Policy) Model
        self.actor = Actor(self.state_size, self.action_size, self.action_low, self.action_high, actor_lr)

        # Critic (Value) Model
        self.critic = Critic(self.state_size, self.action_size, critic_lr)

        #epsilon greedy
        self.noise_std = noise_std

        # Replay memory
        self.buffer_size = 100000
        self.batch_size = 64
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
    
        #critic error tracking
        self.mean_errs = []
        self.errs = []

    def reset_episode(self):
        state = self.task.reset()
        self.last_state = state
        self.errs = []
        return state

    def step(self, action, reward, next_state, done):
        # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)
        
        if done:
            self.mean_errs.append(np.mean(self.errs))

        # Roll over last state
        self.last_state = next_state

    def act(self, state):
        """Returns actions for given state(s) as per current policy."""
        state = np.reshape(state, [-1, self.state_size])
        action = self.actor.model.predict(state)[0]
        #add noise from a normal distribution with stdev = noise_std * action_range
        action = np.random.normal(action, self.action_range * self.noise_std)
        return action

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor.model.predict_on_batch(next_states)
        Q_targets_next = self.critic.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic
        Q_targets = rewards + gamma * Q_targets_next * (1 - dones)
        err = self.critic.model.train_on_batch(x=[states, actions], y=Q_targets)
        self.errs.append(err)

        # Train actor
        action_gradients = np.reshape(self.critic.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor.train_fn([states, action_gradients, 1])  # custom training function
