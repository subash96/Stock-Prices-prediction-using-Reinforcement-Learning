from keras import layers, models,regularizers,initializers
from keras import backend as K
from keras.optimizers import Adam


class Critic:


    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):

        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        net_state_values = layers.Dense(units=16,kernel_regularizer=regularizers.l2(1e-6))(states)
        net_state_values = layers.BatchNormalization()(net_state_values)
        net_state_values = layers.Activation("relu")(net_state_values)

        net_state_values = layers.Dense(units=32, kernel_regularizer=regularizers.l2(1e-6))(net_state_values)

        net_action_values = layers.Dense(units=32,kernel_regularizer=regularizers.l2(1e-6))(actions)

        net = layers.Add()([net_state_values, net_action_values])
        net = layers.Activation('relu')(net)

        Q_values = layers.Dense(units=1, name='q_values',kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003))(net)

        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        optimizer = Adam(lr=0.001)
        self.model.compile(optimizer=optimizer, loss='mse')

        action_gradients = K.gradients(Q_values, actions)

        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)
