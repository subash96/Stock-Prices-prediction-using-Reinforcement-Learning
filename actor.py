from keras import layers, models,regularizers
from keras import backend as K
from keras.optimizers import Adam

class Actor:

    def __init__(self, state_size, action_size):

        self.state_size = state_size
        self.action_size = action_size

        self.build_model()

    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        
        net_val = layers.Dense(units=16,kernel_regularizer=regularizers.l2(1e-6))(states)
        net_val = layers.BatchNormalization()(net_val)
        net_val = layers.Activation("relu")(net_val)
        net_val = layers.Dense(units=32,kernel_regularizer=regularizers.l2(1e-6))(net_val)
        net_val = layers.BatchNormalization()(net_val)
        net_val = layers.Activation("relu")(net_val)

        actions = layers.Dense(units=self.action_size, activation='softmax', name = 'actions')(net_val)
        
        self.model = models.Model(inputs=states, outputs=actions)

        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        optimizer = Adam(lr=.00001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[actions],
            updates=updates_op)