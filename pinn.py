import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error


class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense2 = tf.keras.layers.Dense(64, activation='tanh')
        self.dense3 = tf.keras.layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        output = self.dense3(x)
        return output



def physics_informed_loss(model, X_data, Y_data, X_collocation):
    # Mean Squared Error for Data
    predictions = model(X_data)
    mse_data = tf.reduce_mean(tf.square(predictions - Y_data))

    # Physics-informed loss using heat transfer PDE (Example for 1D Heat Equation)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_collocation)
        u = model(X_collocation)
        u_x = tape.gradient(u, X_collocation)
        u_xx = tape.gradient(u_x, X_collocation)
    
    pde_loss = tf.reduce_mean(tf.square(u_xx))  # Example: (d^2u/dx^2)

    return mse_data + pde_loss


def train_pinn(model, X_data, Y_data, X_collocation, epochs=1000, learning_rate=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = physics_informed_loss(model, X_data, Y_data, X_collocation)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.numpy()}')


def integrate_fea_results(pinn_model, fea_results, X_test):
    # Example: Compare PINN predictions with FEA results
    pinn_predictions = pinn_model(X_test).numpy()
    mae = mean_absolute_error(fea_results, pinn_predictions)
    print(f'Mean Absolute Error between PINN and FEA: {mae}')
    return mae

# Example Data (Replace with real data)
X_data = np.random.rand(100, 1)
Y_data = np.sin(np.pi * X_data)
X_collocation = np.random.rand(100, 1)

# Initialize and train the PINN
pinn_model = PINN()
train_pinn(pinn_model, X_data, Y_data, X_collocation)

# Integrate with FEA results
fea_results = np.sin(np.pi * X_data)  # Placeholder for FEA results
mae = integrate_fea_results(pinn_model, fea_results, X_data)
