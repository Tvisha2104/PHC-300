

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


class MPPT_ANN:
    def __init__(self):
        self.model    = None
        self.scaler_X = StandardScaler()

        self.history  = None

    def generate_training_data(self, n_samples=5000):
        """
        Generate synthetic training data for MPPT.
        Features : [Irradiance, Temperature, V_mpp, I_mpp]
        Target   : Optimal duty cycle (V_mpp / V_oc) — physics-correct
        """
        np.random.seed(42)

        irradiance  = np.random.uniform(200, 1000, n_samples)
        temperature = np.random.uniform(-20,  60,  n_samples)

        Voc      = 21.5
        Isc      = 5.0
        G_norm   = irradiance / 1000
        T_factor = 1 - 0.004 * (temperature - 25)

        V_mpp = 0.77 * Voc * T_factor        
        I_mpp = Isc  * G_norm * T_factor       

      
        duty_cycle  = V_mpp / Voc
        duty_cycle  = np.clip(duty_cycle, 0.1, 0.9)
        duty_cycle += np.random.normal(0, 0.01, n_samples)  
        duty_cycle  = np.clip(duty_cycle, 0.1, 0.9)

        X = np.column_stack([irradiance, temperature, V_mpp, I_mpp])
        y = duty_cycle.reshape(-1, 1)
        return X, y

    def build_model(self, input_dim=4):
        """
        ANN for MPPT: 4 inputs → hidden layers → 1 output (duty cycle).
        """
       
        reg = keras.regularizers.l2(1e-4)

        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64,  activation='relu', kernel_regularizer=reg, name='hidden1'),
            layers.Dense(128, activation='relu', kernel_regularizer=reg, name='hidden2'),
            layers.Dense(64,  activation='relu', kernel_regularizer=reg, name='hidden3'),
            layers.Dense(32,  activation='relu', kernel_regularizer=reg, name='hidden4'),
       
            layers.Dense(1, activation='sigmoid', name='output'),
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss='mse',
            metrics=['mae']
        )
        return model

    def train(self, X, y, epochs=200, batch_size=128, validation_split=0.2):
        """Train the ANN model."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42)


        X_train_s = self.scaler_X.fit_transform(X_train)
        X_val_s   = self.scaler_X.transform(X_val)

        self.model = self.build_model()

        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=25, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)

        self.history = self.model.fit(
            X_train_s, y_train,     
            validation_data=(X_val_s, y_val),
            epochs=epochs,
            batch_size=batch_size,   
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        print("\nTraining completed!")
        print(f"Best validation loss : {min(self.history.history['val_loss']):.6f}")
        print(f"Final training loss  : {self.history.history['loss'][-1]:.6f}")

    def predict_duty_cycle(self, irradiance, temperature, v_mpp, i_mpp):
        """
        Predict optimal duty cycle for given conditions.

        FIX 5: arguments changed from (irr, temp, voltage, current) to
        (irr, temp, v_mpp, i_mpp) to match the features the model was trained on.
        inverse_transform removed (no scaler_y).
        """
        X_input  = np.array([[irradiance, temperature, v_mpp, i_mpp]])
        X_scaled = self.scaler_X.transform(X_input)
        y_pred   = self.model.predict(X_scaled, verbose=0)
        return float(np.clip(y_pred[0, 0], 0.1, 0.9))

    def plot_training_history(self):
        """Plot training and validation loss/MAE."""
        if self.history is None:
            print("No training history available!")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('ANN-MPPT Training History', fontsize=14, fontweight='bold')

        ax1.plot(self.history.history['loss'],     label='Training Loss',   linewidth=2, color='#1D9E75')
        ax1.plot(self.history.history['val_loss'], label='Validation Loss', linewidth=2, color='#D85A30')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss', fontweight='bold')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history['mae'],     label='Training MAE',   linewidth=2, color='#378ADD')
        ax2.plot(self.history.history['val_mae'], label='Validation MAE', linewidth=2, color='#7F77DD')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('Mean Absolute Error')
        ax2.set_title('Training and Validation MAE', fontweight='bold')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('./mppt_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Training history plot saved → mppt_training_history.png")

    def save_model(self, filepath='./mppt_model.keras'):
        """Save the trained model."""
        if self.model is None:
            print("No model to save!")
            return
        self.model.save(filepath)
        print(f"Model saved → {filepath}")

    def load_model(self, filepath='./mppt_model.keras'):
        """Load a trained model."""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")


def simulate_mppt_control(mppt_ann):
    """
    Simulate MPPT control under varying conditions during drone flight.

    FIX 6: simulation now derives V_mpp and I_mpp from physics (matching the
    training feature space) instead of using random voltage/current which
    would be out-of-distribution for the trained model.
    """
    print("\n" + "="*60)
    print("MPPT SIMULATION — Drone Flight Scenario")
    print("="*60)

    Voc = 21.5
    Isc = 5.0

    time_steps = 100
    time = np.linspace(0, 60, time_steps)

  
    irradiance  = 800 + 200 * np.sin(2 * np.pi * time / 30) \
                  + np.random.normal(0, 30, time_steps)
    irradiance  = np.clip(irradiance, 200, 1000)

   
    temperature = 25 + 8.0 * (1 - np.exp(-time / 30)) \
                  + np.random.normal(0, 1.5, time_steps)


    T_factor = 1 - 0.004 * (temperature - 25)
    V_mpp_arr = 0.77 * Voc * T_factor
    I_mpp_arr = Isc * (irradiance / 1000) * T_factor

    duty_cycles  = []
    power_output = []

    for i in range(time_steps):
        dc = mppt_ann.predict_duty_cycle(
            irradiance[i], temperature[i], V_mpp_arr[i], I_mpp_arr[i])
        duty_cycles.append(dc)

        mppt_eff = 0.85 + 0.12 * (1 - abs(dc - 0.77) / 0.77)
        power    = V_mpp_arr[i] * I_mpp_arr[i] * mppt_eff
        power_output.append(max(0.0, power))

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ANN-MPPT Flight Simulation', fontsize=15, fontweight='bold')

    ax1.plot(time, irradiance, color='#E8910A', linewidth=2)
    ax1.set_ylabel('Irradiance (W/m²)'); ax1.set_title('Solar Irradiance', fontweight='bold')
    ax1.grid(True, alpha=0.3)

    ax2.plot(time, temperature, color='#D85A30', linewidth=2)
    ax2.set_ylabel('Temperature (°C)'); ax2.set_title('Panel Temperature', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    ax3.plot(time, duty_cycles, color='#378ADD', linewidth=2)
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Duty Cycle')
    ax3.set_ylim(0, 1)
    ax3.set_title('ANN-Predicted Optimal Duty Cycle', fontweight='bold')
    ax3.grid(True, alpha=0.3)

    ax4.plot(time, power_output, color='#1D9E75', linewidth=2)
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Power (W)')
    ax4.set_title('Power Output (MPPT-corrected)', fontweight='bold')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('./mppt_simulation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Simulation plot saved → mppt_simulation.png")

    print(f"\nAverage Power Output : {np.mean(power_output):.2f} W")
    print(f"Peak Power Output    : {np.max(power_output):.2f} W")


def main():
    print("="*60)
    print("ANN-Based MPPT for Drone Solar Panels  [FIXED]")
    print("="*60)

    mppt = MPPT_ANN()

   
    print("\n1. Generating training data...")
    X, y = mppt.generate_training_data(n_samples=5000)
    print(f"   {X.shape[0]} samples | Features: Irradiance, Temperature, V_mpp, I_mpp")
    print(f"   Target: Optimal Duty Cycle (V_mpp / V_oc)")


    print("\n2. Training ANN model...")
    mppt.train(X, y, epochs=200, batch_size=128)


    print("\n3. Plotting training history...")
    mppt.plot_training_history()


    print("\n4. Saving model...")
    mppt.save_model()

    
    print("\n5. Testing predictions...")
    Voc, Isc = 21.5, 5.0

    def test(label, irr, temp):
        T_f   = 1 - 0.004 * (temp - 25)
        v_mpp = 0.77 * Voc * T_f
        i_mpp = Isc * (irr / 1000) * T_f
        dc    = mppt.predict_duty_cycle(irr, temp, v_mpp, i_mpp)
        print(f"   {label}: irr={irr} W/m², temp={temp}°C → duty={dc:.4f}")

    test("High irradiance, normal temp", 900, 25)
    test("Low irradiance, cold temp",    300, -10)
    test("Medium irradiance, hot temp",  600,  45)


    print("\n6. Running flight simulation...")
    simulate_mppt_control(mppt)

    print("\n" + "="*60)
    print("MPPT ANN Training and Simulation Complete!")
    print("="*60)
    print("\nGenerated files:")
    print("  - mppt_model.keras")
    print("  - mppt_training_history.png")
    print("  - mppt_simulation.png")


if __name__ == "__main__":
    main()
