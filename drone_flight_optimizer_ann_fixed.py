

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



class MPPT_ANN:
    def __init__(self):
        self.model    = None
        self.scaler_X = StandardScaler()
       
        self.history  = None

    def generate_training_data(self, n_samples=5000):
       
        np.random.seed(42)
        irradiance  = np.random.uniform(200, 1000, n_samples)
        temperature = np.random.uniform(-20,  60,  n_samples)
        
        Voc      = 21.5
        Isc      = 5.0
        G_norm   = irradiance / 1000
        T_factor = 1 - 0.004 * (temperature - 25)

        V_mpp = 0.77 * Voc * T_factor                   
        I_mpp = Isc  * G_norm * T_factor              

        duty_cycle = V_mpp / Voc
        duty_cycle = np.clip(duty_cycle, 0.1, 0.9)
        duty_cycle += np.random.normal(0, 0.01, n_samples)  
        duty_cycle = np.clip(duty_cycle, 0.1, 0.9)

     
        X = np.column_stack([irradiance, temperature, V_mpp, I_mpp])
        y = duty_cycle.reshape(-1, 1)
        return X, y

    def build_model(self, input_dim=4):
     
        reg = keras.regularizers.l2(1e-4)
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(64,  activation='relu', kernel_regularizer=reg, name='hidden1'),
            layers.Dense(128, activation='relu', kernel_regularizer=reg, name='hidden2'),
            layers.Dense(64,  activation='relu', kernel_regularizer=reg, name='hidden3'),
            layers.Dense(32,  activation='relu', kernel_regularizer=reg, name='hidden4'),
            layers.Dense(1,   activation='sigmoid', name='output'),
        ])
     
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=3e-4),
            loss='mse', metrics=['mae'])
        return model

    def train(self, X, y, epochs=200, batch_size=128, validation_split=0.2):
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
            epochs=epochs, batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=0)                

        best_loss = min(self.history.history['val_loss'])
        print(f"  Training done — best val_loss: {best_loss:.6f}  "
              f"({len(self.history.history['loss'])} epochs)")

    def predict_duty_cycle(self, irradiance, temperature, v_mpp, i_mpp):
        X_input  = np.array([[irradiance, temperature, v_mpp, i_mpp]])
        X_scaled = self.scaler_X.transform(X_input)
        y_pred   = self.model.predict(X_scaled, verbose=0)
        return float(np.clip(y_pred[0, 0], 0.1, 0.9))

    def plot_training_history(self, save_path='training_history.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('ANN-MPPT Training History', fontsize=14, fontweight='bold')

        ax1.plot(self.history.history['loss'],     label='Train Loss',
                 linewidth=2, color='#1D9E75')
        ax1.plot(self.history.history['val_loss'], label='Val Loss',
                 linewidth=2, color='#D85A30')
        ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss (MSE)')
        ax1.set_title('Training and Validation Loss')
        ax1.legend(); ax1.grid(True, alpha=0.3)

        ax2.plot(self.history.history['mae'],     label='Train MAE',
                 linewidth=2, color='#378ADD')
        ax2.plot(self.history.history['val_mae'], label='Val MAE',
                 linewidth=2, color='#7F77DD')
        ax2.set_xlabel('Epoch'); ax2.set_ylabel('MAE')
        ax2.set_title('Training and Validation MAE')
        ax2.legend(); ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Training history plot saved → {save_path}")

    def save_model(self, filepath='./mppt_model.keras'):
        if self.model:
            self.model.save(filepath)
            print(f"  Model saved → {filepath}")

class SolarModel:
    def __init__(self, mppt_ann):
        self.mppt       = mppt_ann
     
        self.panel_area = 0.15
        self.efficiency = 0.20      
        self.Voc        = 21.5
        self.Isc        = 5.0
        self.temp_coeff = 0.004

    def get_power(self, irradiance, temperature, panel_angle_deg, cloud_cover=0.0):
        T_factor = 1 - self.temp_coeff * (temperature - 25)
        V_mpp    = 0.77 * self.Voc  * T_factor
        I_mpp    = self.Isc * (irradiance / 1000) * T_factor

        duty = self.mppt.predict_duty_cycle(irradiance, temperature, V_mpp, I_mpp)

        angle_factor = np.cos(np.radians(panel_angle_deg))
        cloud_factor = 1 - cloud_cover * 0.85

    
        raw_power = (irradiance * self.panel_area
                     * self.efficiency * T_factor
                     * angle_factor * cloud_factor)

      
        mppt_efficiency = 0.85 + 0.12 * (1 - abs(duty - 0.77) / 0.77)
        mppt_power      = min(raw_power * mppt_efficiency, raw_power)
        return max(0.0, mppt_power), duty



class DronePowerModel:
    def __init__(self):
    
        self.base_hover_power = 12.0

    def total_power(self, speed_ms, altitude_m, payload_g=0):
        payload_factor = 1 + (payload_g / 1000) * 0.4
        alt_factor     = 1 + (altitude_m / 400) * 0.08
        drag_power     = 0.5 * 1.225 * 0.05 * (speed_ms ** 3)
        hover          = self.base_hover_power * payload_factor * alt_factor
        return hover * 0.85 + drag_power

class BatteryModel:
    def __init__(self, capacity_wh=80, initial_soc=1.0):
        self.capacity_wh   = capacity_wh
        self.soc           = initial_soc
        self.min_soc       = 0.15
        self.charge_eff    = 0.95
        self.discharge_eff = 0.92

    def update(self, net_power_w, dt_s):
        energy_wh = net_power_w * dt_s / 3600
        if energy_wh > 0:
            self.soc += (energy_wh * self.charge_eff) / self.capacity_wh
        else:
            self.soc += (energy_wh / self.discharge_eff) / self.capacity_wh
        self.soc = float(np.clip(self.soc, 0, 1))

    def is_depleted(self):
        return self.soc <= self.min_soc

class FlightOptimizer:
    def __init__(self, mppt_ann):
        self.solar = SolarModel(mppt_ann)
        self.drone = DronePowerModel()

    def optimize_speed(self, irr, temp, alt, payload_g, angle, cloud):
        def cost(s):
            solar_pwr, _ = self.solar.get_power(irr, temp, angle, cloud)
            demand       = self.drone.total_power(s[0], alt, payload_g)
            return abs(demand - solar_pwr)
        res = minimize(cost, x0=[6.0], bounds=[(1.0, 20.0)], method='L-BFGS-B')
        return round(float(res.x[0]), 2)

    def optimize_panel_angle(self, irr, temp, cloud, alt=100, payload_g=200):
       
        if irr >= 800:
            base_angle = 20.0
        elif irr >= 400:
            base_angle = 20.0 + (800 - irr) / 400 * 25.0   
        else:
            base_angle = 45.0 + (400 - irr) / 400 * 10.0   


        best_angle = base_angle
        best_power = -np.inf
        for delta in np.linspace(-10, 10, 21):
            candidate = float(np.clip(base_angle + delta, 5.0, 85.0))
            p, _ = self.solar.get_power(irr, temp, candidate, cloud)
            if p > best_power:
                best_power = p
                best_angle = candidate
        return round(best_angle, 2)

    def full_optimization(self, conditions):
        irr   = conditions.get('irradiance',  800)
        temp  = conditions.get('temperature',  25)
        alt   = conditions.get('altitude',    100)
        pay   = conditions.get('payload_g',   200)
        cloud = conditions.get('cloud_cover', 0.1)

        opt_angle       = self.optimize_panel_angle(irr, temp, cloud, alt, pay)
        opt_speed       = self.optimize_speed(irr, temp, alt, pay, opt_angle, cloud)
        solar_pwr, duty = self.solar.get_power(irr, temp, opt_angle, cloud)
        demand          = self.drone.total_power(opt_speed, alt, pay)

        return {
            'optimal_speed_ms':    opt_speed,
            'optimal_panel_angle': opt_angle,
            'ann_duty_cycle':      round(duty, 4),
            'solar_power_w':       round(solar_pwr, 2),
            'power_demand_w':      round(demand, 2),
            'net_power_w':         round(solar_pwr - demand, 2),
            'solar_surplus':       solar_pwr >= demand,
        }

def simulate_flight(conditions, mppt_ann, duration_s=3600, dt=10):
    solar_model = SolarModel(mppt_ann)
    drone_model = DronePowerModel()
    battery     = BatteryModel(conditions.get('battery_wh', 80))
    optimizer   = FlightOptimizer(mppt_ann)

    opt   = optimizer.full_optimization(conditions)
    speed = opt['optimal_speed_ms']
    angle = opt['optimal_panel_angle']

    time_arr, solar_arr, demand_arr, soc_arr, duty_arr = [], [], [], [], []

    for t in range(0, duration_s, dt):
        irr = conditions['irradiance'] * (
            0.85 + 0.15 * np.sin(2 * np.pi * t / 600)
            + np.random.normal(0, 0.03))
        irr = float(np.clip(irr, 100, 1200))

   
        base_temp = conditions.get('temperature', 25)
        temp = base_temp + 8.0 * (1 - np.exp(-t / 1800)) \
               + np.random.normal(0, 1.0)

        cloud = conditions.get('cloud_cover', 0.05)

        s_pwr, duty = solar_model.get_power(irr, temp, angle, cloud)
        d_pwr       = drone_model.total_power(
            speed,
            conditions.get('altitude', 100),
            conditions.get('payload_g', 200))

        battery.update(s_pwr - d_pwr, dt)

        time_arr.append(t / 60)
        solar_arr.append(s_pwr)
        demand_arr.append(d_pwr)
        soc_arr.append(battery.soc * 100)
        duty_arr.append(duty)

        if battery.is_depleted():
            print(f"  Battery depleted at {t/60:.1f} min")
            break

    return (np.array(time_arr),  np.array(solar_arr),
            np.array(demand_arr), np.array(soc_arr),  np.array(duty_arr))

def plot_results(time, solar, demand, soc, duty,
                 save_path='flight_optimization_ann.png'):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle('Solar Drone — ANN-MPPT Flight Optimization',
                 fontsize=15, fontweight='bold')

    axes[0].plot(time, solar,  color='#1D9E75', lw=2, label='Solar power (W)')
    axes[0].plot(time, demand, color='#D85A30', lw=2,
                 linestyle='--', label='Power demand (W)')
    axes[0].fill_between(time, solar, demand,
                         where=solar >= demand,
                         alpha=0.2, color='#1D9E75', label='Solar surplus')
    axes[0].fill_between(time, solar, demand,
                         where=solar < demand,
                         alpha=0.2, color='#D85A30', label='Battery drain')
    axes[0].set_ylabel('Power (W)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Power balance (ANN-MPPT controlled)')

    axes[1].plot(time, soc, color='#378ADD', lw=2)
    axes[1].axhline(y=15, color='#E24B4A', ls='--', lw=1, label='Min SoC 15%')
    axes[1].fill_between(time, soc, 15, where=soc >= 15,
                         alpha=0.15, color='#378ADD')
    axes[1].set_ylabel('Battery SoC (%)')
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Battery state of charge')

    axes[2].plot(time, duty, color='#7F77DD', lw=2)
    axes[2].set_ylabel('ANN Duty Cycle')
    axes[2].set_xlabel('Time (min)')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('ANN-MPPT predicted duty cycle')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Flight optimization plot saved → {save_path}")



def main():
    print("=" * 60)
    print("  ANN-MPPT Solar Drone Flight Optimizer  [FIXED]")
    print("=" * 60)

   
    print("\n[1/4] Training ANN-MPPT model...")
    mppt = MPPT_ANN()
    X, y = mppt.generate_training_data(n_samples=5000)
    mppt.train(X, y, epochs=200, batch_size=128)
    mppt.save_model('./mppt_model.keras')

 
    print("\n[2/4] Saving training history plots...")
    mppt.plot_training_history(save_path='training_history.png')

    print("\n[3/4] Optimising flight and running simulation...")
    conditions = {
        'irradiance':  900,   
        'temperature':  28,
        'altitude':    100,
        'payload_g':   150,
        'battery_wh':   80,
        'cloud_cover':  0.05,
    }

    optimizer = FlightOptimizer(mppt)
    result    = optimizer.full_optimization(conditions)

    print(f"\n  Optimal speed      : {result['optimal_speed_ms']} m/s")
    print(f"  Optimal panel angle: {result['optimal_panel_angle']}°")
    print(f"  ANN duty cycle     : {result['ann_duty_cycle']}")
    print(f"  Solar power        : {result['solar_power_w']} W")
    print(f"  Power demand       : {result['power_demand_w']} W")
    print(f"  Net power          : {result['net_power_w']} W")
    print(f"  Solar surplus      : {result['solar_surplus']}")

    time, solar, demand, soc, duty = simulate_flight(
        conditions, mppt, duration_s=3600)

    print(f"\n  Avg solar power    : {np.mean(solar):.2f} W")
    print(f"  Avg power demand   : {np.mean(demand):.2f} W")
    print(f"  Avg ANN duty cycle : {np.mean(duty):.4f}")
    print(f"  Final battery SoC  : {soc[-1]:.1f}%")
    print(f"  Total flight time  : {time[-1]:.1f} min")

    # Step 4 — Save plots
    print("\n[4/4] Saving all plots...")
    plot_results(time, solar, demand, soc, duty,
                 save_path='flight_optimization_ann.png')

    print("\n" + "=" * 60)
    print("  All done! 3 files saved:")
    print("  - mppt_model.keras")
    print("  - training_history.png")
    print("  - flight_optimization_ann.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
