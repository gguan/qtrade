import pandas as pd
import talib as ta
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import StandardScaler
from qtrade.core.commission import PercentageCommission
from qtrade.env import TradingEnv


if __name__ == "__main__":
    """Load and process data"""
    df = pd.read_csv('examples/data/XAUUSD_15m.csv', parse_dates=True, index_col='Timestamp')

    # Calculate technical indicators
    df['Rsi'] = ta.RSI(df['Close'], timeperiod=14)
    df['Diff'] = df['Close'].diff()
    df.dropna(inplace=True)

    # Normalize technical indicators
    scaler = StandardScaler()

    df[['Rsi', 'Diff', 'Price']] = scaler.fit_transform(df[['Rsi', 'Diff', 'Close']])
    
    commission = PercentageCommission(0.001)
    env = TradingEnv(data=df, window_size=10, max_steps=550, verbose=False, 
                     cash=3000,
                     commission=commission, 
                     random_start=True,
                     )
    obs = env.reset()


    # Initialize the model
    monitor_env = Monitor(env, filename='monitor.csv', info_keywords=('equity', 'total_trades'))
    model = PPO("MlpPolicy", monitor_env, verbose=1)

    # Create evaluation callback to evaluate the model during training and save the best performing model
    eval_callback = EvalCallback(
        monitor_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/',
        eval_freq=50000,  # Evaluate every 50000 steps
        deterministic=True,
        render=False,
        verbose=1
    )

    # Load the best performing model
    # model = PPO.load("./logs/best_model/best_model.zip", env=env)

    # Start training the model
    model.learn(total_timesteps=500000, callback=eval_callback)

    # Evaluate the model and render at each step
    
    obs, _ = env.reset()
    for _ in range(400):
        env.render('human')  # Render at each step
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.plot()




