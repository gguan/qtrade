import pandas as pd
import pandas_ta as ta
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import StandardScaler
from qtrade.core.commission import PercentageCommission
from qtrade.env import TradingEnv


if __name__ == "__main__":
    """加载并处理数据"""
    df = pd.read_csv('examples/data/XAUUSD_15m.csv', parse_dates=True, index_col='timestamp')

    # 计算技术指标
    df['rsi'] = ta.rsi(df['close'], length=14)
    df['diff'] = df['close'].diff()
    df.dropna(inplace=True)

    # 归一化技术指标
    scaler = StandardScaler()

    df[['rsi', 'diff', 'price']] = scaler.fit_transform(df[['rsi', 'diff', 'close']])
    
    features = ['price', 'diff', 'rsi']
    commission = PercentageCommission(0.001)
    env = TradingEnv(data=df, window_size=10, features=features, max_steps=550, verbose=False, 
                     cash=3000,
                     commission=commission, 
                     random_start=True,
                     )
    obs = env.reset()


    # 初始化模型
    monitor_env = Monitor(env, filename='monitor.csv',info_keywords=('equity', 'total_trades'))
    model = PPO("MlpPolicy", monitor_env, verbose=1)

    # 创建评估回调，用于在训练期间评估模型并保存表现最好的模型
    eval_callback = EvalCallback(
        monitor_env,
        best_model_save_path='./logs/best_model/',
        log_path='./logs/',
        eval_freq=50000,  # 每隔多少步进行一次评估
        deterministic=True,
        render=False,
        verbose=1
    )

    # 加载表现最好的模型
    # model = PPO.load("./logs/best_model/best_model.zip", env=env)

    # 开始训练模型
    model.learn(total_timesteps=500000, callback=eval_callback)

    # 评估模型，在评估过程中每步调用 env.render()
    
    obs, _ = env.reset()
    for _ in range(400):
        env.render('human')           # 每步渲染
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.plot()
    
    
 

