import yfinance as yh
import pandas as pd
import matplotlib.pyplot as plt
import polars as pl
import numpy as np

def bollinger_band(df, window_size, std_dev, plot_):
    df = df.with_columns([
        pl.col("Close").rolling_mean(window_size).alias("moving_avg"),
        pl.col("Close").rolling_std(window_size).alias("rolling_std_dev")
    ])

    df = df.with_columns([
        (pl.col("moving_avg") + std_dev * pl.col("rolling_std_dev")).alias("upper_band"),
        (pl.col("moving_avg") - std_dev * pl.col("rolling_std_dev")).alias("lower_band")
    ])

    df = df.with_columns([
        pl.when((pl.col("Close") < pl.col("lower_band"))).then(pl.lit(-1))
        .when(pl.col("Close") > pl.col("upper_band")).then(pl.lit(1))
        .otherwise(pl.lit(0)).alias("signal_bollinger"+"_"+str(window_size)+"_"+str(std_dev))
    ])
    if plot_== False:
        df = df.drop(['moving_avg', 'rolling_std_dev', 'upper_band', 'lower_band'])
    
    return df

def bollinger_band_comb(df):
    df = bollinger_band(df, 12, 1, False)
    df = bollinger_band(df, 12, 2, False)
    df = bollinger_band(df, 20, 1, False)
    df = bollinger_band(df, 20, 2, False)
    return df

def calculate_rsi(df, period, thr_h, thr_l, plot_):
    df = df.with_columns(
        (pl.col('Close').diff().alias("price_change"))
    )
    
    df = df.with_columns(
        pl.when(pl.col("price_change") > 0).then(pl.col("price_change")).otherwise(0).alias("gain"),
        pl.when(pl.col("price_change") < 0).then(-pl.col("price_change")).otherwise(0).alias("loss")
    )

    df = df.with_columns(
        pl.col("gain").rolling_mean(period).alias("avg_gain"),
        pl.col("loss").rolling_mean(period).alias("avg_loss")
    )
    
    df = df.with_columns(
        (pl.col("avg_gain") / pl.col("avg_loss")).alias("RS")
    )
    
    df = df.with_columns(
        (100 - (100 / (1 + pl.col("RS")))).alias("RSI"+"_"+str(period)+"_"+str(thr_l)+"_"+str(thr_h))
    )
    
    df = df.with_columns(
        pl.when(pl.col("RSI"+"_"+str(period)+"_"+str(thr_l)+"_"+str(thr_h)) < thr_l).then(1) 
        .when(pl.col("RSI"+"_"+str(period)+"_"+str(thr_l)+"_"+str(thr_h)) > thr_h).then(-1)  
        .otherwise(0).alias("signal_rsi"+"_"+str(period)+"_"+str(thr_l)+"_"+str(thr_h))     
    )

    if plot_ == False:
        df = df.drop(['price_change','gain', 'loss', 'avg_gain', 'avg_loss', "RS"])
    
    return df

def rsi_comb(df):
    df = calculate_rsi(df, 14, 70, 30, False)
    df = calculate_rsi(df, 14, 80, 20, False)
    df = calculate_rsi(df, 8, 70, 30, False)
    df = calculate_rsi(df, 8, 80, 20, False)
    return df

def calculate_stochastic_oscillator(df, period, smooth_period, plot_):

    df = df.with_columns(
        pl.col('Low').rolling_min(window_size=period, min_periods=1).alias('Lowest_Low')
    )
    df = df.with_columns(
        pl.col('High').rolling_max(window_size=period, min_periods=1).alias('Highest_High')
    )
    
    df = df.with_columns(
        ((pl.col("Close") - pl.col("Lowest_Low")) / (pl.col("Highest_High") - pl.col("Lowest_Low")) * 100)
        .alias("Stochastic_K"+"_"+str(period)+"_"+str(smooth_period))
    )

    df = df.with_columns(
        pl.col("Stochastic_K"+"_"+str(period)+"_"+str(smooth_period)).rolling_mean(window_size=smooth_period).alias("Stochastic_D"+"_"+str(period)+"_"+str(smooth_period))
    )
    if plot_== False:
        df = df.drop(['Lowest_Low', 'Highest_High'])
    return df

def stochastic_oscillator_comb(df):
    df = calculate_stochastic_oscillator(df, period=14, smooth_period=3, plot_=False)
    df = calculate_stochastic_oscillator(df, period=14, smooth_period=5, plot_=False)
    df = calculate_stochastic_oscillator(df, period=8, smooth_period=3, plot_=False)
    df = calculate_stochastic_oscillator(df, period=8, smooth_period=5, plot_=False)
    return df

def calculate_atr(df, period, atr_multiplier, plot_):
    df = df.with_columns([
        (pl.col("High") - pl.col("Low")).alias("TR1"),
        (pl.col("High") - pl.col("Close").shift(1)).abs().alias("TR2"),
        (pl.col("Low") - pl.col("Close").shift(1)).abs().alias("TR3")
    ])
    
    df = df.with_columns(
        pl.max_horizontal('TR1', 'TR2', 'TR3').alias('True_Range')
    )
    
    df = df.with_columns(
        pl.col("True_Range").rolling_mean(window_size=period, min_periods=1).alias("ATR")
    )

    df = df.with_columns([
        (
            pl.when((pl.col("Close") > (pl.col("Close").shift(1) + (pl.col("ATR") * atr_multiplier))))
            .then(1)
            .when((pl.col("Close") < (pl.col("Close").shift(1) - (pl.col("ATR") * atr_multiplier))))
            .then(-1)
            .otherwise(0)
            .alias("Signal")
    )])

    if plot_ == False:
        df = df.drop(['TR1', 'TR2', 'TR3', 'True_Range', 'ATR'])
        df = df.rename({"Signal": "signal"+ "_"+ "atr"+"_"+str(period)+"_"+str(atr_multiplier)})
    
    return df

def atr_comb(df):
    df = calculate_atr(df, 14, 1, plot_=False)
    df = calculate_atr(df, 14, 1.5, plot_=False)
    df = calculate_atr(df, 8, 1, plot_=False)
    df = calculate_atr(df, 8, 1.5, plot_=False)
    df = calculate_atr(df, 30, 1, plot_=False)
    df = calculate_atr(df, 30, 1.5, plot_=False)
    return df

def calculate_obv(df, price_threshold, plot_):
    df = df.with_columns([
        pl.lit(0).alias("OBV")
    ])
    
    df = df.with_columns(
        pl.when(pl.col("Close") > pl.col("Close").shift(1)) 
        .then(pl.col("Volume"))
        .when(pl.col("Close") < pl.col("Close").shift(1))  
        .then(-pl.col("Volume"))
        .otherwise(0)
        .alias("OBV_Change") 
    )
    
    df = df.with_columns(
        pl.col("OBV_Change").cum_sum().alias("OBV")
    )

    df = df.with_columns(
        pl.when(
            (pl.col("OBV") > pl.col("OBV").shift(1))  
        )
        .then(1).otherwise(-1).alias("OBV_trend")
    )

    df = df.with_columns(
        pl.when(
            (pl.col("OBV") > pl.col("OBV").shift(1)) 
            & (pl.col("Close") > pl.col("Close").shift(1) * (1 + price_threshold / 100))  
        )
        .then(1)  
        .when(
            (pl.col("OBV") < pl.col("OBV").shift(1))  
            & (pl.col("Close") < pl.col("Close").shift(1) * (1 - price_threshold / 100))  
        )
        .then(-1)
        .otherwise(0)  
        .alias("Signal")
    )
    
    if plot_ == False:
        df = df.drop(['OBV', 'OBV_Change'])
        df = df.rename({"Signal": "signal"+ "_"+ "obv"+"_"+str(price_threshold)})
    
    return df

def obv_comb(df):
    df = calculate_obv(df, 1, False)
    df = calculate_obv(df, 3, False)
    df = calculate_obv(df, 4, False)
    return df

def calculate_williams_r(df, period, plot_):

    df = df.with_columns([
        pl.col("High").rolling_max(window_size=period, min_periods=1).alias("Highest_High"),
        pl.col("Low").rolling_min(window_size=period, min_periods=1).alias("Lowest_Low")
    ])
    df = df.with_columns(
        ((pl.col("Highest_High") - pl.col("Close")) / 
         (pl.col("Highest_High") - pl.col("Lowest_Low")) * -100).alias("Williams_R")
    )

    if plot_== False:
        df = df.rename({"Williams_R": "Williams_R"+"_"+str(period)}).drop(['Highest_High', 'Lowest_Low'])
    return df

def williams_comb(df):
    df = calculate_williams_r(df, 14, False)
    df = calculate_williams_r(df, 8, False)
    df = calculate_williams_r(df, 30, False)
    return df

def calculate_adx(df, period, plot_):
    df = df.with_columns([
        (pl.col("High") - pl.col("Low")).alias("TR1"),
        (pl.col("High") - pl.col("Close").shift(1)).abs().alias("TR2"),
        (pl.col("Low") - pl.col("Close").shift(1)).abs().alias("TR3")
    ])
    df = df.with_columns(
        pl.max_horizontal('TR1', 'TR2', 'TR3').alias('True_Range')
    )
    df = df.with_columns([
        pl.when(pl.col("High") -pl.col("High").shift(1) > pl.col("Low").shift(1) - pl.col("Low"))
          .then(pl.col("High") -pl.col("High").shift(1))
          .otherwise(0)
          .alias("Positive_DM"),
        
        pl.when(pl.col("Low").shift(1) - pl.col("Low") > pl.col("High") - pl.col("High").shift(1))
          .then(pl.col("Low").shift(1) - pl.col("Low"))
          .otherwise(0)
          .alias("Negative_DM")
    ])

    df = df.with_columns([
         pl.col("Positive_DM").rolling_mean(window_size=period, min_periods=1).alias("Smoothed_Positive_DM"),
        pl.col("Negative_DM").rolling_mean(window_size=period, min_periods=1).alias("Smoothed_Negative_DM"),
        pl.col("True_Range").rolling_mean(window_size=period, min_periods=1).alias("Smoothed_True_Range")
    ])
    
    df = df.with_columns([
         (pl.col("Smoothed_Positive_DM") /pl.col("Smoothed_True_Range") * 100).alias("Positive_DI"),
        (pl.col("Smoothed_Negative_DM") / pl.col("Smoothed_True_Range") * 100).alias("Negative_DI")
    ])
    
    df = df.with_columns([
        (pl.col("Positive_DI") - pl.col("Negative_DI")).abs().rolling_mean(window_size=period, min_periods=1).alias("ADX")
    ])

    if plot_==False:
        df = df.drop(['TR1', 'TR2', 'TR3', 'True_Range',
                      'Positive_DM', 'Negative_DM', 'Smoothed_Positive_DM',
                      'Smoothed_Negative_DM', 'Smoothed_True_Range',
                      'Positive_DI', 'Negative_DI'])
        df = df.rename({"ADX": "adx"+ "_"+str(period)})
    
    return df

def adx_comb(df):
    df = calculate_adx(df, 14, False)
    df = calculate_adx(df, 8, False)
    df = calculate_adx(df, 30, False)
    return df

def calculate_aroon_pandas(df, period=14):

    df['Aroon_Up'] = df['High'].rolling(window=period, min_periods=1).apply(lambda x: period - x.argmax(), raw=False)
    df['Aroon_Down'] = df['Low'].rolling(window=period, min_periods=1).apply(lambda x: period - x.argmin(), raw=False)
    
    df['Aroon_Up_Percentage'] = (df['Aroon_Up'] / period) * 100
    df['Aroon_Down_Percentage'] = (df['Aroon_Down'] / period) * 100
    
    df['Aroon_Oscillator'] = df['Aroon_Up_Percentage'] - df['Aroon_Down_Percentage']
    
    return df

def convert_and_calculate_aroon(df, period, plot_):

    df_pandas = df.to_pandas()
    df_pandas = calculate_aroon_pandas(df_pandas, period)

    df_polars = pl.from_pandas(df_pandas)

    if plot_==False:
        df_polars = df_polars.rename({"Aroon_Oscillator": "Aroon_Oscillator"+ "_"+str(period)}).drop(['Aroon_Up', 'Aroon_Down', 'Aroon_Up_Percentage', 'Aroon_Down_Percentage'])
    
    return df_polars

def aroon_comb(df):
    df = convert_and_calculate_aroon(df, 14, False)
    df = convert_and_calculate_aroon(df, 8, False)
    df = convert_and_calculate_aroon(df, 30, False)
    return df

def calculate_atr2(df, period=14):

    df = df.with_columns([
        (pl.col("High") - pl.col("Low")).alias("TR1"),
        (pl.col("High") - pl.col("Close").shift(1)).abs().alias("TR2"),
        (pl.col("Low") - pl.col("Close").shift(1)).abs().alias("TR3")
    ])
    
    df = df.with_columns(
        pl.max_horizontal('TR1', 'TR2', 'TR3').alias('True_Range')
    )
    
    df = df.with_columns(
        pl.col("True_Range").rolling_mean(window_size=period, min_periods=1).alias("ATR")
    )
    
    return df

def calculate_keltner_channels(df, period=20, atr_multiplier=2):

    df = df.with_columns(
        pl.col("Close").ewm_mean(com=period-1).alias("Middle") 
    )
    
    df = calculate_atr2(df, period=14) 
    
    df = df.with_columns([
        (pl.col("Middle") + atr_multiplier * pl.col("ATR")).alias("Upper"),
        (pl.col("Middle") - atr_multiplier * pl.col("ATR")).alias("Lower")
    ])
    
    return df

def calculate_keltner_percent(df, period, atr_multiplier, plot_):

    df = calculate_keltner_channels(df, period=period, atr_multiplier=atr_multiplier)
    
    df = df.with_columns(
        ((pl.col("Close") - pl.col("Lower")) / (pl.col("Upper") - pl.col("Lower")) * 100).alias("Keltner%")
    )
    
    if plot_ == False:
        df = df.drop(['TR1', 'TR2', 'TR3', 'True_Range', 'ATR', 'Upper', 'Lower'])
        df = df.rename({"Keltner%": "Keltner"+ "_"+str(period)})
    
    return df


def calculate_keltner_channels(df, period=20, atr_multiplier=2):

    df = df.with_columns(
        pl.col("Close").ewm_mean(com=period-1).alias("Middle") 
    )
    
    df = calculate_atr2(df, period=14)  

    df = df.with_columns([
        (pl.col("Middle") + atr_multiplier * pl.col("ATR")).alias("Upper"),
        (pl.col("Middle") - atr_multiplier * pl.col("ATR")).alias("Lower")
    ])
    
    return df

def calculate_keltner_percent(df, period, atr_multiplier, plot_):

    df = calculate_keltner_channels(df, period=period, atr_multiplier=atr_multiplier)
    
    df = df.with_columns(
        ((pl.col("Close") - pl.col("Lower")) / (pl.col("Upper") - pl.col("Lower")) * 100).alias("Keltner%")
    )

    df = df.with_columns(
        pl.col("Keltner%").fill_null(0)
    )

    df = df.with_columns(
        pl.when(pl.col("Keltner%")>1000).then(50).when(pl.col("Keltner%")<-1000).then(50).otherwise(pl.col("Keltner%")).alias("Keltner%")
    )
    
    if plot_ == False:
        df = df.drop(['TR1', 'TR2', 'TR3', 'True_Range', 'ATR', 'Upper', 'Lower', 'Middle'])
        df = df.rename({"Keltner%": "Keltner"+ "_"+str(period)+"_"+str(atr_multiplier)})
    
    return df

def keltner_comb(df):
    df = calculate_keltner_percent(df, 20, 1.5, False)
    df = calculate_keltner_percent(df, 20, 2, False)
    df = calculate_keltner_percent(df, 20, 2.5, False)
    df = calculate_keltner_percent(df, 14, 1.5, False)
    df = calculate_keltner_percent(df, 14, 2, False)
    df = calculate_keltner_percent(df, 14, 2.5, False)
    df = calculate_keltner_percent(df, 8, 1.5, False)
    df = calculate_keltner_percent(df, 8, 2, False)
    df = calculate_keltner_percent(df, 8, 2.5, False)
    return df

def calculate_cmf(df, period, plot_):

    
    df = df.with_columns(
            (((pl.col("Close") - pl.col("Low")) - (pl.col("High") - pl.col("Close"))) /(pl.col("High") - pl.col("Low"))).alias("Money_Flow_Multiplier")
        )

    df = df.with_columns(
            (pl.col("Money_Flow_Multiplier") * pl.col("Volume")).alias("Money_Flow_Volume")
        )

    df = df.with_columns([
            pl.col("Money_Flow_Volume").rolling_sum(window_size=period, min_periods=1).alias("Cumulative_Money_Flow_Volume"),
            pl.col("Volume").rolling_sum(window_size=period, min_periods=1).alias("Cumulative_Volume")
        ])

    df = df.with_columns(
            ((pl.col("Cumulative_Money_Flow_Volume") / pl.col("Cumulative_Volume"))).fill_null(0).alias("CMF")
    )


    if plot_ == False:
        df = df.drop(['Money_Flow_Multiplier', 'Money_Flow_Volume', 'Cumulative_Money_Flow_Volume',
                     'Cumulative_Volume'])
        df = df.rename({"CMF": "CMF"+ "_"+str(period)})
    

    return df

def cmf_comb(df):
    df = calculate_cmf(df, 20, False)
    df = calculate_cmf(df, 14, False)
    df = calculate_cmf(df, 8, False)
    return df

def generate_orders(data, n_shares, initial_cash, clf, var):
    df_bitcoin = data
    df_bitcoin = df_bitcoin.with_columns(
            pl.Series("prediction", clf.predict(df_bitcoin.select(var)))
        )


    data_with_signals = df_bitcoin.to_pandas()
    data_with_signals = data_with_signals.set_index('Date')

    df_order = pd.DataFrame()
    df_order['Date'] = data_with_signals.index
    df_order = df_order.set_index('Date')
    df_order['Close'] = data_with_signals.Close
    df_order['prediction'] = data_with_signals.prediction
    df_order['prediction_scaled'] = np.where(df_order['prediction']==2, 1,
                            np.where(df_order['prediction']==1, -1, 0))

    h_signal = 0
    signal_vec = []
    for i in range(len(df_order['prediction_scaled'])):
            if df_order['prediction_scaled'].iloc[i]==1:
                h_signal +=1
                if h_signal > 1:
                    h_signal = 1

            if df_order['prediction_scaled'].iloc[i]==-1:
                h_signal += -1
                if h_signal < 0:
                    h_signal = 0

            if df_order['prediction_scaled'].iloc[i]==0:
                h_signal += 0
            signal_vec.append(h_signal)
    df_order['signal'] = signal_vec
    orders_vec = [df_order['signal'].iloc[0]]
    for i in range(1, len(df_order['signal'])):
            if df_order['signal'].iloc[i-1]==1:
                if df_order['prediction_scaled'].iloc[i]==1 or df_order['prediction_scaled'].iloc[i]==0:
                    orders_vec.append(0)
                else:
                    orders_vec.append(-1)
            #if df_order['signal'].iloc[i-1]==-1:
            #    if df_order['prediction_scaled'].iloc[i]==-1 or df_order['prediction_scaled'].iloc[i]==0:
            #        orders_vec.append(0)
            #    else:
            #        orders_vec.append(1)
            
            if df_order['signal'].iloc[i-1]==0:
                if df_order['prediction_scaled'].iloc[i]==-1 or df_order['prediction_scaled'].iloc[i]==0:
                    orders_vec.append(0)
                elif df_order['prediction_scaled'].iloc[i]==1:
                    orders_vec.append(1)
                #else:
                #    orders_vec.append(-1)

    df_order['order'] = orders_vec

    df_order['order'] = orders_vec
    df_order['order_cash'] = n_shares
    df_order['holdings'] = df_order['signal']*df_order['order_cash']
    #df_order['Cash'] = initial_cash
    df_order.at[df_order.index[0], 'Cash'] = initial_cash - df_order.iloc[0]['holdings']*df_order.iloc[0]['Close']

    for i in range(1, len(df_order)):
        order = df_order.iloc[i]['order']
        close_price = df_order.iloc[i]['Close']
        order_cash = df_order.iloc[i]['order_cash']
        
        if order == 1:  # Buy
            df_order.loc[df_order.index[i], 'Cash'] = df_order.iloc[i-1]['Cash'] - close_price*order*order_cash
        elif order == -1:  # Sell
            df_order.loc[df_order.index[i], 'Cash'] = df_order.iloc[i-1]['Cash'] - close_price*order*order_cash
        else:  # No action
            df_order.loc[df_order.index[i], 'Cash'] = df_order.iloc[i-1]['Cash']
            
    df_order['portfolio_value'] = df_order['Cash'] + (df_order['holdings']*df_order['Close'])

    df_order['signal_2'] = 1
    df_order['order_2'] = 0
    df_order.at[df_order.index[0], 'order_2']= 1

    df_order['holdings_2'] = df_order['signal_2']*df_order['order_cash']
    df_order['Cash_2'] = initial_cash

    df_order.at[df_order.index[0], 'Cash_2'] = initial_cash - df_order.iloc[0]['holdings_2']*df_order.iloc[0]['Close']

    for i in range(1, len(df_order)):
        order = df_order.iloc[i]['order_2']
        close_price = df_order.iloc[i]['Close']
        
        if order == 1: 
            df_order.loc[df_order.index[i], 'Cash_2'] = df_order.iloc[i-1]['Cash_2'] - close_price
        elif order == -1:  
            df_order.loc[df_order.index[i], 'Cash_2'] = df_order.iloc[i-1]['Cash_2'] + close_price
        else:  
            df_order.loc[df_order.index[i], 'Cash_2'] = df_order.iloc[i-1]['Cash_2']

    df_order['portfolio_value_2'] = df_order['Cash_2'] + (df_order['holdings_2']*df_order['Close'])
    return df_order

def portfolio_stats(portfolio_value):  		  	   		 	   			  		 			 	 	 		 		 	
    daily_rets = (portfolio_value / portfolio_value.shift(1)) - 1  		  	   		 	   			  		 			 	 	 		 		 	
    daily_rets = daily_rets[1:]  		  	   		 	   			  		 			 	 	 		 		 		  	   		 	   			  		 			 	 	 		 		 	
    cumulative_return = (portfolio_value[-1]/portfolio_value[0])- 1.0
    mean_daily_ret = daily_rets.mean()  		  	   		 	   			  		 			 	 	 		 		 	
    std_daily_ret = daily_rets.std()   	   		 	   			  		 			 	 	 		 		 	
    return cumulative_return, std_daily_ret, mean_daily_ret

def generate_testing_orders(data, symbol, clf, var, n_shares, initial_cash):
    info = yh.Ticker(symbol)
    data = pl.DataFrame(info.history(period="1Y")[['Open', 'High', 'Low', 'Close', 'Volume']].reset_index())
    data = data.with_columns(pl.lit(symbol).alias("Symbol"))

    data = bollinger_band_comb(data)
    data = rsi_comb(data)
    data = stochastic_oscillator_comb(data)
    data = atr_comb(data)
    data = obv_comb(data)
    data = williams_comb(data)
    data = adx_comb(data)
    data = aroon_comb(data)
    data = keltner_comb(data)
    data = cmf_comb(data)

    start_date = data[30, "Date"]
    data = data.filter(pl.col("Date") >= start_date)

    df_orders = generate_orders(data, n_shares, initial_cash, clf, var)

    return df_orders

def plot_signal(df_orders, symbol):
    data_with_signals = df_orders

    plt.figure(figsize=(10,6))
    plt.plot(data_with_signals['Close'], label='Close Price', color='purple', alpha=0.6)

    # Plot Buy and Sell signals
    buy_signals = data_with_signals[data_with_signals['order'] == 1]
    sell_signals = data_with_signals[data_with_signals['order'] == -1]

    plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', alpha=1, s=100)
    plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', alpha=1, s=100)

    plt.title(symbol)
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()
