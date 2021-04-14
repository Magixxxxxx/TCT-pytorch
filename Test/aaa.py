import pandas as pd
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
 
 
 
 
# 读取数据，pd.read_csv默认生成DataFrame对象，需将其转换成Series对象
df = pd.read_csv('1.csv', encoding='GBK', index_col='date')
df.index = pd.to_datetime(df.index)  # 将字符串索引转换成时间索引
ts = df['people']  # 生成pd.Series对象
print(ts.head())
 
 
# 移动平均图
def draw_trend(timeseries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeseries.rolling(window=size).mean()
    # 对size个数据移动平均的方差
    rol_std = timeseries.rolling(window=size).std()
 
    timeseries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_std.plot(color='black', label='Rolling standard deviation')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
 
 
def draw_ts(timeseries):
    f = plt.figure(facecolor='white')
    timeseries.plot(color='blue')
    plt.show()
 
 
# Dickey-Fuller test:
def teststationarity(ts):
    dftest = adfuller(ts)
    # 对上述函数求得的值进行语义描述
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    return dfoutput
 
draw_trend(ts, 7)
# 原始数据、均值、方差
print(teststationarity(ts))
# Dickey-Fuller检验
# ts_log = np.log(ts)
# 对数变换
# ts_log.plot()
# plt.show()
# 查看变换后情况
# print(teststationarity(ts_log))
# Dickey-Fuller检验
# 移动平均
def draw_moving(timeSeries, size):
    f = plt.figure(facecolor='white')
    # 对size个数据进行移动平均
    rol_mean = timeSeries.rolling(window=size).mean()
    # 对size个数据进行加权移动平均
    rol_weighted_mean = pd.DataFrame(timeSeries).ewm(span=size).mean()
    # rol_weighted_mean=timeSeries.ewm(halflife=size,min_periods=0,adjust=True,ignore_na=False).mean()
 
    timeSeries.plot(color='blue', label='Original')
    rol_mean.plot(color='red', label='Rolling Mean')
    rol_weighted_mean.plot(color='black', label='Weighted Rolling Mean')
    plt.legend(loc='best')
    plt.title('Rolling Mean')
    plt.show()
 
 
draw_moving(ts, 7)
# 窗口为7的移动平均，剔除周期性因素，再对周期内数据进行加权，一定程度上减小周期因素
diff_7 = ts.diff(7)
diff_7.dropna(inplace=True)
print(teststationarity(diff_7))
diff_7_1 = diff_7.diff(1)
diff_7_1.dropna(inplace=True)
# 以上为差分
print(teststationarity(diff_7_1))
# Dickey-Fuller检验
# 预测
rol_mean = ts.rolling(window=7).mean()
rol_mean.dropna(inplace=True)
ts_diff_1 = rol_mean.diff(1)
ts_diff_1.dropna(inplace=True)
print(teststationarity(ts_diff_1))
# 二阶差分
ts_diff_2 = ts_diff_1.diff(1)
ts_diff_2.dropna(inplace=True)
print(teststationarity(ts_diff_2))
 
# 画出ACF、PACF图
def draw_acf_pacf(ts,lags):
    f = plt.figure(facecolor='white')
    ax1 = f.add_subplot(211)
    plot_acf(ts,ax=ax1,lags=lags)
    ax2 = f.add_subplot(212)
    plot_pacf(ts,ax=ax2,lags=lags)
    plt.subplots_adjust(hspace=0.5)
    plt.show()
 
# 画出2阶差分后的ACF、PACF图.PACF-p,ACF-q
draw_acf_pacf(ts_diff_2, 30)
 
 
# 拟合模型
model = ARIMA(ts_diff_1, order=(1, 1, 1))
result_arima = model.fit( disp=-1, method='css')
 
 
# 以下为数据还原
predict_ts = result_arima.predict()
# 一阶差分还原
diff_shift_ts = ts_diff_1.shift(1)
diff_recover_1 = predict_ts.add(diff_shift_ts)
# 再次一阶差分还原
rol_shift_ts = rol_mean.shift(1)
diff_recover = diff_recover_1.add(rol_shift_ts)
# 移动平均还原
rol_sum = ts.rolling(window=6).sum()
rol_recover = diff_recover*7 - rol_sum.shift(1)
# 对数还原
# log_recover = np.exp(rol_recover)
# log_recover.dropna(inplace=True)
# log_recover.plot()
# plt.show()
 
 
# 使用均方根误差（RMSE）评估模型拟合好坏。利用该准则进行判别时需要剔除“非预测”数据的影响
rol_recover.plot(color='red', label='Predict')
ts.plot(color='blue', label='Original')
plt.show()