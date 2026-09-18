from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA


# 读取包含销售数据的 CSV 文件
df = pd.read_csv("异常值处理3.csv", encoding='utf-8')

# 将销售日期列转换为日期类型
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 获取唯一的分类名称
unique_categories = df['分类名称'].unique()

# 创建一个结果 DataFrame，包含销售日期和各个分类的列
result_df = pd.DataFrame(columns=['销售日期'] + list(unique_categories))

# 循环遍历每个日期，计算每天的类别加权售价
dates = df['销售日期'].unique()
for date in dates:
    # 获取当天的数据
    date_data = df[df['销售日期'] == date]

    # 计算当天的总销量
    total_sales = date_data['销量(千克)'].sum()

    # 初始化一个字典，用于存储各个类别的加权售价
    weighted_prices = {}

    # 遍历每个类别，计算加权售价
    for category in unique_categories:
        category_data = date_data[date_data['分类名称'] == category]
        # 计算当天的总销量
        total_sales = category_data['销量(千克)'].sum()
        category_sales = category_data['销量(千克)'].sum()
        category_weighted_price = (
                    category_data['销售单价(元/千克)'] * (category_data['销量(千克)'] / total_sales)).sum()
        weighted_prices[category] = category_weighted_price

    # 创建一个 DataFrame，用于存储当天的结果
    daily_result = {'销售日期': date, **weighted_prices}

    # 将当天的结果添加到结果 DataFrame 中
    result_df = pd.concat([result_df, pd.DataFrame([daily_result])], ignore_index=True)

# 将结果保存到 CSV 文件
result_df.to_csv("加权平均售价.csv", index=False, encoding='utf-8')


# 修改matplotlib字体设置，支持中文显示（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取CSV文件
data = pd.read_csv('7月.csv', encoding='gbk')

# 提取销量和售价列数据
sales = data['销量'].values
price = data['售价'].values

# 定义指数函数模型
def exponential_func(x, a, b):
    return a * np.exp(b / x)

# 使用curve_fit拟合数据
params, covariance = curve_fit(exponential_func, price, sales)

# 提取拟合的参数
a, b = params

# 生成平滑的拟合曲线
x_smooth = np.linspace(min(price), max(price), 100)
y_smooth = exponential_func(x_smooth, a, b)

# 提取拟合的参数
a, b = params

# 计算R-squared值
y_predicted = exponential_func(price, a, b)
r_squared = r2_score(sales, y_predicted)
print(a)
print(b)
# 输出R-squared值
print(f'R-squared值: {r_squared:.2f}')
# 设置字体系列为Times New Roman
annot_kws = {'family': 'Times New Roman', 'size': 12, 'weight': 'normal'}

# 绘制实际数据的散点图和拟合曲线
plt.scatter(price, sales, label='实际数据')
plt.plot(x_smooth, y_smooth, color='red', linewidth=2, label='拟合曲线')

plt.title('水生根茎类销量随售价变化拟合曲线',fontsize=16)
# 添加标签和图例
plt.xlabel('售价',fontsize=12)
plt.ylabel('销量',fontsize=12)
plt.legend()

# 显示图形
plt.show()



# 读取附件4.xlsx和附件1.xlsx
data4 = pd.read_csv('附件4.csv',encoding='gbk')
data1 = pd.read_csv('附件1.csv',encoding='gbk')

# 使用merge函数将附件4和附件1连接在一起，基于单品名称列进行连接
joined_data = data4.merge(data1, on='单品名称')

# 导出连接后的数据为CSV文件
joined_data.to_csv('损耗率.csv', index=False, encoding='utf-8-sig')




# 修改matplotlib字体设置，支持中文显示（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 尝试不同的编码方式读取CSV文件
possible_encodings = ['utf-8', 'utf-16', 'gbk', 'latin-1', 'utf-8-sig']
for encoding in possible_encodings:
    try:
        data = pd.read_csv("Arima.csv", encoding=encoding)
        break
    except UnicodeDecodeError:
        continue
else:
    raise ValueError("无法使用已知编码方式解码CSV文件。")




# 将日期列转换为日期格式
data['销售日期'] = pd.to_datetime(data['销售日期'])

# 设置日期列为索引
data.set_index('销售日期', inplace=True)

# 创建时间序列变量
time_series = data['水生根茎类3']

# 定义平稳性检验函数


def check_stationarity(timeseries):


    # 进行Dickey-Fuller检验
    result = adfuller(timeseries, autolag='AIC')
    print("ADF Test Results:")
    print(f'ADF Statistic: {result[0]}')
    print(f'p-value: {result[1]}')
    print(f'Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')

    if result[1] <= 0.05:
        print("数据平稳，无需差分处理。")
        d = 0
        return timeseries, d
    else:
        print("数据非平稳，进行一阶差分处理。")
        d = 1
        first_diff = timeseries.diff().dropna()

        # 进行一阶差分后再次进行平稳性检验
        result_diff = adfuller(first_diff, autolag='AIC')
        print("一阶差分后的平稳性检验结果:")
        print(f'ADF Statistic: {result_diff[0]}')
        print(f'p-value: {result_diff[1]}')
        print(f'Critical Values:')
        for key, value in result_diff[4].items():
            print(f'   {key}: {value}')

        if result_diff[1] <= 0.05:
            print("一阶差分后数据平稳，无需二阶差分处理。")
            return first_diff, d
        else:
            print("一阶差分后数据仍非平稳，进行二阶差分处理。")
            d = 2
            second_diff = first_diff.diff().dropna()

            # 进行二阶差分后再次进行平稳性检验
            result_diff = adfuller(second_diff, autolag='AIC')
            print("二阶差分后的平稳性检验结果:")
            print(f'ADF Statistic: {result_diff[0]}')
            print(f'p-value: {result_diff[1]}')
            print(f'Critical Values:')
            for key, value in result_diff[4].items():
                print(f'   {key}: {value}')

            if result_diff[1] <= 0.05:
                print("二阶差分后数据平稳。")
                return second_diff, d
            else:
                print("经过二阶差分后数据仍非平稳，建议进一步处理。")
                return None, None


# 进行平稳性检验
stationary_time_series, d = check_stationarity(time_series)

def white_noise_test(data, lags_list):
    from statsmodels.stats.diagnostic import acorr_ljungbox

    for lag in lags_list:
        re = acorr_ljungbox(data, lags=lag)
        print(f"延迟阶数: {lag}")
        print("卡方统计量:", re.lb_stat)
        print("P值:", re.lb_pvalue)

        for p in re.lb_pvalue:
            if p <= 0.05:
                print("序列不是白噪声。")
                return True

    print("序列为白噪声。")
    return False
lags_list=[9]
result=white_noise_test(time_series,lags_list)

if stationary_time_series is not None :
    # 绘制差分后的ACF和PACF图
    plt.figure(figsize=(12, 6))
    plot_acf(stationary_time_series, lags=20, title='ACF 自相关函数图')
    plt.show()

    plt.figure(figsize=(12, 6))
    plot_pacf(stationary_time_series, lags=20, title='PACF 偏自相关函数图')
    plt.show()

    # 网格搜索选择ARIMA模型的阶数
    best_aic = np.inf
    best_order = None

    for p in range(3):
            for q in range(10):
                try:
                    model = ARIMA(stationary_time_series, order=(p, d, q))
                    results = model.fit()
                    aic = results.aic

                    if aic < best_aic:
                        best_aic = aic
                        best_order = (p, d, q)

                except:
                    continue

    # 输出选择的最优阶数
    print(f"选择的最优阶数为：{best_order}, AIC值为：{best_aic}")

    # 构建最优ARIMA模型
    model = ARIMA(stationary_time_series, order=best_order)
    results = model.fit()

    print(stationary_time_series)

    model_summary = results.summary()

    # # 获取自回归参数 (AR)
    # ar_params = results.arparams
    # # 将AR参数转换为列表并打印
    # print("自回归参数 (AR):", list(ar_params))
    # # 获取滑动平均参数 (MA)
    # ma_params = results.maparams
    # # 将MA参数转换为列表，并使用join()方法将其连接成一个字符串，然后打印
    # ma_params_str = ", ".join(map(str, ma_params))
    # print("滑动平均参数 (MA): " + ma_params_str)
    #
    # # 获取残差 (Residuals)
    # residuals = results.resid
    # # 将残差转换为列表并打印
    # residuals_list = list(residuals)
    # print("残差:", residuals_list)
    #
    # # 计算残差的方差
    # residual_variance = np.var(residuals)
    # print("残差方差:", residual_variance)

    # 将模型参数转换为DataFrame
    model_summary_df = pd.DataFrame(model_summary.tables[1].data[1:], columns=model_summary.tables[1].data[0])

    # 保存模型参数为CSV文件
    model_summary_file_path = "model_summary.csv"
    model_summary_df.to_csv(model_summary_file_path, index=False)

    # 绘制拟合图形
    plt.figure(figsize=(10, 6))
    plt.plot(stationary_time_series, label='Actual')
    plt.plot(results.fittedvalues, color='red', label='Fitted')
    plt.legend()
    plt.title('ARIMA 拟合预测加成率')
    plt.show()

    # 预测未来时间步数的结果
    forecast_steps = 7
    forecast_values = results.forecast(steps=forecast_steps)

    # 如果进行了差分，将预测结果逆差分得到原始数据的尺度
    if d > 0:
        forecast_values_original_scale = time_series.iloc[-1] + forecast_values.cumsum()
    else:
        forecast_values_original_scale = forecast_values

    # 获取模型对历史数据的拟合值
    fitted_values = results.fittedvalues
    # 获取残差
    residuals = stationary_time_series - fitted_values



    # 检查白噪声
    lags_list = [9]
    result = white_noise_test(residuals, lags_list)


    # 计算R方（拟合优度）
    def calculate_r_squared(actual, predicted):
        # 计算总平均值
        mean_actual = np.mean(actual)
        # 计算总平方和
        total_sum_of_squares = np.sum((actual - mean_actual) ** 2)
        # 计算残差平方和
        residual_sum_of_squares = np.sum((actual - predicted) ** 2)
        # 计算R方
        r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
        return r_squared


    # 计算R方
    r_squared = calculate_r_squared(stationary_time_series, fitted_values)
    print(f"拟合优度（R方）: {r_squared:.2f}")
