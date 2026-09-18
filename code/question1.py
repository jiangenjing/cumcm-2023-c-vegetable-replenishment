import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 数据表连接
# 读取附件1和附件2的数据
fujian1 = pd.read_excel('附件1.xlsx')
fujian2 = pd.read_excel('附件2.xlsx')

# 使用merge函数将两个数据框连接起来，根据单品编码进行连接
result = pd.merge(fujian2, fujian1[['单品编码', '单品名称', '分类名称']], on='单品编码', how='left')

result.to_excel('附件2_带单品名称和分类名称.xlsx', index=False)


#连接附件2和附件3
# 读取附件1和附件2的数据
fujian1 = pd.read_excel('附件1.xlsx')
fujian2 = pd.read_excel('附件2.xlsx')

# 获取附件1和附件2中的单品编码
product_codes_fujian1 = set(fujian1['单品编码'])
product_codes_fujian2 = set(fujian2['单品编码'])

# 找出在附件1中出现但在附件2中没有出现的单品编码
missing_product_codes = product_codes_fujian1 - product_codes_fujian2

# 根据缺失的单品编码从附件1中筛选相应的单品名称和分类名称
missing_products_info = fujian1[fujian1['单品编码'].isin(missing_product_codes)][['单品编码', '单品名称', '分类名称']]

# 输出缺失的商品信息
print("以下是在附件1中出现但在附件2中没有出现的商品信息：")
print(missing_products_info)


# 读取附件2_带单品名称和分类名称和附件3的数据
fujian2 = pd.read_excel('附件2_带单品名称和分类名称.xlsx')
df3 = pd.read_excel('附件3.xlsx')

# 使用merge函数进行左连接，根据销售日期和单品编码连接，保留附件2的所有数据
result = pd.merge(fujian2, df3[['日期', '单品编码', '批发价格(元/千克)']], left_on=['销售日期', '单品编码'], right_on=['日期', '单品编码'], how='left')

# 删除连接后多余的日期列
result.drop(columns=['日期'], inplace=True)

# 将结果中的NaN值（没有匹配的批发价格）替换为空白字符串
result['批发价格(元/千克)'].fillna('', inplace=True)

# 将结果保存到一个新的Excel文件中
result.to_excel('附件2_带单品名称分类名称和批发价格.xlsx', index=False)



# 单品销量汇总
# 读取附件2_带单品名称分类名称和批发价格的数据
df = pd.read_excel('附件2_带单品名称分类名称和批发价格.xlsx')

# 将销售日期转换为日期格式，并提取年月日
df['销售日期'] = pd.to_datetime(df['销售日期']).dt.date

# 按销售日期和单品编码分组，累加销量
grouped = df.groupby(['销售日期', '单品编码', '单品名称', '分类名称'], as_index=False)['销量(千克)'].sum()

# 将单品编码转为文本格式，以避免科学计数法
grouped['单品编码'] = grouped['单品编码'].astype(str)

# 保存结果到一个新的Excel文件
grouped.to_excel('销售汇总表.xlsx', index=False)


#补全日期序列
# 读取水生根茎类数据文件
df = pd.read_csv("六类单日.csv",encoding='gbk')

# 创建一个包含完整日期范围的日期列
start_date = pd.to_datetime('2020-07-01')
end_date = pd.to_datetime('2023-06-30')
date_range = pd.date_range(start_date, end_date, freq='D')
date_df = pd.DataFrame({'销售日期': date_range})

# 将原始数据表中的"销售日期"列转换为datetime64[ns]格式
df['销售日期'] = pd.to_datetime(df['销售日期'])

# 合并数据表并填充缺失值为0
merged_df = date_df.merge(df, on='销售日期', how='left').fillna({'食用菌': 0})


merged_df.to_csv('填充后.csv', index=False)


# 异常值处理


# 异常值处理1：删除利润率大于75%的数据
# 读取包含数据的文件，使用实际的文件名
df = pd.read_excel('附件2_带单品名称分类名称和批发价格.xlsx')

# 计算利润率
df['利润率'] = (df['销售单价(元/千克)'] - df['批发价格(元/千克)']) / df['销售单价(元/千克)']

# 筛选出利润率不大于0.75的记录
yichang1 = df[df['利润率'] <= 0.856]

# 将销售日期列转换为日期时间类型
yichang1['销售日期'] = pd.to_datetime(yichang1['销售日期'])
# 保持日期列的格式不变，保留年月日，不包含时分秒
yichang1['销售日期'] = yichang1['销售日期'].dt.strftime('%Y-%m-%d')

# 将单品编码列转换为文本格式，避免科学计数法
yichang1['单品编码'] = yichang1['单品编码'].astype(str)

# 将筛选后的数据保存到一个新文件，假设文件名为filtered_data.csv
yichang1.to_csv('异常值处理1.csv', index=False)
#

# 读取包含数据的Excel文件，假设文件名为异常值处理1.xlsx，你需要将文件名替换成实际的文件名
df = pd.read_csv('异常值处理1.csv')

# 筛选出是否打折销售为"否"且销售单价小于批发价格的记录
yichang2 = df[((df['是否打折销售'] == '否') & (df['销售单价(元/千克)'] >= df['批发价格(元/千克)']))|(df['是否打折销售'] == '是')]

yichang2['销售日期'] = pd.to_datetime(yichang2['销售日期'])
# 保持日期列的格式不变，保留年月日，不包含时分秒
yichang2['销售日期'] = yichang2['销售日期'].dt.strftime('%Y-%m-%d')

# 将单品编码列转换为文本格式，避免科学计数法
yichang2['单品编码'] = yichang2['单品编码'].astype(str)

# 将删除后的数据保存到一个新Excel文件，假设文件名为filtered_data.xlsx
yichang2.to_csv('异常值处理2.csv', index=False)


#异常值处理3
# 读取 Excel 文件
yichang3 = pd.read_csv('异常值处理2.csv',encoding='utf-8')

# 定义一个函数来筛选符合条件的行
def filter_group(group):
    # 找到是否打折销售为是的行
    discount_sales = group[group['是否打折销售'] == '是']

    if not discount_sales.empty:
        # 找到组内非打折销售为否的最低销售单价
        min_price = group[group['是否打折销售'] == '否']['销售单价(元/千克)'].min()

        # 筛选出满足条件的行
        return discount_sales[discount_sales['销售单价(元/千克)'] > min_price]
    else:
        # 如果该分组没有打折销售为是的行，则返回整个分组
        return


# 根据销售日期和单品名称进行分组，然后应用筛选函数
filtered_data = yichang3.groupby(['销售日期', '单品名称']).apply(filter_group).reset_index(drop=True)
# 删除满足条件的行
yichang3 = yichang3.drop(filtered_data.index)

yichang3['销售日期'] = pd.to_datetime(yichang3['销售日期'])
# 保持日期列的格式不变，保留年月日，不包含时分秒
yichang3['销售日期'] = yichang3['销售日期'].dt.strftime('%Y-%m-%d')

# 将单品编码列转换为文本格式，避免科学计数法
yichang3['单品编码'] = yichang3['单品编码'].astype(str)

# 将筛选后的结果保存到新的 Excel 文件
yichang3.to_csv('异常值处理3.csv', index=False)


import pandas as pd
#单日单品汇总
# 读取数据
data = pd.read_csv('单品单日.csv', encoding='gbk')

# 创建一个新的DataFrame来保存转换后的数据
new_data = pd.DataFrame()

# 遍历每个单品名称
unique_products = data['单品名称'].unique()
for product in unique_products:
    # 从原始数据中选择特定单品名称的记录
    product_data = data[data['单品名称'] == product]
    print(product_data)

    # 获取销售日期和销售量的列，并重命名列名
    product_data = product_data.rename(columns={'销售日期': f'{product}销售日期', '销量(千克)': f'{product}销售量'})

    # 将这些列合并到新的DataFrame中
    new_data = pd.concat([new_data, product_data[[f'{product}销售日期', f'{product}销售量']]], axis=1)

# 保存转换后的数据到新的CSV文件
new_data.to_csv('单日单品汇总.csv', index=False, encoding='utf-8')


# 读取包含多个类别销售数据的文件
df = pd.read_csv("单日单品汇总.csv", encoding='gbk')

# 创建一个包含完整日期范围的日期列
start_date = pd.to_datetime('2020-07-01')
end_date = pd.to_datetime('2023-06-30')
date_range = pd.date_range(start_date, end_date, freq='D')
date_df = pd.DataFrame({'销售日期': date_range})

# 创建一个空的DataFrame用于存储最终结果
final_df = date_df.copy()

# 获取每个类别的销售日期列和销售量列，并合并到最终DataFrame
categories = df.columns[::2]  # 假设每两列为一组，第一列是销售日期，第二列是销售量
for category in categories:
    date_column = category
    sales_column = category.replace("销售日期", "销售量")

    # 提取类别名称，假设列名格式为"类别名称销售日期"
    category_name = category.split("销售日期")[0]

    # 将原始数据表中的销售日期列转换为datetime64[ns]格式
    df[date_column] = pd.to_datetime(df[date_column])

    # 合并数据并填充缺失值
    merged_df = date_df.merge(df[[date_column, sales_column]], left_on='销售日期', right_on=date_column,
                              how='left').fillna({sales_column: 0})

    # 重命名销售量列，以便区分不同类别
    merged_df = merged_df.rename(columns={sales_column: f'销售量(千克)_{category_name}'})

    # 合并结果到最终DataFrame
    final_df = final_df.merge(merged_df[['销售日期', f'销售量(千克)_{category_name}']], on='销售日期', how='left')

# 输出结果到CSV文件
final_df.to_csv('填充后.csv', index=False)


# 读取填充后的CSV文件
data = pd.read_csv('填充后.csv', encoding='utf-8')

# 删除销售日期列中的重复日期行
data = data.drop_duplicates(subset=['销售日期'])

# 保存修改后的数据到新的CSV文件
data.to_csv('填充后单品单日汇总.csv', index=False, encoding='utf-8')




# 六大品类相关性矩阵
# 修改matplotlib字体设置，支持中文显示（Windows）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 或者 ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取包含销量数据的CSV文件
df = pd.read_csv("填充后六类单日汇总.csv", encoding='gbk')

# 选择包含销售量数据的列
selected_columns = ['水生根茎类', '花叶类', '花菜类', '茄类', '辣椒类', '食用菌']

# 选择这些列的数据
selected_data = df[selected_columns]

# 计算相关系数矩阵
correlation_matrix = selected_data.corr(method='spearman')

# 绘制热力图，使用高科技感的颜色
cmap = sns.color_palette("Blues", as_cmap=True)  # 选择深蓝色系

# 设置字体系列为Times New Roman
annot_kws = {'family': 'Times New Roman', 'size': 12, 'weight': 'normal'}

ax = sns.heatmap(correlation_matrix, annot=True, cmap=cmap, linewidths=.5, annot_kws=annot_kws)

# 获取颜色条对象
cbar = ax.collections[0].colorbar

# 设置颜色条上标签的字体系列和大小
for label in cbar.ax.get_yticklabels():
    label.set_family('Times New Roman')
    label.set_fontsize(12)

# 计算注释文本的高度和宽度，以确定画布大小
height, width = correlation_matrix.shape

# 设置 DPI 值，增加分辨率和清晰度
plt.gcf().set_size_inches(0.2 * width, 0.2 * height)
plt.gcf().set_dpi(1200)  # 设置 DPI

plt.style.use("ggplot")
plt.title('销售量相关系数矩阵热力图', fontsize=18)
plt.show()




# 读取填充后的CSV文件
data = pd.read_csv('填充后单品单日汇总.csv', encoding='utf-8')

# 提取除第一列之外的所有列
data_numeric = data.iloc[:, 1:]

# 计算Spearman相关系数
spearman_corr = data_numeric.corr(method='spearman')

# 绘制相关系数矩阵图
plt.figure(figsize=(12, 8))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('Spearman相关系数矩阵')
plt.show()

# 导出相关系数矩阵的表格
spearman_corr.to_csv('所有单品Spearman相关系数矩阵.csv', index=True, encoding='utf-8')







