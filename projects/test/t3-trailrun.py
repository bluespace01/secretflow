# https://www.secretflow.org.cn/zh-CN/docs/secretflow/v1.6.1b0/tutorial/risk_control_scenario
import secretflow as sf

# Check the version of your SecretFlow
print('The version of SecretFlow: {}'.format(sf.__version__))

sf.shutdown()
sf.init(['alice', 'bob'], address='local')
alice, bob = sf.PYU('alice'), sf.PYU('bob')
spu = sf.SPU(sf.utils.testing.cluster_def(['alice', 'bob']))

#------------------------------------------------------------------------------------
# 数据集
#------------------------------------------------------------------------------------
import pandas as pd

# secretflow.utils.simulation.datasets contains mirrors of some popular open dataset.
from secretflow.utils.simulation.datasets import dataset

df = pd.read_csv(dataset('bank_marketing_full'), sep=';')
df['uid'] = df.index + 1

import numpy as np
df_alice = df.iloc[:, np.r_[0:8, -1]].sample(frac=0.9)
df_bob = df.iloc[:, 8:].sample(frac=0.9)


import tempfile

_, alice_path = tempfile.mkstemp()
_, bob_path = tempfile.mkstemp()
df_alice.reset_index(drop=True).to_csv(alice_path, index=False)
df_bob.reset_index(drop=True).to_csv(bob_path, index=False)

#------------------------------------------------------------------------------------
# 样本对齐（隐私求交）
#------------------------------------------------------------------------------------
# 方式一：将隐私求交结果保存至文件
_, alice_psi_path = tempfile.mkstemp()
_, bob_psi_path = tempfile.mkstemp()

spu.psi_csv(
    key="uid",
    input_path={alice: alice_path, bob: bob_path},
    output_path={alice: alice_psi_path, bob: bob_psi_path},
    receiver="alice",
    protocol="ECDH_PSI_2PC",
    sort=True,
)

# 方式二：将求交结果保存至VDataFrame
# VDataFrame是隐语中保存垂直切分数据的数据结构，在接下来的任务中，我们将会不断使用VDataFrame的数据结构。
from secretflow.data.vertical import read_csv as v_read_csv
vdf = v_read_csv(
    {alice: alice_path, bob: bob_path},
    spu=spu,
    keys="uid",
    drop_keys="uid",
    psi_protocl="ECDH_PSI_2PC",
)

print(vdf.columns)

#-------------------------------------------------------------------------------------
# 特征预处理
#------------------------------------------------------------------------------------
# 在开始特征预处理之前，我们先使用 stats.table_statistics.table_statistics 来查看一下特征总体情况，我们会在后面专门讨论全表统计模块。
from secretflow.stats.table_statistics import table_statistics
pd.set_option('display.max_rows', None)
data_stats = table_statistics(vdf)
print(data_stats)
pd.reset_option('display.max_rows')

# 值替换
vdf['education'] = vdf['education'].replace(
    {'tertiary': 3, 'secondary': 2, 'primary': 1, 'unknown': np.NaN}
)

vdf['default'] = vdf['default'].replace({'no': 0, 'yes': 1, 'unknown': np.NaN})

vdf['housing'] = vdf['housing'].replace({'no': 0, 'yes': 1, 'unknown': np.NaN})

vdf['loan'] = vdf['loan'].replace({'no': 0, 'yes': 1, 'unknown': np.NaN})

vdf['month'] = vdf['month'].replace(
    {
        'jan': 1,
        'feb': 2,
        'mar': 3,
        'apr': 4,
        'may': 5,
        'jun': 6,
        'jul': 7,
        'aug': 8,
        'sep': 9,
        'oct': 10,
        'nov': 11,
        'dec': 12,
    }
)

vdf['y'] = vdf['y'].replace(
    {
        'no': 0,
        'yes': 1,
    }
)
# 
print(sf.reveal(vdf.partitions[alice].data).head(10))
print(sf.reveal(vdf.partitions[bob].data).head(10))

#------------------------------------------------------------------------------------
# 缺失值填充
# 接下来我们对缺失值进行填充。我们在这里均填充了众数，其他可选的策略还包括平均数、中位数等。
vdf["education"] = vdf["education"].fillna(vdf["education"].mode())
vdf["default"] = vdf["default"].fillna(vdf["default"].mode())
vdf["housing"] = vdf["housing"].fillna(vdf["housing"].mode())
vdf["loan"] = vdf["loan"].fillna(vdf["loan"].mode())

print(sf.reveal(vdf.partitions[alice].data).head(10))
print(sf.reveal(vdf.partitions[bob].data).head(10))

#-------------------------------------------------------------------------------------
# woe分箱
# 变量duration的75%分位数远小于最大值，而且该变量的标准差相对也比较大。因此需要对变量duration进行离散化。
from secretflow.preprocessing.binning.vert_woe_binning import VertWoeBinning
from secretflow.preprocessing.binning.vert_bin_substitution import VertBinSubstitution

binning = VertWoeBinning(spu)
bin_rules = binning.binning(
    vdf,
    binning_method="chimerge",
    bin_num=4,
    bin_names={alice: [], bob: ["duration"]},
    label_name="y",
)

woe_sub = VertBinSubstitution()
vdf, _ = woe_sub.substitution(vdf, bin_rules)

print(sf.reveal(vdf.partitions[alice].data).head(10))
print(sf.reveal(vdf.partitions[bob].data).head(10))

#-------------------------------------------------------------------------------------
# one-hot编码适用于将类型编码转化为数值编码。 对于job、marital等特征我们需要one-hot编码。
from secretflow.preprocessing.encoder import OneHotEncoder

encoder = OneHotEncoder()
# for vif and correlation only
vdf_hat = vdf.drop(columns=["job", "marital", "contact", "month", "day", "poutcome"])

tranformed_df = encoder.fit_transform(vdf['job'])
vdf[tranformed_df.columns] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['marital'])
vdf[tranformed_df.columns] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['contact'])
vdf[tranformed_df.columns] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['month'])
vdf[tranformed_df.columns] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['day'])
vdf[tranformed_df.columns] = tranformed_df

tranformed_df = encoder.fit_transform(vdf['poutcome'])
vdf[tranformed_df.columns] = tranformed_df

vdf = vdf.drop(columns=["job", "marital", "contact", "month", "day", "poutcome"])

print(sf.reveal(vdf.partitions[alice].data).head(10))
print(sf.reveal(vdf.partitions[bob].data).head(10))

#-------------------------------------------------------------------------------------
# 标准化
#-------------------------------------------------------------------------------------
# 特征之间数值差距太大会使得模型收敛困难，我们一般先对数值进行标准化。
from secretflow.preprocessing import StandardScaler

X = vdf.drop(columns=['y'])
y = vdf['y']
scaler = StandardScaler()
X = scaler.fit_transform(X)
vdf[X.columns] = X
print(sf.reveal(vdf.partitions[alice].data).head(10))
print(sf.reveal(vdf.partitions[bob].data).head(10))


#-------------------------------------------------------------------------------------
# 数据分析
#-------------------------------------------------------------------------------------
from secretflow.stats.table_statistics import table_statistics

pd.set_option('display.max_rows', None)
data_stats = table_statistics(vdf)
print(data_stats)
pd.reset_option('display.max_rows')

#-------------------------------------------------------------------------------------
# 相关系数矩阵
from secretflow.stats.ss_pearsonr_v import PearsonR

pearson_r_calculator = PearsonR(spu)
corr_matrix = pearson_r_calculator.pearsonr(vdf_hat)

import numpy as np

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(corr_matrix)

#-------------------------------------------------------------------------------------
# VIF指标计算
from secretflow.stats.ss_vif_v import VIF

vif_calculator = VIF(spu)
vif_results = vif_calculator.vif(vdf_hat)
print(vdf_hat.columns)
print(vif_results)

#-------------------------------------------------------------------------------------
# 模型训练
#-------------------------------------------------------------------------------------
# 随机分割
from secretflow.data.split import train_test_split

random_state = 1234

train_vdf, test_vdf = train_test_split(vdf, train_size=0.8, random_state=random_state)

train_x = train_vdf.drop(columns=['y'])
train_y = train_vdf['y']

test_x = test_vdf.drop(columns=['y'])
test_y = test_vdf['y']

# PSI（人群稳定性分析）
stats_df = table_statistics(train_x['balance'])
min_val, max_val = stats_df['min'], stats_df['max']
print(min_val, max_val)


from secretflow.stats import psi_eval
from secretflow.stats.core.utils import equal_range
import jax.numpy as jnp

split_points = equal_range(jnp.array([min_val, max_val]), 3)
balance_psi_score = psi_eval(train_x['balance'], test_x['balance'], split_points)

print(sf.reveal(balance_psi_score))

#-------------------------------------------------------------------------------------
# 逻辑回归模型
from secretflow.ml.linear.ss_sgd import SSRegression

lr_model = SSRegression(spu)
lr_model.fit(
    x=train_x,
    y=train_y,
    epochs=3,
    learning_rate=0.1,
    batch_size=1024,
    sig_type='t1',
    reg_type='logistic',
    penalty='l2',
    l2_norm=0.5,
)

# XGBoost模型
from secretflow.ml.boost.ss_xgb_v import Xgb

xgb = Xgb(spu)
params = {
    'num_boost_round': 3,
    'max_depth': 5,
    'sketch_eps': 0.25,
    'objective': 'logistic',
    'reg_lambda': 0.2,
    'subsample': 1,
    'colsample_by_tree': 1,
    'base_score': 0.5,
}
xgb_model = xgb.train(params=params, dtrain=train_x, label=train_y)



#-------------------------------------------------------------------------------------
# 模型预测
# 逻辑回归模型
lr_y_hat = lr_model.predict(x=test_x, batch_size=1024, to_pyu=bob)
# XGBoost模型
xgb_y_hat = xgb_model.predict(dtrain=test_x, to_pyu=bob)


#-------------------------------------------------------------------------------------
# 模型评估
    # 二分类评估
    # PVA
    # P-Value
    # 评分卡转换

# 二分类评估
# BiClassificationEval 将计算 AUC, KS, F1 Score, Lift, K-S, Gain, Precision, Recall 等统计数值， 并提供（基于prediction score的）等频和等距分箱的统计报告和总报告。

from secretflow.stats.biclassification_eval import BiClassificationEval

biclassification_evaluator = BiClassificationEval(
    y_true=test_y, y_score=lr_y_hat, bucket_size=20
)
lr_report = sf.reveal(biclassification_evaluator.get_all_reports())

print(f'positive_samples: {lr_report.summary_report.positive_samples}')
print(f'negative_samples: {lr_report.summary_report.negative_samples}')
print(f'total_samples: {lr_report.summary_report.total_samples}')
print(f'auc: {lr_report.summary_report.auc}')
print(f'ks: {lr_report.summary_report.ks}')
print(f'f1_score: {lr_report.summary_report.f1_score}')

biclassification_evaluator = BiClassificationEval(
    y_true=test_y, y_score=xgb_y_hat, bucket_size=20
)
xgb_report = sf.reveal(biclassification_evaluator.get_all_reports())
print(f'positive_samples: {xgb_report.summary_report.positive_samples}')
print(f'negative_samples: {xgb_report.summary_report.negative_samples}')
print(f'total_samples: {xgb_report.summary_report.total_samples}')
print(f'auc: {xgb_report.summary_report.auc}')
print(f'ks: {xgb_report.summary_report.ks}')
print(f'f1_score: {xgb_report.summary_report.f1_score}')


# 预测偏差
from secretflow.stats import prediction_bias_eval

prediction_bias = prediction_bias_eval(
    test_y, lr_y_hat, bucket_num=4, absolute=True, bucket_method='equal_width'
)

print(sf.reveal(prediction_bias))

xgb_pva_score = prediction_bias_eval(
    test_y, xgb_y_hat, bucket_num=4, absolute=True, bucket_method='equal_width'
)

print(sf.reveal(xgb_pva_score))

# P-Value
# 双方可通过p-value的值来判断参数是否显著，即该自变量是否可以有效预测因变量的变异, 从而判定对应的解释变量是否应包括在模型中
from secretflow.stats import SSPValue

# model = lr_model.save_model()
# sspv = SSPValue(spu)
# pvalues = sspv.pvalues(test_x, test_y, model)

# print(pvalues)


# 评分卡转换

# 我们将 y = 1 的概率设为p， odds = p / (1 - p), 评分卡设定的分值刻度可以通过将分值表示为比率对数的线性表达式来定义，即可表示为下式：

# Score = A - B log(odds)， A 和 B 是可以设定的常数。隐语中提供了评分卡转换功能，详情可以参考API文档。

from secretflow.stats import BiClassificationEval, ScoreCard

sc = ScoreCard(20, 600, 20)
score = sc.transform(xgb_y_hat)

print(sf.reveal(score.partitions[bob]))


#-------------------------------------------------------------------------------------
# 实验结束
#-------------------------------------------------------------------------------------

import os

try:
    os.remove(alice_path)
    os.remove(alice_psi_path)
    os.remove(bob_path)
    os.remove(bob_psi_path)
except OSError:
    pass

sf.shutdown()



