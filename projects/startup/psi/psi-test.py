import pandas as pd
# 创建示例数据

da = pd.DataFrame({
    'uid': [1, 2, 3],
    'month': [1, 1, 2],
    'value_da': ['a', 'b', 'c'],
    'value': ['a', 'b', 'c']
})

db = pd.DataFrame({
    'uid': [1, 2, 3, 3],
    'month': [1, 2, 2, 3],
    'value_db': ['x', 'y', 'z', 'w'],
    'value': ['x', 'y', 'z', 'w']
})

# 执行 join 操作
df = da.join(
    db.set_index(['uid', 'month']),
    on=['uid', 'month'],
    how='inner',
    rsuffix='_bob',
    sort=True,
)

print(df)


