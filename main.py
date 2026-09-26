import os
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pingouin as pg
except ImportError:
    print("Установка библиотеки pingouin...")
    os.system("pip install pingouin")
    import pingouin as pg

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Создаем папки под результаты
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Исходные данные 
group_A = [3.1, 3.5, 3.3, 2.9, 3.7, 3.4, 3.6, 3.2, 3.0, 3.8, 3.3, 3.5, 3.1, 3.4, 3.6, 3.2, 3.0, 3.7, 3.5, 3.3, 3.4, 3.6, 3.2, 3.1, 3.5, 3.3, 3.7, 3.0, 3.4, 3.6, 3.2, 3.5, 3.3, 3.1, 3.4, 3.6, 3.0, 3.2, 3.5, 3.3, 3.4, 3.1, 3.7, 3.2, 3.5, 3.3, 3.0, 3.6, 3.4, 3.2]
group_B = [5.2, 5.6, 5.4, 5.0, 5.8, 5.5, 5.7, 5.3, 5.1, 5.9, 5.4, 5.6, 5.2, 5.5, 5.7, 5.3, 5.1, 5.8, 5.6, 5.4, 5.5, 5.7, 5.3, 5.2, 5.6, 5.4, 5.8, 5.1, 5.5, 5.7, 5.3, 5.6, 5.4, 5.2, 5.5, 5.7, 5.1, 5.3, 5.6, 5.4, 5.5, 5.2, 5.8, 5.3, 5.6, 5.4, 5.1, 5.7, 5.5, 5.3]
group_C = [2.0, 2.3, 2.1, 1.9, 2.4, 2.2, 2.0, 2.3, 2.1, 2.2, 2.0, 2.3, 2.1, 2.2, 2.4, 2.0, 1.9, 2.3, 2.1, 2.2, 2.3, 2.0, 2.1, 2.2, 2.3, 2.0, 2.4, 1.9, 2.2, 2.1, 2.0, 2.3, 2.1, 2.2, 2.0, 2.3, 1.9, 2.2, 2.1, 2.0, 2.3, 2.1, 2.4, 2.2, 2.0, 2.1, 1.9, 2.3, 2.2, 2.1]

df = pd.DataFrame({
    'Weight_Loss': group_A + group_B + group_C,
    'Diet': ['Diet A']*50 + ['Diet B']*50 + ['Diet C']*50
})

print("======================================================================")
print("======================================================================")

# 1. Описательная статистика
desc = df.groupby('Diet')['Weight_Loss'].agg(['count', 'mean', 'median', 'std', 'min', 'max'])
desc.to_csv("results/descriptive_stats.csv")
print("\n--- 1. Описательная статистика ---")
print(desc.round(3))

# 2. Проверка нормальности
print("\n--- 2. Проверка условий (Шапиро-Уилк) ---")
for name, group in [('Diet A', group_A), ('Diet B', group_B), ('Diet C', group_C)]:
    w, p = stats.shapiro(group)
    print(f"{name}: p-value = {p:.4f}")

# 3. Критерий Левена
levene_stat, levene_p = stats.levene(group_A, group_B, group_C)
print(f"Критерий Левена: p-value = {levene_p:.4f}")

# 4. Welch ANOVA
print("\n--- 3-4. Однофакторный ANOVA Уэлча ---")
anova_table = pg.welch_anova(dv='Weight_Loss', between='Diet', data=df)
anova_table.to_csv("results/anova_results.csv", index=False)
print(anova_table.to_string(index=False))

# 5. Попарные сравнения Тьюки
print("\n--- 5. Попарные сравнения (Tukey HSD) ---")
tukey = pairwise_tukeyhsd(endog=df['Weight_Loss'], groups=df['Diet'], alpha=0.05)
print(tukey)

# Сохранение графика
plt.figure(figsize=(7, 5))
sns.boxplot(x='Diet', y='Weight_Loss', data=df, palette='Set2')
plt.title('Сравнение снижения веса по группам диет')
plt.savefig("figures/boxplot.png", dpi=300)
print("\n[INFO] График успешно сохранен в папочку figures/boxplot.png")
