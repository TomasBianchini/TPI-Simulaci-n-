import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Cargar datos
df_FI = pd.read_csv('retornos_FI.csv')
valores_FI = df_FI['retornos']

# QQ plot
fig, axs = plt.subplots(1, figsize=(14, 7))

# QQ para los retornos de los fondos de inversión

(stats.probplot(valores_FI, dist="norm", plot=axs))
axs.set_title('QQ plot para los retornos de los fondos de inversión')
axs.set_xlabel('Cuantiles teóricos')
axs.set_ylabel('Cuantiles Empiricos')

# Extraer la linea de regresión para ajustar la leyenda correctamente
line = axs.get_lines()[1]
line.set_label('Ajuste lineal')
axs.legend()

plt.tight_layout()
plt.show()
