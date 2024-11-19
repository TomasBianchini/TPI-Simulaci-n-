import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Cargar datos
df_PG = pd.read_csv('PowerGenerated.csv')
df_PG['powerGenerated'] = pd.to_numeric(df_PG['powerGenerated'], errors='coerce')
valores_PG = df_PG['powerGenerated'].dropna()


# Ajustar los datos a una distribución Weibull
shape, loc, scale = stats.weibull_min.fit(valores_PG)

# Crear el Q-Q plot para la distribución Weibull
fig, axs = plt.subplots(1, figsize=(14, 7))
res = stats.probplot(valores_PG, dist="weibull_min", sparams=(shape, loc, scale))

# Extraer los datos teóricos y empíricos del Q-Q plot
teoricos = res[0][0]  # Cuantiles teóricos
empiricos = res[0][1]  # Cuantiles empíricos

# Filtrar para limitar los valores teóricos hasta 3000
teoricos_filtrados = teoricos[teoricos <= 3000]
empiricos_filtrados = empiricos[:len(teoricos_filtrados)]  # Ajustar los empíricos a la misma longitud

# Volver a graficar con los datos limitados
axs.plot(teoricos_filtrados, empiricos_filtrados, 'bo', label="Datos filtrados")
axs.plot([0, 3000], [0, 3000], 'r-', label="Ajuste ideal (45°)")  # Línea de referencia
axs.set_title('Q-Q plot para la distribución Weibull')
axs.set_xlabel('Cuantiles teóricos (Weibull)')
axs.set_ylabel('Cuantiles empíricos')

# Extraer la linea de regresión para ajustar la leyenda correctamente
line = axs.get_lines()[1]
line.set_label('Ajuste lineal')
axs.legend()

plt.tight_layout()
plt.show()
