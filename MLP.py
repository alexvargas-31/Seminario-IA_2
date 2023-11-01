#Calculo Gradiente

import sympy as sp

# Empezamos definiendo las variables y la función
x, y = sp.symbols('x y')
f = x**2 + 2*y**2

# Calcula las derivadas parciales que se muestran a continuacion
df_dx = sp.diff(f, x)
df_dy = sp.diff(f, y)

# Se crea el vector gradiente
gradiente = [df_dx, df_dy]

# Es el punto en el que deseas evaluar el gradiente
punto = (1, 2)

# Evalúa el gradiente en el punto
gradiente_punto = [g.evalf(subs={x: punto[0], y: punto[1]}) for g in gradiente]

# Print
print("Gradiente en el punto {}: {}".format(punto, gradiente_punto))