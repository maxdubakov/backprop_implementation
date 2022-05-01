import numpy as np

z = np.random.uniform(-2, 2, (4, 4))
print(z)
print(z * (z > 0))
der_z = (z > 0) * 1
print(der_z)
