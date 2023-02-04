import numpy as np
import scipy.stats

_SPEARMAN_TOP_2 = [(0.5256, 0.4402),
                   (0.6800, 0.3391),
                   (0.6249, 0.0972),
                   (0.3922, 0.3798),
                   (0.6696, 0.2735),
                   (0.7652, 0.6280),
                   (0.9302, 0.8687),
                   (0.9221, 0.7919),
                   (0.6128, 0.5063),
                   (0.6427, 0.5999)]
n1 = 60
n2 = 60

for comp in _SPEARMAN_TOP_2:
    z1 = 0.5*(np.log((1+comp[0])/(1-comp[0])))
    z2 = 0.5*(np.log((1+comp[1])/(1-comp[1])))
    z_eval = (z1-z2)/np.sqrt((1/(n1-3))+1/(n2-3))
    print(z_eval, scipy.stats.norm.sf(abs(z_eval))*2)
