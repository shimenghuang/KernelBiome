import numpy as np
from matplotlib import rc


rc('font', **{'family': 'tex-gyre-termes', 'size': 6.5})
rc('text', usetex=True)
rc('text.latex',
   preamble=r'\usepackage{amsfonts,amssymb,amsthm,amsmath}')

# plt.cm.tab10(range(10)), plt.cm.tab20(range(20)), plt.cm.tab20b(range(20)),
colors_all = np.vstack([plt.cm.tab20c(range(20)), plt.cm.tab20b(range(20))])
colors_all = np.unique(colors_all, axis=0)
print(colors_all.shape)
