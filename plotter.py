import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

###################### Plot cdf ###################### 

def plot_cdf(samples, label = None, ax = None):
    if True:
        samples = np.sort(samples)
        y = np.arange(len(samples))/float(len(samples))
    else :
        ecdf = ECDF(samples)
        samples, y = ecdf.x, ecdf.y
    if ax : ax.plot(samples, y, label=label)
    else : plt.plot(samples, y, label=label)
    return samples, y