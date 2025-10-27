from scipy.optimize import minimize
from scipy.interpolate import make_smoothing_spline

from xasproc.metrics import stat_scores
from xasproc.preprocess import get_edge

class LossFn(object):
    def __init__(self, exp_spectrum, metric='l2_norm_dist'):
        self._exp_spectrum = exp_spectrum
        self._metric = metric
        if metric not in stat_scores:
            raise ValueError(f"Metric '{metric}' is not supported.")

    def calculate(self, R):

        metric_fun = stat_scores[self._metric]
        eshift_metric_fun = stat_scores[self._metric]
        if self._metric == 'spearman_corr':
            eshift_metric_fun = stat_scores['l2_norm_dist']

        scale_min, scale_max = 0.9, 1.2
        e0 = self._exp_spectrum.e0
        X = R[0]
        X.energy = X.energy - (get_edge(X) - e0) # rough energy alignment
        energy_range = \
            max(
                self._exp_spectrum.energy.min(), 
                (e0*(scale_max-1.) + X.energy.min())/scale_max
            )+0.1,  \
            min(
                self._exp_spectrum.energy.max(),
                (e0*(scale_max-1.) + X.energy.max())/scale_max
            )-0.1

        emask = (energy_range[0] <= self._exp_spectrum.energy) & (self._exp_spectrum.energy <= energy_range[1])

        def xmu(energy):
            return make_smoothing_spline(X.energy, X.mu, lam=0.1)(energy, extrapolate=False)

        def diff_func(x):
            xe = e0 + x[0]*(self._exp_spectrum.energy-e0)
            xe = xe[emask]

            return eshift_metric_fun(self._exp_spectrum.mu[emask], xmu(xe))

        ediff_res = minimize(diff_func, [1.0], method='L-BFGS-B', bounds=[(scale_min, scale_max)])

        if not ediff_res.success:
            raise RuntimeError("Failed to find best energy offset for the metric function.")

        eres = e0 + (ediff_res.x[0]*(self._exp_spectrum.energy - e0))
        result = metric_fun(self._exp_spectrum.mu[emask], xmu(eres)[emask])

        return result, ediff_res.x

    def calculate_mult(self, R):

        metric_fun = stat_scores[self._metric]
        energy_shift_fun = stat_scores[self._metric]
        if self._metric == 'spearman_corr':
            energy_shift_fun = stat_scores['l2_norm_dist']

        # return metric_fun(xmu(self._exp_spectrum.energy), self._exp_spectrum.mu)
        max_de = self._exp_spectrum.e0*0.003

        def xmu(energy):
            return sum(make_smoothing_spline(X.energy, X.mu, lam=0.1)(e) for X, e in zip(R, energy))

        def diff_func(x):
            xe = [self._exp_spectrum.energy + xx for xx in x]

            return energy_shift_fun(xmu(xe), self._exp_spectrum.mu)

        ediff_res = minimize(diff_func, [0.]*len(R), method='L-BFGS-B', bounds=[(-max_de, max_de)]*len(R))

        if not ediff_res.success:
            raise RuntimeError("Failed to find best energy offset for the metric function.")

        eres = [self._exp_spectrum.energy + xx for xx in ediff_res.x]
        result = metric_fun(xmu(eres), self._exp_spectrum.mu)

        return result, ediff_res.x