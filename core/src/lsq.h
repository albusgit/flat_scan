// Port of scipy.optimize.least_squares(fun, x0, method='trf', jac='2-point', loss='cauchy',
// f_scale, max_nfev) for unbounded problems (scipy 1.15.3: trf_no_bounds with
// tr_solver='exact', x_scale=1, ftol=xtol=gtol=1e-8). Same control flow, step sizes and
// termination tests; only BLAS/LAPACK rounding differs. The Jacobian columns are evaluated
// in parallel (each column is an independent residual evaluation, so this is deterministic).
#pragma once
#include <functional>

#include "nputil.h"

namespace fs {

struct LsqResult {
    Vec x;
    double cost = 0;     // robust cost 0.5 * sum(rho(f^2)), as scipy's result.cost
    int nfev = 0, njev = 0, status = 0;
};

using ResidualFn = std::function<Vec(const Vec&)>;

LsqResult least_squares_cauchy(const ResidualFn& fun, const Vec& x0, double f_scale, int max_nfev, int threads = 1);

}  // namespace fs
