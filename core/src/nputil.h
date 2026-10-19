// numpy / scipy helpers with the reference's exact semantics (float64 throughout).
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace fs {

using Vec = std::vector<double>;
static const double NaN = std::numeric_limits<double>::quiet_NaN();

// np.linspace(a, b, n): y[i] = i*step + a, last element exactly b
inline Vec linspace(double a, double b, int n) {
    Vec y(n);
    if (n == 1) { y[0] = a; return y; }
    double step = (b - a) / (n - 1);
    for (int i = 0; i < n; ++i) y[i] = i * step + a;
    y[n - 1] = b;
    return y;
}

// np.interp(x, xp, fp): xp increasing; clamps to fp[0] / fp[-1] outside
inline double interp(double x, const double* xp, const double* fp, size_t n) {
    if (n == 0) return NaN;
    if (std::isnan(x)) return x;
    if (x <= xp[0]) return x < xp[0] ? fp[0] : fp[0];
    if (x >= xp[n - 1]) return fp[n - 1];
    size_t j = std::upper_bound(xp, xp + n, x) - xp - 1;   // xp[j] <= x < xp[j+1]
    double dx = xp[j + 1] - xp[j];
    double slope = (fp[j + 1] - fp[j]) / dx;
    double r = slope * (x - xp[j]) + fp[j];
    if (std::isnan(r)) {   // numpy fallback for inf slope
        r = slope * (x - xp[j + 1]) + fp[j + 1];
        if (std::isnan(r) && fp[j] == fp[j + 1]) r = fp[j];
    }
    return r;
}
inline double interp(double x, const Vec& xp, const Vec& fp) { return interp(x, xp.data(), fp.data(), xp.size()); }

// np.polyval (Horner, highest power first)
inline double polyval(const Vec& p, double x) {
    double y = 0;
    for (double c : p) y = y * x + c;
    return y;
}

// np.polyfit(x, y, deg): column-scaled Vandermonde, least squares (Householder QR in place
// of LAPACK gelsd; identical to rounding for these well-posed fits)
Vec polyfit(const double* x, const double* y, size_t n, int deg);
inline Vec polyfit(const Vec& x, const Vec& y, int deg) { return polyfit(x.data(), y.data(), x.size(), deg); }

// np.percentile(a, q) with the default 'linear' method
double percentile(Vec a, double q);
inline double median(const Vec& a) { return percentile(a, 50.0); }

// scipy.ndimage.median_filter(a, size, mode='nearest') for 1-D input
Vec median_filter1d_nearest(const Vec& a, int size);

}  // namespace fs
