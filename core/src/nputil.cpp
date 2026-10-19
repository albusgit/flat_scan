#include "nputil.h"

namespace fs {

Vec polyfit(const double* x, const double* y, size_t n, int deg) {
    int m = deg + 1;
    // Vandermonde, highest power first, columns scaled to unit norm (as numpy does)
    std::vector<double> A(n * m);
    for (size_t i = 0; i < n; ++i) {
        double v = 1;
        for (int j = m - 1; j >= 0; --j) { A[i * m + j] = v; v *= x[i]; }
    }
    Vec scale(m, 0.0);
    for (int j = 0; j < m; ++j) {
        double s = 0;
        for (size_t i = 0; i < n; ++i) s += A[i * m + j] * A[i * m + j];
        scale[j] = std::sqrt(s);
        if (scale[j] == 0) scale[j] = 1;
        for (size_t i = 0; i < n; ++i) A[i * m + j] /= scale[j];
    }
    Vec b(y, y + n);
    // Householder QR
    for (int k = 0; k < m && (size_t)k < n; ++k) {
        double nrm = 0;
        for (size_t i = k; i < n; ++i) nrm += A[i * m + k] * A[i * m + k];
        nrm = std::sqrt(nrm);
        if (nrm == 0) continue;
        double alpha = A[k * m + k] > 0 ? -nrm : nrm;
        std::vector<double> v(n - k);
        for (size_t i = k; i < n; ++i) v[i - k] = A[i * m + k];
        v[0] -= alpha;
        double vn = 0;
        for (double t : v) vn += t * t;
        if (vn == 0) continue;
        for (int j = k; j < m; ++j) {
            double d = 0;
            for (size_t i = k; i < n; ++i) d += v[i - k] * A[i * m + j];
            d = 2 * d / vn;
            for (size_t i = k; i < n; ++i) A[i * m + j] -= d * v[i - k];
        }
        double d = 0;
        for (size_t i = k; i < n; ++i) d += v[i - k] * b[i];
        d = 2 * d / vn;
        for (size_t i = k; i < n; ++i) b[i] -= d * v[i - k];
    }
    Vec c(m, 0.0);
    int r = (int)std::min<size_t>(m, n);
    for (int k = r - 1; k >= 0; --k) {
        double s = b[k];
        for (int j = k + 1; j < m; ++j) s -= A[k * m + j] * c[j];
        c[k] = A[k * m + k] != 0 ? s / A[k * m + k] : 0;
    }
    for (int j = 0; j < m; ++j) c[j] /= scale[j];
    return c;
}

double percentile(Vec a, double q) {
    size_t n = a.size();
    if (n == 0) return NaN;
    for (double v : a) if (std::isnan(v)) return NaN;
    std::sort(a.begin(), a.end());
    double idx = q / 100.0 * (double)(n - 1);
    size_t lo = (size_t)std::floor(idx);
    if (lo >= n - 1) return a[n - 1];
    double t = idx - (double)lo, A = a[lo], B = a[lo + 1], d = B - A;
    return t >= 0.5 ? B - d * (1 - t) : A + d * t;
}

double median(Vec a) {
    size_t n = a.size();
    if (n == 0) return NaN;
    for (double v : a) if (std::isnan(v)) return NaN;
    std::sort(a.begin(), a.end());
    return n % 2 ? a[n / 2] : (a[n / 2 - 1] + a[n / 2]) / 2.0;
}

double np_sum(const double* a, size_t n) {
    if (n < 8) {
        double r = 0.;
        for (size_t i = 0; i < n; ++i) r += a[i];
        return r;
    }
    if (n <= 128) {
        double r[8];
        for (int j = 0; j < 8; ++j) r[j] = a[j];
        size_t i = 8;
        for (; i < n - (n % 8); i += 8)
            for (int j = 0; j < 8; ++j) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; ++i) res += a[i];
        return res;
    }
    size_t n2 = n / 2;
    n2 -= n2 % 8;
    return np_sum(a, n2) + np_sum(a + n2, n - n2);
}

Vec median_filter1d_nearest(const Vec& a, int size) {
    int n = (int)a.size(), h = size / 2;
    Vec out(n), w(size);
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < size; ++k) w[k] = a[std::min(n - 1, std::max(0, i - h + k))];
        std::nth_element(w.begin(), w.begin() + h, w.end());
        out[i] = w[h];
    }
    return out;
}

}  // namespace fs
