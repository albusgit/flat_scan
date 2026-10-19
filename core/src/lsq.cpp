#include "lsq.h"

#include <algorithm>
#include <cmath>
#include <thread>

#pragma STDC FP_CONTRACT OFF

namespace fs {

static const double EPS = 2.220446049250313e-16;

using Mat = std::vector<double>;   // row major m x n

static double dotv(const double* a, const double* b, size_t n) {
    double s = 0;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];
    return s;
}
static double norm(const Vec& v) { return std::sqrt(dotv(v.data(), v.data(), v.size())); }

// One-sided Jacobi SVD of an m x n matrix (m >= n): A = U diag(s) V^T, s descending.
static void svd(const Mat& A, int m, int n, Mat& U, Vec& s, Mat& V) {
    Mat B = A;                       // columns orthogonalised in place
    V.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i) V[(size_t)i * n + i] = 1.0;
    for (int sweep = 0; sweep < 60; ++sweep) {
        double off = 0;
        for (int p = 0; p < n - 1; ++p)
            for (int q = p + 1; q < n; ++q) {
                double a = 0, b = 0, c = 0;
                for (int i = 0; i < m; ++i) {
                    double x = B[(size_t)i * n + p], y = B[(size_t)i * n + q];
                    a += x * x; b += y * y; c += x * y;
                }
                if (std::fabs(c) <= 1e-300 || std::fabs(c) < 1e-16 * std::sqrt(a * b)) continue;
                off = std::max(off, std::fabs(c) / std::sqrt(a * b));
                double zeta = (b - a) / (2 * c);
                double t = (zeta >= 0 ? 1.0 : -1.0) / (std::fabs(zeta) + std::sqrt(1 + zeta * zeta));
                double cs = 1 / std::sqrt(1 + t * t), sn = cs * t;
                for (int i = 0; i < m; ++i) {
                    double x = B[(size_t)i * n + p], y = B[(size_t)i * n + q];
                    B[(size_t)i * n + p] = cs * x - sn * y;
                    B[(size_t)i * n + q] = sn * x + cs * y;
                }
                for (int i = 0; i < n; ++i) {
                    double x = V[(size_t)i * n + p], y = V[(size_t)i * n + q];
                    V[(size_t)i * n + p] = cs * x - sn * y;
                    V[(size_t)i * n + q] = sn * x + cs * y;
                }
            }
        if (off < 1e-15) break;
    }
    std::vector<int> ord(n);
    Vec sv(n);
    for (int j = 0; j < n; ++j) {
        double a = 0;
        for (int i = 0; i < m; ++i) a += B[(size_t)i * n + j] * B[(size_t)i * n + j];
        sv[j] = std::sqrt(a);
        ord[j] = j;
    }
    std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return sv[a] > sv[b]; });
    s.resize(n);
    U.assign((size_t)m * n, 0.0);
    Mat V2((size_t)n * n);
    for (int k = 0; k < n; ++k) {
        int j = ord[k];
        s[k] = sv[j];
        for (int i = 0; i < m; ++i) U[(size_t)i * n + k] = sv[j] > 0 ? B[(size_t)i * n + j] / sv[j] : 0.0;
        for (int i = 0; i < n; ++i) V2[(size_t)i * n + k] = V[(size_t)i * n + j];
    }
    V = V2;
}

struct Rho { Vec r0, r1, r2; };

// construct_loss_function('cauchy', f_scale)(f)  (scaled rho, as scipy returns it)
static Rho cauchy(const Vec& f, double fs) {
    Rho r;
    size_t m = f.size();
    r.r0.resize(m); r.r1.resize(m); r.r2.resize(m);
    double fs2 = fs * fs;
    for (size_t i = 0; i < m; ++i) {
        double q = f[i] / fs, z = q * q;
        double t = 1 + z;
        r.r0[i] = std::log1p(z) * fs2;
        r.r1[i] = 1 / t;
        r.r2[i] = (-1 / (t * t)) / fs2;
    }
    return r;
}
static double cauchy_cost(const Vec& f, double fs) {
    Vec l(f.size());
    for (size_t i = 0; i < f.size(); ++i) { double q = f[i] / fs; l[i] = std::log1p(q * q); }
    return 0.5 * (fs * fs) * np_sum(l);
}

// scale_for_robust_loss_function (in place)
static void scale_robust(Mat& J, Vec& f, const Rho& rho, int n) {
    for (size_t i = 0; i < f.size(); ++i) {
        double js = rho.r1[i] + 2 * rho.r2[i] * (f[i] * f[i]);
        if (js < EPS) js = EPS;
        js = std::sqrt(js);
        f[i] *= rho.r1[i] / js;
        for (int j = 0; j < n; ++j) J[i * n + j] *= js;
    }
}

static Vec grad(const Mat& J, const Vec& f, int n) {
    Vec g(n, 0.0);
    for (size_t i = 0; i < f.size(); ++i)
        for (int j = 0; j < n; ++j) g[j] += J[i * n + j] * f[i];
    return g;
}

// approx_derivative(fun, x, method='2-point', f0=f)
static Mat jacobian(const ResidualFn& fun, const Vec& x0, const Vec& f0, int threads) {
    int n = (int)x0.size();
    size_t m = f0.size();
    Mat J(m * n);
    const double rstep = std::sqrt(EPS);
    auto col = [&](int i) {
        Vec x1 = x0;
        double sign = x0[i] >= 0 ? 1.0 : -1.0;
        double h = rstep * sign * std::max(1.0, std::fabs(x0[i]));
        x1[i] += h;
        double dx = x1[i] - x0[i];
        Vec f1 = fun(x1);
        for (size_t r = 0; r < m; ++r) J[r * n + i] = (f1[r] - f0[r]) / dx;
    };
    if (threads <= 1) {
        for (int i = 0; i < n; ++i) col(i);
    } else {
        std::vector<std::thread> ts;
        for (int t = 0; t < threads; ++t)
            ts.emplace_back([&, t] { for (int i = t; i < n; i += threads) col(i); });
        for (auto& t : ts) t.join();
    }
    return J;
}

// common.solve_lsq_trust_region
static Vec solve_tr(int n, int m, const Vec& uf, const Vec& s, const Mat& V, double Delta, double& alpha) {
    Vec suf(n);
    for (int i = 0; i < n; ++i) suf[i] = s[i] * uf[i];
    bool full_rank = m >= n && s[n - 1] > EPS * m * s[0];
    auto apply_V = [&](const Vec& w) {
        Vec p(n, 0.0);
        for (int i = 0; i < n; ++i) {
            double acc = 0;
            for (int k = 0; k < n; ++k) acc += V[(size_t)i * n + k] * w[k];
            p[i] = -acc;
        }
        return p;
    };
    if (full_rank) {
        Vec w(n);
        for (int i = 0; i < n; ++i) w[i] = uf[i] / s[i];
        Vec p = apply_V(w);
        if (norm(p) <= Delta) { alpha = 0.0; return p; }
    }
    auto phi = [&](double a, double& phi_prime) {
        Vec q(n), r(n);
        for (int i = 0; i < n; ++i) {
            double den = s[i] * s[i] + a;
            q[i] = suf[i] / den;
            r[i] = (suf[i] * suf[i]) / (den * den * den);
        }
        double pn = norm(q);
        phi_prime = -np_sum(r) / pn;
        return pn - Delta;
    };
    double alpha_upper = norm(suf) / Delta, alpha_lower = 0.0, pp;
    if (full_rank) {
        double ph = phi(0.0, pp);
        alpha_lower = -ph / pp;
    }
    double a = alpha;
    if (!full_rank && alpha == 0) a = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
    for (int it = 0; it < 10; ++it) {
        if (a < alpha_lower || a > alpha_upper) a = std::max(0.001 * alpha_upper, std::sqrt(alpha_lower * alpha_upper));
        double ph = phi(a, pp);
        if (ph < 0) alpha_upper = a;
        double ratio = ph / pp;
        alpha_lower = std::max(alpha_lower, a - ratio);
        a -= (ph + Delta) * ratio / Delta;
        if (std::fabs(ph) < 0.01 * Delta) break;
    }
    Vec w(n);
    for (int i = 0; i < n; ++i) w[i] = suf[i] / (s[i] * s[i] + a);
    Vec p = apply_V(w);
    double sc = Delta / norm(p);
    for (double& v : p) v *= sc;
    alpha = a;
    return p;
}

LsqResult least_squares_cauchy(const ResidualFn& fun, const Vec& x0, double fs, int max_nfev, int threads) {
    const double ftol = 1e-8, xtol = 1e-8, gtol = 1e-8;
    int n = (int)x0.size();
    Vec x = x0;
    Vec f = fun(x);
    int m = (int)f.size();
    int nfev = 1, njev = 1;
    Mat J = jacobian(fun, x, f, threads);
    Rho rho = cauchy(f, fs);
    double cost = 0.5 * np_sum(rho.r0);
    scale_robust(J, f, rho, n);
    Vec g = grad(J, f, n);
    double Delta = norm(x0);
    if (Delta == 0) Delta = 1.0;
    double alpha = 0.0;
    int status = -1;
    double actual_reduction = 0, step_norm = 0;
    while (true) {
        double g_norm = 0;
        for (double v : g) g_norm = std::max(g_norm, std::fabs(v));
        if (g_norm < gtol) status = 1;
        if (status != -1 || nfev == max_nfev) break;
        Mat U; Vec s; Mat V;
        svd(J, m, n, U, s, V);
        Vec uf(n, 0.0);
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j) uf[j] += U[(size_t)i * n + j] * f[i];
        actual_reduction = -1;
        Vec x_new, f_new;
        double cost_new = 0;
        while (actual_reduction <= 0 && nfev < max_nfev) {
            Vec step = solve_tr(n, m, uf, s, V, Delta, alpha);
            // predicted_reduction = -evaluate_quadratic(J, g, step)
            Vec Js(m, 0.0);
            for (int i = 0; i < m; ++i) Js[i] = dotv(&J[(size_t)i * n], step.data(), n);
            double q = dotv(Js.data(), Js.data(), m), l = dotv(step.data(), g.data(), n);
            double predicted = -(0.5 * q + l);
            x_new = x;
            for (int i = 0; i < n; ++i) x_new[i] += step[i];
            f_new = fun(x_new);
            ++nfev;
            double step_h_norm = norm(step);
            bool finite = true;
            for (double v : f_new) finite &= std::isfinite(v);
            if (!finite) { Delta = 0.25 * step_h_norm; continue; }
            cost_new = cauchy_cost(f_new, fs);
            actual_reduction = cost - cost_new;
            // update_tr_radius
            double ratio;
            if (predicted > 0) ratio = actual_reduction / predicted;
            else if (predicted == actual_reduction && predicted == 0) ratio = 1;
            else ratio = 0;
            double Delta_new = Delta;
            if (ratio < 0.25) Delta_new = 0.25 * step_h_norm;
            else if (ratio > 0.75 && step_h_norm > 0.95 * Delta) Delta_new = Delta * 2.0;
            step_norm = step_h_norm;
            // check_termination
            bool ftol_ok = actual_reduction < ftol * cost && ratio > 0.25;
            bool xtol_ok = step_norm < xtol * (xtol + norm(x));
            if (ftol_ok && xtol_ok) status = 4;
            else if (ftol_ok) status = 2;
            else if (xtol_ok) status = 3;
            if (status != -1) break;
            alpha *= Delta / Delta_new;
            Delta = Delta_new;
        }
        if (actual_reduction > 0) {
            x = x_new;
            f = f_new;
            cost = cost_new;
            J = jacobian(fun, x, f, threads);
            ++njev;
            rho = cauchy(f, fs);
            scale_robust(J, f, rho, n);
            g = grad(J, f, n);
        } else {
            step_norm = 0;
            actual_reduction = 0;
        }
    }
    LsqResult r;
    r.x = x; r.cost = cost; r.nfev = nfev; r.njev = njev; r.status = status == -1 ? 0 : status;
    return r;
}

}  // namespace fs
