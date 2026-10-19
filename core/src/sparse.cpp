#include "sparse.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace fs {

void Csr::matvec(const double* x, double* y) const {
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int k = ptr[i]; k < ptr[i + 1]; ++k) s += val[k] * x[idx[k]];
        y[i] = s;
    }
}

Csr csr_from_triplets(int n_rows, int n_cols, std::vector<int64_t>& keys, std::vector<double>& vals) {
    std::vector<size_t> ord(keys.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b) { return keys[a] < keys[b]; });
    Csr M;
    M.n = n_rows;
    M.ptr.assign(n_rows + 1, 0);
    int64_t last = -1;
    for (size_t o : ord) {
        if (keys[o] == last) { M.val.back() += vals[o]; continue; }
        last = keys[o];
        int r = (int)(keys[o] / n_cols);
        M.idx.push_back((int)(keys[o] % n_cols));
        M.val.push_back(vals[o]);
        M.ptr[r + 1]++;
    }
    for (int i = 0; i < n_rows; ++i) M.ptr[i + 1] += M.ptr[i];
    return M;
}

// C = A * B (A: n x k, B: k x m), dense row accumulator
static Csr spgemm(const Csr& A, const Csr& B, int m) {
    Csr C;
    C.n = A.n;
    C.ptr.assign(A.n + 1, 0);
    std::vector<double> acc(m, 0.0);
    std::vector<int> mark(m, -1), cols;
    for (int i = 0; i < A.n; ++i) {
        cols.clear();
        for (int a = A.ptr[i]; a < A.ptr[i + 1]; ++a) {
            int k = A.idx[a];
            double av = A.val[a];
            for (int b = B.ptr[k]; b < B.ptr[k + 1]; ++b) {
                int j = B.idx[b];
                if (mark[j] != i) { mark[j] = i; acc[j] = 0; cols.push_back(j); }
                acc[j] += av * B.val[b];
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int j : cols) { C.idx.push_back(j); C.val.push_back(acc[j]); }
        C.ptr[i + 1] = (int)C.idx.size();
    }
    return C;
}

static Csr transpose(const Csr& A, int n_cols) {
    Csr T;
    T.n = n_cols;
    T.ptr.assign(n_cols + 1, 0);
    for (int j : A.idx) T.ptr[j + 1]++;
    for (int i = 0; i < n_cols; ++i) T.ptr[i + 1] += T.ptr[i];
    T.idx.resize(A.idx.size());
    T.val.resize(A.val.size());
    std::vector<int> pos(T.ptr.begin(), T.ptr.end() - 1);
    for (int i = 0; i < A.n; ++i)
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) {
            int p = pos[A.idx[k]]++;
            T.idx[p] = i;
            T.val[p] = A.val[k];
        }
    return T;
}

// bilinear prolongation from a (cw x ch) grid with twice the spacing to (gw x gh)
static Csr prolongation(int gw, int gh, int cw, int ch) {
    Csr P;
    P.n = gw * gh;
    P.ptr.assign(P.n + 1, 0);
    auto w1d = [](int i, int nc, int* j, double* w) {
        if (i % 2 == 0) { j[0] = std::min(i / 2, nc - 1); w[0] = 1; return 1; }
        int a = std::min((i - 1) / 2, nc - 1), b = std::min((i + 1) / 2, nc - 1);
        if (a == b) { j[0] = a; w[0] = 1; return 1; }
        j[0] = a; w[0] = 0.5; j[1] = b; w[1] = 0.5;
        return 2;
    };
    for (int iy = 0; iy < gh; ++iy)
        for (int ix = 0; ix < gw; ++ix) {
            int jx[2], jy[2];
            double wx[2], wy[2];
            int nx = w1d(ix, cw, jx, wx), ny = w1d(iy, ch, jy, wy);
            std::vector<std::pair<int, double>> e;
            for (int a = 0; a < ny; ++a)
                for (int b = 0; b < nx; ++b) e.emplace_back(jy[a] * cw + jx[b], wy[a] * wx[b]);
            std::sort(e.begin(), e.end());
            for (auto& q : e) { P.idx.push_back(q.first); P.val.push_back(q.second); }
            P.ptr[iy * gw + ix + 1] = (int)P.idx.size();
        }
    return P;
}

struct Level {
    Csr A, P, R;           // P: this level <- next coarser level; R = P^T
    std::vector<int> diag;
    int gw, gh;
    std::vector<double> x, b, r;
};

struct Hierarchy {
    std::vector<Level> L;
    std::vector<double> chol;   // dense Cholesky factor of the coarsest operator (row major)
    int nc = 0;
};

static void find_diag(Level& l) {
    l.diag.assign(l.A.n, -1);
    for (int i = 0; i < l.A.n; ++i)
        for (int k = l.A.ptr[i]; k < l.A.ptr[i + 1]; ++k)
            if (l.A.idx[k] == i) l.diag[i] = k;
}

static Hierarchy build(const Csr& M, int gw, int gh) {
    Hierarchy H;
    H.L.push_back(Level{M, {}, {}, {}, gw, gh, {}, {}, {}});
    while (H.L.back().A.n > 400 && H.L.back().gw > 3 && H.L.back().gh > 3) {
        Level& f = H.L.back();
        int cw = f.gw / 2 + 1, ch = f.gh / 2 + 1;
        f.P = prolongation(f.gw, f.gh, cw, ch);
        f.R = transpose(f.P, cw * ch);
        Csr AP = spgemm(f.A, f.P, cw * ch);
        Csr Ac = spgemm(f.R, AP, cw * ch);
        H.L.push_back(Level{Ac, {}, {}, {}, cw, ch, {}, {}, {}});
    }
    for (auto& l : H.L) {
        find_diag(l);
        l.x.resize(l.A.n); l.b.resize(l.A.n); l.r.resize(l.A.n);
    }
    // dense Cholesky of the coarsest level
    const Csr& A = H.L.back().A;
    int n = A.n;
    H.nc = n;
    H.chol.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) H.chol[(size_t)i * n + A.idx[k]] = A.val[k];
    for (int j = 0; j < n; ++j) {
        double d = H.chol[(size_t)j * n + j];
        for (int k = 0; k < j; ++k) d -= H.chol[(size_t)j * n + k] * H.chol[(size_t)j * n + k];
        if (d <= 0) throw std::runtime_error("coarse operator not SPD");
        d = std::sqrt(d);
        H.chol[(size_t)j * n + j] = d;
        for (int i = j + 1; i < n; ++i) {
            double s = H.chol[(size_t)i * n + j];
            for (int k = 0; k < j; ++k) s -= H.chol[(size_t)i * n + k] * H.chol[(size_t)j * n + k];
            H.chol[(size_t)i * n + j] = s / d;
        }
    }
    return H;
}

static void gs_sweep(const Level& l, std::vector<double>& x, const std::vector<double>& b, bool forward) {
    const Csr& A = l.A;
    for (int c = 0; c < A.n; ++c) {
        int i = forward ? c : A.n - 1 - c;
        double s = b[i];
        for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k)
            if (A.idx[k] != i) s -= A.val[k] * x[A.idx[k]];
        x[i] = s / A.val[l.diag[i]];
    }
}

static void vcycle(Hierarchy& H, size_t lev) {
    Level& l = H.L[lev];
    if (lev + 1 == H.L.size()) {   // coarsest: x = L^-T L^-1 b
        int n = H.nc;
        std::vector<double>& x = l.x;
        for (int i = 0; i < n; ++i) {
            double s = l.b[i];
            for (int k = 0; k < i; ++k) s -= H.chol[(size_t)i * n + k] * x[k];
            x[i] = s / H.chol[(size_t)i * n + i];
        }
        for (int i = n - 1; i >= 0; --i) {
            double s = x[i];
            for (int k = i + 1; k < n; ++k) s -= H.chol[(size_t)k * n + i] * x[k];
            x[i] = s / H.chol[(size_t)i * n + i];
        }
        return;
    }
    const int nu = 2;
    std::fill(l.x.begin(), l.x.end(), 0.0);
    for (int s = 0; s < nu; ++s) gs_sweep(l, l.x, l.b, true);
    l.A.matvec(l.x.data(), l.r.data());
    for (int i = 0; i < l.A.n; ++i) l.r[i] = l.b[i] - l.r[i];
    Level& c = H.L[lev + 1];
    l.R.matvec(l.r.data(), c.b.data());
    vcycle(H, lev + 1);
    std::vector<double> e(l.A.n);
    l.P.matvec(c.x.data(), e.data());
    for (int i = 0; i < l.A.n; ++i) l.x[i] += e[i];
    for (int s = 0; s < nu; ++s) gs_sweep(l, l.x, l.b, false);
}

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

std::vector<double> solve_spd_grid(const Csr& M, const std::vector<double>& b, int gw, int gh,
                                   const std::vector<double>* x0, double rtol, int max_iter, SolveStats* st) {
    int n = M.n;
    Hierarchy H = build(M, gw, gh);
    std::vector<double> x = x0 ? *x0 : std::vector<double>(n, 0.0), r(n), z(n), p(n), q(n);
    M.matvec(x.data(), q.data());
    for (int i = 0; i < n; ++i) r[i] = b[i] - q[i];
    double bn = std::sqrt(dot(b, b));
    if (bn == 0) bn = 1;
    auto precond = [&](const std::vector<double>& in, std::vector<double>& out) {
        H.L[0].b = in;
        vcycle(H, 0);
        out = H.L[0].x;
    };
    precond(r, z);
    p = z;
    double rz = dot(r, z), rn = std::sqrt(dot(r, r));
    int it = 0;
    while (rn / bn > rtol && it < max_iter) {
        M.matvec(p.data(), q.data());
        double alpha = rz / dot(p, q);
        for (int i = 0; i < n; ++i) { x[i] += alpha * p[i]; r[i] -= alpha * q[i]; }
        rn = std::sqrt(dot(r, r));
        ++it;
        if (rn / bn <= rtol) break;
        precond(r, z);
        double rz2 = dot(r, z);
        double beta = rz2 / rz;
        rz = rz2;
        for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
    }
    if (st) { st->iterations = it; st->levels = (int)H.L.size(); st->rel_residual = rn / bn; }
    return x;
}

}  // namespace fs
