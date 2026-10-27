#include "sparse.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <mutex>
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

namespace {

// The sparsity pattern of C = A * B (A: n rows, B: m columns): the columns each row of A reaches
// through B, sorted (every product counts, zero or not)
void spgemm_pattern(const std::vector<int>& aptr, const std::vector<int>& aidx, const std::vector<int>& bptr,
                    const std::vector<int>& bidx, int n, int m, std::vector<int>& cptr, std::vector<int>& cidx) {
    cptr.assign(n + 1, 0);
    cidx.clear();
    std::vector<int> mark(m, -1), cols;
    for (int i = 0; i < n; ++i) {
        cols.clear();
        for (int a = aptr[i]; a < aptr[i + 1]; ++a)
            for (int b = bptr[aidx[a]]; b < bptr[aidx[a] + 1]; ++b) {
                int j = bidx[b];
                if (mark[j] != i) { mark[j] = i; cols.push_back(j); }
            }
        std::sort(cols.begin(), cols.end());
        cidx.insert(cidx.end(), cols.begin(), cols.end());
        cptr[i + 1] = (int)cidx.size();
    }
}

// An operator: a pattern and its values
struct View {
    int n;
    const int *ptr, *idx;
    const double* val;
    void matvec(const double* x, double* y) const {
        for (int i = 0; i < n; ++i) {
            double s = 0;
            for (int k = ptr[i]; k < ptr[i + 1]; ++k) s += val[k] * x[idx[k]];
            y[i] = s;
        }
    }
};

// C = A * B on C's pattern (spgemm_pattern), each entry summed in the order of A's entries, then B's
// (a dense row accumulator, as the products were always summed)
void spgemm_values(const View& A, const View& B, const std::vector<int>& cptr, const std::vector<int>& cidx,
                   std::vector<double>& acc, std::vector<double>& cval) {
    cval.resize(cidx.size());
    for (int i = 0; i < A.n; ++i) {
        for (int k = cptr[i]; k < cptr[i + 1]; ++k) acc[cidx[k]] = 0;
        for (int a = A.ptr[i]; a < A.ptr[i + 1]; ++a) {
            const double av = A.val[a];
            const int r = A.idx[a];
            for (int b = B.ptr[r]; b < B.ptr[r + 1]; ++b) acc[B.idx[b]] += av * B.val[b];
        }
        for (int k = cptr[i]; k < cptr[i + 1]; ++k) cval[k] = acc[cidx[k]];
    }
}

Csr transpose(const Csr& A, int n_cols) {
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
Csr prolongation(int gw, int gh, int cw, int ch) {
    Csr P;
    P.n = gw * gh;
    P.ptr.assign(P.n + 1, 0);
    P.idx.reserve((size_t)P.n * 4);
    P.val.reserve((size_t)P.n * 4);
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
            std::pair<int, double> e[4];
            int ne = 0;
            for (int a = 0; a < ny; ++a)
                for (int b = 0; b < nx; ++b) e[ne++] = {jy[a] * cw + jx[b], wy[a] * wx[b]};
            std::sort(e, e + ne);
            for (int q = 0; q < ne; ++q) { P.idx.push_back(e[q].first); P.val.push_back(e[q].second); }
            P.ptr[iy * gw + ix + 1] = (int)P.idx.size();
        }
    return P;
}

// What the multigrid hierarchy of an operator takes from its grid and its sparsity pattern alone:
// the levels' sizes, patterns and diagonals, the prolongations and the patterns of the Galerkin products.
// Built once per grid size and pattern (GridPlan cache below), the values per operator.
struct LevelPlan {
    int gw = 0, gh = 0, n = 0;
    std::vector<int> ptr, idx, diag;   // this level's operator: pattern, the diagonal's entry per row
    Csr P, R;                          // P: this level <- the next coarser one, R = P^T (none on the coarsest)
    std::vector<int> ap_ptr, ap_idx;   // the pattern of A P
};

struct GridPlan {
    std::vector<LevelPlan> L;
};

std::shared_ptr<const GridPlan> build_plan(const Csr& M, int gw, int gh) {
    auto G = std::make_shared<GridPlan>();
    LevelPlan l0;
    l0.gw = gw; l0.gh = gh; l0.n = M.n;
    l0.ptr = M.ptr; l0.idx = M.idx;
    G->L.push_back(std::move(l0));
    while (G->L.back().n > 400 && G->L.back().gw > 3 && G->L.back().gh > 3) {
        LevelPlan& f = G->L.back();
        int cw = f.gw / 2 + 1, ch = f.gh / 2 + 1;
        f.P = prolongation(f.gw, f.gh, cw, ch);
        f.R = transpose(f.P, cw * ch);
        spgemm_pattern(f.ptr, f.idx, f.P.ptr, f.P.idx, f.n, cw * ch, f.ap_ptr, f.ap_idx);
        LevelPlan c;
        c.gw = cw; c.gh = ch; c.n = cw * ch;
        spgemm_pattern(f.R.ptr, f.R.idx, f.ap_ptr, f.ap_idx, c.n, c.n, c.ptr, c.idx);
        G->L.push_back(std::move(c));
    }
    for (auto& l : G->L) {
        l.diag.assign(l.n, -1);
        for (int i = 0; i < l.n; ++i)
            for (int k = l.ptr[i]; k < l.ptr[i + 1]; ++k)
                if (l.idx[k] == i) l.diag[i] = k;
    }
    return G;
}

// The last few plans (a page uses three grid sizes; the x and y fields and the IRLS re-solves share them)
std::shared_ptr<const GridPlan> plan_for(const Csr& M, int gw, int gh) {
    static std::mutex mu;
    static std::vector<std::shared_ptr<const GridPlan>> cache;   // most recent last
    std::lock_guard<std::mutex> lock(mu);
    for (size_t i = 0; i < cache.size(); ++i) {
        const LevelPlan& l = cache[i]->L[0];
        if (l.gw == gw && l.gh == gh && l.ptr == M.ptr && l.idx == M.idx) {
            auto p = cache[i];
            cache.erase(cache.begin() + (long)i);
            cache.push_back(p);
            return p;
        }
    }
    auto p = build_plan(M, gw, gh);
    cache.push_back(p);
    if (cache.size() > 4) cache.erase(cache.begin());
    return p;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

}  // namespace

// The hierarchy's values (constant once built) and the solves' scratch (one per solve)
struct SpdGridSolver::Impl {
    const Csr& M;
    std::shared_ptr<const GridPlan> plan;
    std::vector<std::vector<double>> val;   // the levels' operator values (level 0: M's own)
    std::vector<double> chol;               // dense Cholesky factor of the coarsest operator (row major)
    int nc = 0;

    struct Work {
        std::vector<std::vector<double>> x, b, r, e;
    };

    Impl(const Csr& m, int gw, int gh) : M(m), plan(plan_for(m, gw, gh)) {
        const auto& L = plan->L;
        val.resize(L.size());
        std::vector<double> acc(M.n), ap;
        for (size_t lev = 0; lev + 1 < L.size(); ++lev) {
            const LevelPlan& f = L[lev];
            View A = op(lev), P{f.P.n, f.P.ptr.data(), f.P.idx.data(), f.P.val.data()}, R{f.R.n, f.R.ptr.data(), f.R.idx.data(), f.R.val.data()};
            spgemm_values(A, P, f.ap_ptr, f.ap_idx, acc, ap);
            View AP{f.n, f.ap_ptr.data(), f.ap_idx.data(), ap.data()};
            spgemm_values(R, AP, L[lev + 1].ptr, L[lev + 1].idx, acc, val[lev + 1]);
        }
        // dense Cholesky of the coarsest level
        View A = op(L.size() - 1);
        int n = A.n;
        nc = n;
        chol.assign((size_t)n * n, 0.0);
        for (int i = 0; i < n; ++i)
            for (int k = A.ptr[i]; k < A.ptr[i + 1]; ++k) chol[(size_t)i * n + A.idx[k]] = A.val[k];
        for (int j = 0; j < n; ++j) {
            double d = chol[(size_t)j * n + j];
            for (int k = 0; k < j; ++k) d -= chol[(size_t)j * n + k] * chol[(size_t)j * n + k];
            if (d <= 0) throw std::runtime_error("coarse operator not SPD");
            d = std::sqrt(d);
            chol[(size_t)j * n + j] = d;
            for (int i = j + 1; i < n; ++i) {
                double s = chol[(size_t)i * n + j];
                for (int k = 0; k < j; ++k) s -= chol[(size_t)i * n + k] * chol[(size_t)j * n + k];
                chol[(size_t)i * n + j] = s / d;
            }
        }
    }

    View op(size_t lev) const {
        const LevelPlan& l = plan->L[lev];
        return View{l.n, l.ptr.data(), l.idx.data(), lev == 0 ? M.val.data() : val[lev].data()};
    }

    void gs_sweep(size_t lev, double* x, const double* b, bool forward) const {
        const View A = op(lev);
        const int* diag = plan->L[lev].diag.data();
        for (int c = 0; c < A.n; ++c) {
            int i = forward ? c : A.n - 1 - c;
            double s = b[i];
            // the row's entries in order, the diagonal left out
            for (int k = A.ptr[i]; k < diag[i]; ++k) s -= A.val[k] * x[A.idx[k]];
            for (int k = diag[i] + 1; k < A.ptr[i + 1]; ++k) s -= A.val[k] * x[A.idx[k]];
            x[i] = s / A.val[diag[i]];
        }
    }

    // one V-cycle on level lev: x from b
    void vcycle(Work& w, size_t lev, const double* b, double* x) const {
        const auto& L = plan->L;
        if (lev + 1 == L.size()) {   // coarsest: x = L^-T L^-1 b
            int n = nc;
            for (int i = 0; i < n; ++i) {
                double s = b[i];
                for (int k = 0; k < i; ++k) s -= chol[(size_t)i * n + k] * x[k];
                x[i] = s / chol[(size_t)i * n + i];
            }
            for (int i = n - 1; i >= 0; --i) {
                double s = x[i];
                for (int k = i + 1; k < n; ++k) s -= chol[(size_t)k * n + i] * x[k];
                x[i] = s / chol[(size_t)i * n + i];
            }
            return;
        }
        const LevelPlan& l = L[lev];
        const int nu = 2;
        std::fill(x, x + l.n, 0.0);
        for (int s = 0; s < nu; ++s) gs_sweep(lev, x, b, true);
        double* r = w.r[lev].data();
        op(lev).matvec(x, r);
        for (int i = 0; i < l.n; ++i) r[i] = b[i] - r[i];
        double *cb = w.b[lev + 1].data(), *cx = w.x[lev + 1].data(), *e = w.e[lev].data();
        l.R.matvec(r, cb);
        vcycle(w, lev + 1, cb, cx);
        l.P.matvec(cx, e);
        for (int i = 0; i < l.n; ++i) x[i] += e[i];
        for (int s = 0; s < nu; ++s) gs_sweep(lev, x, b, false);
    }
};

SpdGridSolver::SpdGridSolver(const Csr& M, int gw, int gh) : impl_(new Impl(M, gw, gh)) {}
SpdGridSolver::~SpdGridSolver() = default;

int SpdGridSolver::levels() const { return (int)impl_->plan->L.size(); }

std::vector<double> SpdGridSolver::solve(const std::vector<double>& b, const std::vector<double>* x0, double rtol, int max_iter,
                                         SolveStats* st) const {
    const Impl& H = *impl_;
    const Csr& M = H.M;
    int n = M.n;
    Impl::Work w;
    for (const auto& l : H.plan->L) {
        w.x.emplace_back(l.n); w.b.emplace_back(l.n); w.r.emplace_back(l.n); w.e.emplace_back(l.n);
    }
    std::vector<double> x = x0 ? *x0 : std::vector<double>(n, 0.0), r(n), z(n), p(n), q(n);
    M.matvec(x.data(), q.data());
    for (int i = 0; i < n; ++i) r[i] = b[i] - q[i];
    double bn = std::sqrt(dot(b, b));
    if (bn == 0) bn = 1;
    auto precond = [&](const std::vector<double>& in, std::vector<double>& out) { H.vcycle(w, 0, in.data(), out.data()); };
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
    if (st) { st->iterations = it; st->levels = levels(); st->rel_residual = rn / bn; }
    return x;
}

std::vector<double> solve_spd_grid(const Csr& M, const std::vector<double>& b, int gw, int gh,
                                   const std::vector<double>* x0, double rtol, int max_iter, SolveStats* st) {
    return SpdGridSolver(M, gw, gh).solve(b, x0, rtol, max_iter, st);
}

}  // namespace fs
