// Sparse SPD solves for the warp-map fields (warpmap.solve): the regularised least-squares
// normal equations on a gw x gh grid. Solved by conjugate gradients preconditioned with a
// geometric multigrid V-cycle (Galerkin coarse operators from bilinear prolongation,
// symmetric Gauss-Seidel smoothing, dense Cholesky on the coarsest grid), optionally
// warm-started. The reference uses SuperLU (pass 1, global stage) and CG + smoothed-
// aggregation AMG at tol 1e-5 (pass 2); both are within 0.005 px of the exact solution
// (tools/solver_audit.py), so this solver targets the exact solution with a tight tolerance.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

namespace fs {

struct Csr {
    int n = 0;
    std::vector<int> ptr, idx;
    std::vector<double> val;
    void matvec(const double* x, double* y) const;
};

// Build from (row, col, value) triplets; duplicates are summed.
Csr csr_from_triplets(int n_rows, int n_cols, std::vector<int64_t>& keys, std::vector<double>& vals);

struct SolveStats {
    int iterations = 0, levels = 0;
    double rel_residual = 0;
};

// The multigrid hierarchy of M on a gw x gh grid (node index = iy * gw + ix), for one or more
// right-hand sides. What depends only on the grid and M's sparsity pattern (the levels, prolongations
// and the Galerkin products' patterns) is built once per grid size and pattern and kept for the next
// operators; the values are M's. solve is const: several threads can solve with one hierarchy. M must
// outlive the solver.
class SpdGridSolver {
public:
    SpdGridSolver(const Csr& M, int gw, int gh);
    ~SpdGridSolver();
    std::vector<double> solve(const std::vector<double>& b, const std::vector<double>* x0 = nullptr, double rtol = 1e-10,
                              int max_iter = 1000, SolveStats* st = nullptr) const;
    int levels() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Solve M x = b for M on a gw x gh grid (node index = iy * gw + ix).
std::vector<double> solve_spd_grid(const Csr& M, const std::vector<double>& b, int gw, int gh,
                                   const std::vector<double>* x0 = nullptr, double rtol = 1e-10,
                                   int max_iter = 1000, SolveStats* st = nullptr);

}  // namespace fs
