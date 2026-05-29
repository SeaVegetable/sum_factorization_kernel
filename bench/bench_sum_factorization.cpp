#include "sum_factorization.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{

template<typename Kernel>
struct DimArrayHelper;

template<int p, int nq>
struct DimArrayHelper<SumFactorizationKernel<2, p, nq>>
{
    static constexpr int dim = 2;
};

template<int p, int nq>
struct DimArrayHelper<SumFactorizationKernel<3, p, nq>>
{
    static constexpr int dim = 3;
};

template<typename Kernel>
constexpr int kernel_dim()
{
    return DimArrayHelper<Kernel>::dim;
}

template<typename Array, typename RNG>
void fill_random(Array& values, RNG& rng, const double lo = -1.0, const double hi = 1.0)
{
    std::uniform_real_distribution<double> dist(lo, hi);
    for (auto& v : values)
        v = dist(rng);
}

template<typename Kernel, typename RNG>
std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()> make_extraction(RNG& rng)
{
    std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()> C{};
    std::uniform_real_distribution<double> dist(-0.5, 0.5);

    for (int d = 0; d < kernel_dim<Kernel>(); ++d)
        for (int i = 0; i < Kernel::n1d; ++i)
            for (int a = 0; a < Kernel::n1d; ++a)
                C[d][i * Kernel::n1d + a] = (i == a ? 1.0 : 0.0) + 0.1 * dist(rng);

    return C;
}

template<typename Kernel>
std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()> make_identity_extraction()
{
    std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()> C{};

    for (int d = 0; d < kernel_dim<Kernel>(); ++d)
        for (int i = 0; i < Kernel::n1d; ++i)
            for (int a = 0; a < Kernel::n1d; ++a)
                C[d][i * Kernel::n1d + a] = (i == a ? 1.0 : 0.0);

    return C;
}

template<typename Kernel, typename RNG>
std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()> make_basis(RNG& rng)
{
    std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()> B{};
    for (int d = 0; d < kernel_dim<Kernel>(); ++d)
        fill_random(B[d], rng, -0.8, 0.8);
    return B;
}

template<typename Kernel, typename RNG>
typename Kernel::QPointMatrix make_quadrature_matrix(RNG& rng)
{
    typename Kernel::QPointMatrix D{};
    std::uniform_real_distribution<double> dist(-0.3, 0.3);

    for (int i = 0; i < Kernel::nqp; ++i)
        for (int j = 0; j < Kernel::nqp; ++j)
            D[i * Kernel::nqp + j] = dist(rng);

    for (int i = 0; i < Kernel::nqp; ++i)
        D[i * Kernel::nqp + i] += 1.0;

    return D;
}

template<typename Kernel, typename RNG>
typename Kernel::QPointDiagonal make_diagonal_quadrature(RNG& rng)
{
    typename Kernel::QPointDiagonal D_diag{};
    std::uniform_real_distribution<double> dist(0.7, 1.3);
    for (int q = 0; q < Kernel::nqp; ++q)
        D_diag[q] = dist(rng);
    return D_diag;
}

template<typename Kernel>
using DenseDofMatrix =
    std::vector<double>;

template<typename Kernel>
using DenseQMatrix =
    std::vector<double>;

template<typename Kernel>
DenseDofMatrix<Kernel> build_dense_extraction(
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C)
{
    DenseDofMatrix<Kernel> E(Kernel::ndofs * Kernel::ndofs, 0.0);

    if constexpr (kernel_dim<Kernel>() == 2)
    {
        for (int j = 0; j < Kernel::n1d; ++j)
            for (int i = 0; i < Kernel::n1d; ++i)
                for (int ay = 0; ay < Kernel::n1d; ++ay)
                    for (int ax = 0; ax < Kernel::n1d; ++ax)
                    {
                        const int row = Kernel::idx_2d(i, j);
                        const int col = Kernel::idx_2d(ax, ay);
                        E[row * Kernel::ndofs + col] =
                            C[0][i * Kernel::n1d + ax] *
                            C[1][j * Kernel::n1d + ay];
                    }
    }
    else
    {
        for (int k = 0; k < Kernel::n1d; ++k)
            for (int j = 0; j < Kernel::n1d; ++j)
                for (int i = 0; i < Kernel::n1d; ++i)
                    for (int az = 0; az < Kernel::n1d; ++az)
                        for (int ay = 0; ay < Kernel::n1d; ++ay)
                            for (int ax = 0; ax < Kernel::n1d; ++ax)
                            {
                                const int row = Kernel::idx_3d(i, j, k);
                                const int col = Kernel::idx_3d(ax, ay, az);
                                E[row * Kernel::ndofs + col] =
                                    C[0][i * Kernel::n1d + ax] *
                                    C[1][j * Kernel::n1d + ay] *
                                    C[2][k * Kernel::n1d + az];
                            }
    }

    return E;
}

template<typename Kernel>
DenseQMatrix<Kernel> build_dense_bernstein_eval(
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B)
{
    DenseQMatrix<Kernel> M(Kernel::nqp * Kernel::ndofs, 0.0);

    if constexpr (kernel_dim<Kernel>() == 2)
    {
        for (int qy = 0; qy < Kernel::nq1d; ++qy)
            for (int qx = 0; qx < Kernel::nq1d; ++qx)
                for (int ay = 0; ay < Kernel::n1d; ++ay)
                    for (int ax = 0; ax < Kernel::n1d; ++ax)
                    {
                        const int row = Kernel::qidx_2d(qx, qy);
                        const int col = Kernel::idx_2d(ax, ay);
                        M[row * Kernel::ndofs + col] =
                            B[0][qx * Kernel::n1d + ax] *
                            B[1][qy * Kernel::n1d + ay];
                    }
    }
    else
    {
        for (int qz = 0; qz < Kernel::nq1d; ++qz)
            for (int qy = 0; qy < Kernel::nq1d; ++qy)
                for (int qx = 0; qx < Kernel::nq1d; ++qx)
                    for (int az = 0; az < Kernel::n1d; ++az)
                        for (int ay = 0; ay < Kernel::n1d; ++ay)
                            for (int ax = 0; ax < Kernel::n1d; ++ax)
                            {
                                const int row = Kernel::qidx_3d(qx, qy, qz);
                                const int col = Kernel::idx_3d(ax, ay, az);
                                M[row * Kernel::ndofs + col] =
                                    B[0][qx * Kernel::n1d + ax] *
                                    B[1][qy * Kernel::n1d + ay] *
                                    B[2][qz * Kernel::n1d + az];
                            }
    }

    return M;
}

template<typename AMatrix, typename XArray>
std::vector<double> dense_matvec(
    const AMatrix& A,
    const XArray& x,
    const int nrows,
    const int ncols)
{
    std::vector<double> y(nrows, 0.0);
    for (int i = 0; i < nrows; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < ncols; ++j)
            sum += A[i * ncols + j] * x[j];
        y[i] = sum;
    }
    return y;
}

template<typename AMatrix, typename XArray>
std::vector<double> dense_transpose_matvec(
    const AMatrix& A,
    const XArray& x,
    const int nrows,
    const int ncols)
{
    std::vector<double> y(ncols, 0.0);
    for (int j = 0; j < ncols; ++j)
    {
        double sum = 0.0;
        for (int i = 0; i < nrows; ++i)
            sum += A[i * ncols + j] * x[i];
        y[j] = sum;
    }
    return y;
}

struct BenchResult
{
    double seconds;
};

template<typename Kernel>
BenchResult run_sum_factorized_benchmark(
    const typename Kernel::LocalArray& u_init,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointMatrix& D,
    const int warmup,
    const int repetitions)
{
    typename Kernel::LocalArray u = u_init;
    typename Kernel::LocalArray v{};

    for (int i = 0; i < warmup; ++i)
        Kernel::apply_mass(u, C, B, D, v);

    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        Kernel::apply_mass(u, C, B, D, v);
        u[rep % Kernel::ndofs] += 1e-14 * v[rep % Kernel::ndofs];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count()};
}

template<typename Kernel>
BenchResult run_sum_factorized_benchmark_diagonal(
    const typename Kernel::LocalArray& u_init,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointDiagonal& D_diag,
    const int warmup,
    const int repetitions)
{
    typename Kernel::LocalArray u = u_init;
    typename Kernel::LocalArray v{};

    for (int i = 0; i < warmup; ++i)
        Kernel::apply_mass_diagonal(u, C, B, D_diag, v);

    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        Kernel::apply_mass_diagonal(u, C, B, D_diag, v);
        u[rep % Kernel::ndofs] += 1e-14 * v[rep % Kernel::ndofs];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count()};
}

template<typename Kernel>
BenchResult run_dense_baseline_benchmark(
    const typename Kernel::LocalArray& u_init,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointMatrix& D,
    const int warmup,
    const int repetitions)
{
    typename Kernel::LocalArray u = u_init;
    const DenseDofMatrix<Kernel> E = build_dense_extraction<Kernel>(C);
    const DenseQMatrix<Kernel> M = build_dense_bernstein_eval<Kernel>(B);

    for (int i = 0; i < warmup; ++i)
    {
        const auto u_bernstein = dense_transpose_matvec(E, u, Kernel::ndofs, Kernel::ndofs);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        const auto weighted_q = dense_matvec(D, values_q, Kernel::nqp, Kernel::nqp);
        const auto v_bernstein = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        const auto v = dense_matvec(E, v_bernstein, Kernel::ndofs, Kernel::ndofs);
        (void)v;
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        const auto u_bernstein = dense_transpose_matvec(E, u, Kernel::ndofs, Kernel::ndofs);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        const auto weighted_q = dense_matvec(D, values_q, Kernel::nqp, Kernel::nqp);
        const auto v_bernstein = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        const auto v = dense_matvec(E, v_bernstein, Kernel::ndofs, Kernel::ndofs);
        u[rep % Kernel::ndofs] += 1e-14 * v[rep % Kernel::ndofs];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count()};
}

template<typename Kernel>
BenchResult run_dense_baseline_benchmark_diagonal(
    const typename Kernel::LocalArray& u_init,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointDiagonal& D_diag,
    const int warmup,
    const int repetitions)
{
    typename Kernel::LocalArray u = u_init;
    const DenseDofMatrix<Kernel> E = build_dense_extraction<Kernel>(C);
    const DenseQMatrix<Kernel> M = build_dense_bernstein_eval<Kernel>(B);

    for (int i = 0; i < warmup; ++i)
    {
        const auto u_bernstein = dense_transpose_matvec(E, u, Kernel::ndofs, Kernel::ndofs);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        std::vector<double> weighted_q(Kernel::nqp, 0.0);
        for (int q = 0; q < Kernel::nqp; ++q)
            weighted_q[q] = D_diag[q] * values_q[q];
        const auto v_bernstein = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        const auto v = dense_matvec(E, v_bernstein, Kernel::ndofs, Kernel::ndofs);
        (void)v;
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        const auto u_bernstein = dense_transpose_matvec(E, u, Kernel::ndofs, Kernel::ndofs);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        std::vector<double> weighted_q(Kernel::nqp, 0.0);
        for (int q = 0; q < Kernel::nqp; ++q)
            weighted_q[q] = D_diag[q] * values_q[q];
        const auto v_bernstein = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        const auto v = dense_matvec(E, v_bernstein, Kernel::ndofs, Kernel::ndofs);
        u[rep % Kernel::ndofs] += 1e-14 * v[rep % Kernel::ndofs];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count()};
}

template<typename Kernel>
BenchResult run_mixed_csf_bdense_benchmark(
    const typename Kernel::LocalArray& u_init,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointMatrix& D,
    const int warmup,
    const int repetitions)
{
    typename Kernel::LocalArray u = u_init;
    typename Kernel::LocalArray u_bernstein{};
    typename Kernel::LocalArray v_bernstein{};
    typename Kernel::LocalArray v{};
    const DenseQMatrix<Kernel> M = build_dense_bernstein_eval<Kernel>(B);

    for (int i = 0; i < warmup; ++i)
    {
        Kernel::apply_extraction_transpose(u, C, u_bernstein);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        const auto weighted_q = dense_matvec(D, values_q, Kernel::nqp, Kernel::nqp);
        const auto v_bernstein_vec = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        for (int i = 0; i < Kernel::ndofs; ++i)
            v_bernstein[i] = v_bernstein_vec[i];
        Kernel::apply_extraction(v_bernstein, C, v);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        Kernel::apply_extraction_transpose(u, C, u_bernstein);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        const auto weighted_q = dense_matvec(D, values_q, Kernel::nqp, Kernel::nqp);
        const auto v_bernstein_vec = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        for (int i = 0; i < Kernel::ndofs; ++i)
            v_bernstein[i] = v_bernstein_vec[i];
        Kernel::apply_extraction(v_bernstein, C, v);
        u[rep % Kernel::ndofs] += 1e-14 * v[rep % Kernel::ndofs];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count()};
}

template<typename Kernel>
BenchResult run_mixed_csf_bdense_benchmark_diagonal(
    const typename Kernel::LocalArray& u_init,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointDiagonal& D_diag,
    const int warmup,
    const int repetitions)
{
    typename Kernel::LocalArray u = u_init;
    typename Kernel::LocalArray u_bernstein{};
    typename Kernel::LocalArray v_bernstein{};
    typename Kernel::LocalArray v{};
    const DenseQMatrix<Kernel> M = build_dense_bernstein_eval<Kernel>(B);

    for (int i = 0; i < warmup; ++i)
    {
        Kernel::apply_extraction_transpose(u, C, u_bernstein);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        std::vector<double> weighted_q(Kernel::nqp, 0.0);
        for (int q = 0; q < Kernel::nqp; ++q)
            weighted_q[q] = D_diag[q] * values_q[q];
        const auto v_bernstein_vec = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        for (int i = 0; i < Kernel::ndofs; ++i)
            v_bernstein[i] = v_bernstein_vec[i];
        Kernel::apply_extraction(v_bernstein, C, v);
    }

    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        Kernel::apply_extraction_transpose(u, C, u_bernstein);
        const auto values_q = dense_matvec(M, u_bernstein, Kernel::nqp, Kernel::ndofs);
        std::vector<double> weighted_q(Kernel::nqp, 0.0);
        for (int q = 0; q < Kernel::nqp; ++q)
            weighted_q[q] = D_diag[q] * values_q[q];
        const auto v_bernstein_vec = dense_transpose_matvec(M, weighted_q, Kernel::nqp, Kernel::ndofs);
        for (int i = 0; i < Kernel::ndofs; ++i)
            v_bernstein[i] = v_bernstein_vec[i];
        Kernel::apply_extraction(v_bernstein, C, v);
        u[rep % Kernel::ndofs] += 1e-14 * v[rep % Kernel::ndofs];
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count()};
}

template<typename Kernel>
void print_result(
    const char* impl,
    const char* d_mode,
    const char* extraction_mode,
    const int repetitions,
    const BenchResult& result)
{
    const double ns_per_apply = 1.0e9 * result.seconds / static_cast<double>(repetitions);
    const double apps_per_sec = static_cast<double>(repetitions) / result.seconds;

    std::cout << impl
              << ',' << d_mode
              << ',' << extraction_mode
              << ',' << kernel_dim<Kernel>()
              << ',' << (Kernel::n1d - 1)
              << ',' << Kernel::nq1d
              << ',' << Kernel::ndofs
              << ',' << Kernel::nqp
              << ',' << repetitions
              << ',' << std::fixed << std::setprecision(6) << result.seconds
              << ',' << std::fixed << std::setprecision(2) << ns_per_apply
              << ',' << std::fixed << std::setprecision(2) << apps_per_sec
              << '\n';
}

template<typename Kernel>
void bench_case_diagonal(const char* extraction_mode, const bool identity_extraction, const int repetitions)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        kernel_dim<Kernel>() * 10000 + Kernel::n1d * 100 + Kernel::nq1d * 10 +
        (identity_extraction ? 1 : 7)));

    typename Kernel::LocalArray u{};
    fill_random(u, rng);
    const auto C = identity_extraction ? make_identity_extraction<Kernel>() : make_extraction<Kernel>(rng);
    const auto B = make_basis<Kernel>(rng);
    const auto D_diag = make_diagonal_quadrature<Kernel>(rng);

    constexpr int warmup = 100;
    const BenchResult sf_result =
        run_sum_factorized_benchmark_diagonal<Kernel>(u, C, B, D_diag, warmup, repetitions);
    const BenchResult mixed_result =
        run_mixed_csf_bdense_benchmark_diagonal<Kernel>(u, C, B, D_diag, warmup, repetitions);
    const BenchResult dense_result =
        run_dense_baseline_benchmark_diagonal<Kernel>(u, C, B, D_diag, warmup, repetitions);

    print_result<Kernel>("sum_factorized", "diagonal_D", extraction_mode, repetitions, sf_result);
    print_result<Kernel>("sf_C_dense_B", "diagonal_D", extraction_mode, repetitions, mixed_result);
    print_result<Kernel>("naive_dense", "diagonal_D", extraction_mode, repetitions, dense_result);
}

void run_p_study_2d()
{
    bench_case_diagonal<SumFactorizationKernel<2, 3, 4>>("general_C", false, 250000);
    bench_case_diagonal<SumFactorizationKernel<2, 4, 5>>("general_C", false, 120000);
    bench_case_diagonal<SumFactorizationKernel<2, 5, 6>>("general_C", false, 50000);
    bench_case_diagonal<SumFactorizationKernel<2, 6, 7>>("general_C", false, 20000);
    bench_case_diagonal<SumFactorizationKernel<2, 7, 8>>("general_C", false, 8000);
    bench_case_diagonal<SumFactorizationKernel<2, 8, 9>>("general_C", false, 3000);
}

void run_p_study_3d()
{
    bench_case_diagonal<SumFactorizationKernel<3, 3, 4>>("general_C", false, 40000);
    bench_case_diagonal<SumFactorizationKernel<3, 4, 5>>("general_C", false, 10000);
    bench_case_diagonal<SumFactorizationKernel<3, 5, 6>>("general_C", false, 1500);
    bench_case_diagonal<SumFactorizationKernel<3, 6, 7>>("general_C", false, 300);
    bench_case_diagonal<SumFactorizationKernel<3, 7, 8>>("general_C", false, 80);
    bench_case_diagonal<SumFactorizationKernel<3, 8, 9>>("general_C", false, 20);
}

} // namespace

int main()
{
    std::cout << "impl,D_mode,extraction_mode,dim,p,nq,ndofs,nqp,repetitions,total_seconds,ns_per_apply,apps_per_sec\n";
    run_p_study_2d();
    run_p_study_3d();

    return 0;
}
