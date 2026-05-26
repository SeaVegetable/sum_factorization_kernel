#include "sum_factorization.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>

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

template<typename Kernel>
using DenseDofMatrix =
    std::array<double, Kernel::ndofs * Kernel::ndofs>;

template<typename Kernel>
using DenseQMatrix =
    std::array<double, Kernel::nqp * Kernel::ndofs>;

template<typename Kernel>
DenseDofMatrix<Kernel> build_dense_extraction(
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C)
{
    DenseDofMatrix<Kernel> E{};

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
    DenseQMatrix<Kernel> M{};

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

template<int nrows, int ncols>
std::array<double, nrows> dense_matvec(
    const std::array<double, nrows * ncols>& A,
    const std::array<double, ncols>& x)
{
    std::array<double, nrows> y{};
    for (int i = 0; i < nrows; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < ncols; ++j)
            sum += A[i * ncols + j] * x[j];
        y[i] = sum;
    }
    return y;
}

template<int nrows, int ncols>
std::array<double, ncols> dense_transpose_matvec(
    const std::array<double, nrows * ncols>& A,
    const std::array<double, nrows>& x)
{
    std::array<double, ncols> y{};
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
    double checksum;
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

    double checksum = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        Kernel::apply_mass(u, C, B, D, v);
        checksum += v[rep % Kernel::ndofs];
        u[rep % Kernel::ndofs] += 1e-14 * checksum;
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count(), checksum};
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
        const auto u_bernstein = dense_transpose_matvec<Kernel::ndofs, Kernel::ndofs>(E, u);
        const auto values_q = dense_matvec<Kernel::nqp, Kernel::ndofs>(M, u_bernstein);
        const auto weighted_q = dense_matvec<Kernel::nqp, Kernel::nqp>(D, values_q);
        const auto v_bernstein = dense_transpose_matvec<Kernel::nqp, Kernel::ndofs>(M, weighted_q);
        const auto v = dense_matvec<Kernel::ndofs, Kernel::ndofs>(E, v_bernstein);
        (void)v;
    }

    double checksum = 0.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < repetitions; ++rep)
    {
        const auto u_bernstein = dense_transpose_matvec<Kernel::ndofs, Kernel::ndofs>(E, u);
        const auto values_q = dense_matvec<Kernel::nqp, Kernel::ndofs>(M, u_bernstein);
        const auto weighted_q = dense_matvec<Kernel::nqp, Kernel::nqp>(D, values_q);
        const auto v_bernstein = dense_transpose_matvec<Kernel::nqp, Kernel::ndofs>(M, weighted_q);
        const auto v = dense_matvec<Kernel::ndofs, Kernel::ndofs>(E, v_bernstein);
        checksum += v[rep % Kernel::ndofs];
        u[rep % Kernel::ndofs] += 1e-14 * checksum;
    }
    const auto t1 = std::chrono::steady_clock::now();

    const std::chrono::duration<double> dt = t1 - t0;
    return {dt.count(), checksum};
}

template<typename Kernel>
void print_result(
    const char* impl,
    const char* extraction_mode,
    const int repetitions,
    const BenchResult& result)
{
    const double ns_per_apply = 1.0e9 * result.seconds / static_cast<double>(repetitions);
    const double apps_per_sec = static_cast<double>(repetitions) / result.seconds;

    std::cout << impl
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
              << ',' << std::fixed << std::setprecision(6) << result.checksum
              << '\n';
}

template<typename Kernel>
void bench_case(const char* extraction_mode, const bool identity_extraction, const int repetitions)
{
    std::mt19937 rng(static_cast<std::uint32_t>(
        kernel_dim<Kernel>() * 10000 + Kernel::n1d * 100 + Kernel::nq1d * 10 +
        (identity_extraction ? 1 : 7)));

    typename Kernel::LocalArray u{};
    fill_random(u, rng);
    const auto C = identity_extraction ? make_identity_extraction<Kernel>() : make_extraction<Kernel>(rng);
    const auto B = make_basis<Kernel>(rng);
    const auto D = make_quadrature_matrix<Kernel>(rng);

    constexpr int warmup = 100;
    const BenchResult sf_result =
        run_sum_factorized_benchmark<Kernel>(u, C, B, D, warmup, repetitions);
    const BenchResult dense_result =
        run_dense_baseline_benchmark<Kernel>(u, C, B, D, warmup, repetitions);

    print_result<Kernel>("sum_factorized", extraction_mode, repetitions, sf_result);
    print_result<Kernel>("naive_dense", extraction_mode, repetitions, dense_result);
}

} // namespace

int main()
{
    std::cout << "impl,extraction_mode,dim,p,nq,ndofs,nqp,repetitions,total_seconds,ns_per_apply,apps_per_sec,checksum\n";

    bench_case<SumFactorizationKernel<3, 2, 3>>("general_C", false, 400000);
    bench_case<SumFactorizationKernel<3, 2, 3>>("identity_C", true, 400000);

    bench_case<SumFactorizationKernel<3, 3, 4>>("general_C", false, 150000);
    bench_case<SumFactorizationKernel<3, 3, 4>>("identity_C", true, 150000);

    bench_case<SumFactorizationKernel<3, 4, 5>>("general_C", false, 40000);
    bench_case<SumFactorizationKernel<3, 4, 5>>("identity_C", true, 40000);

    bench_case<SumFactorizationKernel<2, 2, 3>>("general_C", false, 600000);
    bench_case<SumFactorizationKernel<2, 2, 3>>("identity_C", true, 600000);

    return 0;
}
