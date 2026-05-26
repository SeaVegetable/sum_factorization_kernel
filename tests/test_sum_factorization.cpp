#include "sum_factorization.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>

namespace
{

template<typename Real>
Real max_abs_diff(const Real* a, const Real* b, const int n)
{
    Real err = Real(0);
    for (int i = 0; i < n; ++i)
        err = std::max(err, std::abs(a[i] - b[i]));
    return err;
}

template<typename Array, typename RNG>
void fill_random(Array& values, RNG& rng, const double lo = -1.0, const double hi = 1.0)
{
    std::uniform_real_distribution<double> dist(lo, hi);
    for (auto& v : values)
        v = dist(rng);
}

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

template<typename Kernel>
typename Kernel::LocalArray naive_mass_reference(
    const typename Kernel::LocalArray& u,
    const std::array<typename Kernel::Extract1D, kernel_dim<Kernel>()>& C,
    const std::array<typename Kernel::Basis1D, kernel_dim<Kernel>()>& B,
    const typename Kernel::QPointMatrix& D)
{
    const auto E = build_dense_extraction<Kernel>(C);
    const auto M = build_dense_bernstein_eval<Kernel>(B);

    const auto u_bernstein = dense_transpose_matvec<Kernel::ndofs, Kernel::ndofs>(E, u);
    const auto values_q = dense_matvec<Kernel::nqp, Kernel::ndofs>(M, u_bernstein);
    const auto weighted_q = dense_matvec<Kernel::nqp, Kernel::nqp>(D, values_q);
    const auto v_bernstein = dense_transpose_matvec<Kernel::nqp, Kernel::ndofs>(M, weighted_q);
    return dense_matvec<Kernel::ndofs, Kernel::ndofs>(E, v_bernstein);
}

template<typename Kernel>
bool run_case(const std::string& label)
{
    std::mt19937 rng(static_cast<std::uint32_t>(Kernel::n1d * 1000 + Kernel::nq1d * 10 + kernel_dim<Kernel>()));

    typename Kernel::LocalArray u{};
    typename Kernel::LocalArray v{};
    fill_random(u, rng);

    const auto C = make_extraction<Kernel>(rng);
    const auto B = make_basis<Kernel>(rng);
    const auto D = make_quadrature_matrix<Kernel>(rng);

    Kernel::apply_mass(u, C, B, D, v);
    const auto v_ref = naive_mass_reference<Kernel>(u, C, B, D);
    const double err_mass = max_abs_diff(v.data(), v_ref.data(), Kernel::ndofs);

    const auto C_id = make_identity_extraction<Kernel>();
    typename Kernel::LocalArray v_id{};
    Kernel::apply_mass(u, C_id, B, D, v_id);
    const auto v_id_ref = naive_mass_reference<Kernel>(u, C_id, B, D);
    const double err_identity = max_abs_diff(v_id.data(), v_id_ref.data(), Kernel::ndofs);

    typename Kernel::LocalArray zero{};
    typename Kernel::LocalArray zero_out{};
    Kernel::apply_mass(zero, C, B, D, zero_out);
    double err_zero = 0.0;
    for (const double x : zero_out)
        err_zero = std::max(err_zero, std::abs(x));

    std::cout << std::fixed << std::setprecision(3);
    std::cout << label
              << " | mass err = " << err_mass
              << " | identity err = " << err_identity
              << " | zero err = " << err_zero
              << '\n';

    return err_mass < 1e-12 &&
           err_identity < 1e-12 &&
           err_zero < 1e-15;
}

} // namespace

int main()
{
    using K2 = SumFactorizationKernel<2, 2, 3>;
    using K3 = SumFactorizationKernel<3, 2, 3>;
    using K4 = SumFactorizationKernel<3, 3, 4>;

    bool ok = true;
    ok = run_case<K2>("dim=2 p=2 nq=3") && ok;
    ok = run_case<K3>("dim=3 p=2 nq=3") && ok;
    ok = run_case<K4>("dim=3 p=3 nq=4") && ok;

    if (!ok)
    {
        std::cerr << "sum_factorization.hpp tests failed.\n";
        return 1;
    }

    std::cout << "All sum_factorization.hpp tests passed.\n";
    return 0;
}
