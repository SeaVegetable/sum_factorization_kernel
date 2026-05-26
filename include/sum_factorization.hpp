#pragma once

#include <array>

constexpr int pow_int(const int base, const int exp)
{
    int value = 1;
    for (int i = 0; i < exp; ++i)
        value *= base;
    return value;
}

template<int dim, int p, int nq>
struct SumFactorizationKernel
{
    static_assert(dim == 2 || dim == 3, "Only dim=2 and dim=3 are supported.");

    static constexpr int n1d        = p + 1;
    static constexpr int nq1d       = nq;
    static constexpr int ndofs      = pow_int(n1d, dim);
    static constexpr int nqp        = pow_int(nq1d, dim);
    static constexpr int extractsize = n1d * n1d;
    static constexpr int Bsize      = n1d * nq1d;
    static constexpr int Dsize      = nqp * nqp;

    using LocalArray = std::array<double, ndofs>;
    using QPointArray = std::array<double, nqp>;
    using QPointMatrix = std::array<double, Dsize>;
    using Extract1D = std::array<double, extractsize>;
    using Basis1D = std::array<double, Bsize>;

    static constexpr int idx_2d(const int i, const int j)
    {
        return i + n1d * j;
    }

    static constexpr int idx_3d(const int i, const int j, const int k)
    {
        return i + n1d * (j + n1d * k);
    }

    static constexpr int qidx_2d(const int qx, const int qy)
    {
        return qx + nq1d * qy;
    }

    static constexpr int qidx_3d(const int qx, const int qy, const int qz)
    {
        return qx + nq1d * (qy + nq1d * qz);
    }

    static constexpr int cidx(const int i, const int a)
    {
        return i * n1d + a;
    }

    static constexpr int bidx(const int q, const int a)
    {
        return q * n1d + a;
    }

    // Centralized small dense matvec hook.
    // Today this is a plain loop-based row-major kernel.
    // Later it can be replaced by a libxsmm-dispatched micro-kernel
    // without changing the tensor-product sweep structure above it.
    template<bool transpose>
    static void apply_dense_matvec(
        const double* matrix,
        const double* x,
        double* y,
        const int rows,
        const int cols)
    {
        if constexpr (!transpose)
        {
            for (int i = 0; i < rows; ++i)
            {
                double sum = 0.0;
                for (int j = 0; j < cols; ++j)
                    sum += matrix[i * cols + j] * x[j];
                y[i] = sum;
            }
        }
        else
        {
            for (int j = 0; j < cols; ++j)
            {
                double sum = 0.0;
                for (int i = 0; i < rows; ++i)
                    sum += matrix[i * cols + j] * x[i];
                y[j] = sum;
            }
        }
    }

    static void apply_quadrature_operator(
        const QPointArray& values_q,
        const QPointMatrix& D,
        QPointArray& weighted_q)
    {
        // Applies the quadrature-space operator y = D x.
        // This is the intended replacement point for a future libxsmm kernel.
        apply_dense_matvec<false>(D.data(), values_q.data(), weighted_q.data(), nqp, nqp);
    }

    static void apply_extraction_transpose(
        const LocalArray& u_spline,
        const std::array<Extract1D, dim>& C,
        LocalArray& u_bernstein)
    {
        if constexpr (dim == 2)
        {
            LocalArray tmp{};
            std::array<double, n1d> x{};
            std::array<double, n1d> y{};

            for (int j = 0; j < n1d; ++j)
            {
                for (int i = 0; i < n1d; ++i)
                    x[i] = u_spline[idx_2d(i, j)];
                apply_dense_matvec<true>(C[0].data(), x.data(), y.data(), n1d, n1d);
                for (int ax = 0; ax < n1d; ++ax)
                    tmp[idx_2d(ax, j)] = y[ax];
            }

            for (int ax = 0; ax < n1d; ++ax)
            {
                for (int j = 0; j < n1d; ++j)
                    x[j] = tmp[idx_2d(ax, j)];
                apply_dense_matvec<true>(C[1].data(), x.data(), y.data(), n1d, n1d);
                for (int ay = 0; ay < n1d; ++ay)
                    u_bernstein[idx_2d(ax, ay)] = y[ay];
            }
        }
        else
        {
            LocalArray tmp_x{};
            LocalArray tmp_xy{};
            std::array<double, n1d> x{};
            std::array<double, n1d> y{};

            for (int k = 0; k < n1d; ++k)
                for (int j = 0; j < n1d; ++j)
                {
                    for (int i = 0; i < n1d; ++i)
                        x[i] = u_spline[idx_3d(i, j, k)];
                    apply_dense_matvec<true>(C[0].data(), x.data(), y.data(), n1d, n1d);
                    for (int ax = 0; ax < n1d; ++ax)
                        tmp_x[idx_3d(ax, j, k)] = y[ax];
                }

            for (int k = 0; k < n1d; ++k)
                for (int ax = 0; ax < n1d; ++ax)
                {
                    for (int j = 0; j < n1d; ++j)
                        x[j] = tmp_x[idx_3d(ax, j, k)];
                    apply_dense_matvec<true>(C[1].data(), x.data(), y.data(), n1d, n1d);
                    for (int ay = 0; ay < n1d; ++ay)
                        tmp_xy[idx_3d(ax, ay, k)] = y[ay];
                }

            for (int ay = 0; ay < n1d; ++ay)
                for (int ax = 0; ax < n1d; ++ax)
                {
                    for (int k = 0; k < n1d; ++k)
                        x[k] = tmp_xy[idx_3d(ax, ay, k)];
                    apply_dense_matvec<true>(C[2].data(), x.data(), y.data(), n1d, n1d);
                    for (int az = 0; az < n1d; ++az)
                        u_bernstein[idx_3d(ax, ay, az)] = y[az];
                }
        }
    }

    static void apply_extraction(
        const LocalArray& v_bernstein,
        const std::array<Extract1D, dim>& C,
        LocalArray& v_spline)
    {
        if constexpr (dim == 2)
        {
            LocalArray tmp{};
            std::array<double, n1d> x{};
            std::array<double, n1d> y{};

            for (int ay = 0; ay < n1d; ++ay)
            {
                for (int ax = 0; ax < n1d; ++ax)
                    x[ax] = v_bernstein[idx_2d(ax, ay)];
                apply_dense_matvec<false>(C[0].data(), x.data(), y.data(), n1d, n1d);
                for (int i = 0; i < n1d; ++i)
                    tmp[idx_2d(i, ay)] = y[i];
            }

            for (int i = 0; i < n1d; ++i)
            {
                for (int ay = 0; ay < n1d; ++ay)
                    x[ay] = tmp[idx_2d(i, ay)];
                apply_dense_matvec<false>(C[1].data(), x.data(), y.data(), n1d, n1d);
                for (int j = 0; j < n1d; ++j)
                    v_spline[idx_2d(i, j)] = y[j];
            }
        }
        else
        {
            LocalArray tmp_x{};
            LocalArray tmp_xy{};
            std::array<double, n1d> x{};
            std::array<double, n1d> y{};

            for (int az = 0; az < n1d; ++az)
                for (int ay = 0; ay < n1d; ++ay)
                {
                    for (int ax = 0; ax < n1d; ++ax)
                        x[ax] = v_bernstein[idx_3d(ax, ay, az)];
                    apply_dense_matvec<false>(C[0].data(), x.data(), y.data(), n1d, n1d);
                    for (int i = 0; i < n1d; ++i)
                        tmp_x[idx_3d(i, ay, az)] = y[i];
                }

            for (int az = 0; az < n1d; ++az)
                for (int i = 0; i < n1d; ++i)
                {
                    for (int ay = 0; ay < n1d; ++ay)
                        x[ay] = tmp_x[idx_3d(i, ay, az)];
                    apply_dense_matvec<false>(C[1].data(), x.data(), y.data(), n1d, n1d);
                    for (int j = 0; j < n1d; ++j)
                        tmp_xy[idx_3d(i, j, az)] = y[j];
                }

            for (int j = 0; j < n1d; ++j)
                for (int i = 0; i < n1d; ++i)
                {
                    for (int az = 0; az < n1d; ++az)
                        x[az] = tmp_xy[idx_3d(i, j, az)];
                    apply_dense_matvec<false>(C[2].data(), x.data(), y.data(), n1d, n1d);
                    for (int k = 0; k < n1d; ++k)
                        v_spline[idx_3d(i, j, k)] = y[k];
                }
        }
    }

    static void evaluate_values_bernstein(
        const LocalArray& u_bernstein,
        const std::array<Basis1D, dim>& B,
        QPointArray& values_q)
    {
        if constexpr (dim == 2)
        {
            std::array<double, nq1d * n1d> tmp_x{};
            std::array<double, n1d> x{};
            std::array<double, nq1d> y{};

            for (int ay = 0; ay < n1d; ++ay)
            {
                for (int ax = 0; ax < n1d; ++ax)
                    x[ax] = u_bernstein[idx_2d(ax, ay)];
                apply_dense_matvec<false>(B[0].data(), x.data(), y.data(), nq1d, n1d);
                for (int qx = 0; qx < nq1d; ++qx)
                    tmp_x[qx + nq1d * ay] = y[qx];
            }

            for (int qx = 0; qx < nq1d; ++qx)
            {
                for (int ay = 0; ay < n1d; ++ay)
                    x[ay] = tmp_x[qx + nq1d * ay];
                apply_dense_matvec<false>(B[1].data(), x.data(), y.data(), nq1d, n1d);
                for (int qy = 0; qy < nq1d; ++qy)
                    values_q[qidx_2d(qx, qy)] = y[qy];
            }
        }
        else
        {
            std::array<double, nq1d * n1d * n1d> tmp_x{};
            std::array<double, nq1d * nq1d * n1d> tmp_xy{};
            std::array<double, n1d> x{};
            std::array<double, nq1d> y{};

            for (int az = 0; az < n1d; ++az)
                for (int ay = 0; ay < n1d; ++ay)
                {
                    for (int ax = 0; ax < n1d; ++ax)
                        x[ax] = u_bernstein[idx_3d(ax, ay, az)];
                    apply_dense_matvec<false>(B[0].data(), x.data(), y.data(), nq1d, n1d);
                    for (int qx = 0; qx < nq1d; ++qx)
                        tmp_x[qx + nq1d * (ay + n1d * az)] = y[qx];
                }

            for (int az = 0; az < n1d; ++az)
                for (int qx = 0; qx < nq1d; ++qx)
                {
                    for (int ay = 0; ay < n1d; ++ay)
                        x[ay] = tmp_x[qx + nq1d * (ay + n1d * az)];
                    apply_dense_matvec<false>(B[1].data(), x.data(), y.data(), nq1d, n1d);
                    for (int qy = 0; qy < nq1d; ++qy)
                        tmp_xy[qx + nq1d * (qy + nq1d * az)] = y[qy];
                }

            for (int qy = 0; qy < nq1d; ++qy)
                for (int qx = 0; qx < nq1d; ++qx)
                {
                    for (int az = 0; az < n1d; ++az)
                        x[az] = tmp_xy[qx + nq1d * (qy + nq1d * az)];
                    apply_dense_matvec<false>(B[2].data(), x.data(), y.data(), nq1d, n1d);
                    for (int qz = 0; qz < nq1d; ++qz)
                        values_q[qidx_3d(qx, qy, qz)] = y[qz];
                }
        }
    }

    static void integrate_values_bernstein(
        const QPointArray& values_q,
        const std::array<Basis1D, dim>& B,
        LocalArray& v_bernstein)
    {
        if constexpr (dim == 2)
        {
            std::array<double, nq1d * n1d> tmp_y{};
            std::array<double, nq1d> x{};
            std::array<double, n1d> y{};

            for (int qx = 0; qx < nq1d; ++qx)
            {
                for (int qy = 0; qy < nq1d; ++qy)
                    x[qy] = values_q[qidx_2d(qx, qy)];
                apply_dense_matvec<true>(B[1].data(), x.data(), y.data(), nq1d, n1d);
                for (int ay = 0; ay < n1d; ++ay)
                    tmp_y[qx + nq1d * ay] = y[ay];
            }

            for (int ay = 0; ay < n1d; ++ay)
            {
                for (int qx = 0; qx < nq1d; ++qx)
                    x[qx] = tmp_y[qx + nq1d * ay];
                apply_dense_matvec<true>(B[0].data(), x.data(), y.data(), nq1d, n1d);
                for (int ax = 0; ax < n1d; ++ax)
                    v_bernstein[idx_2d(ax, ay)] = y[ax];
            }
        }
        else
        {
            std::array<double, nq1d * nq1d * n1d> tmp_z{};
            std::array<double, nq1d * n1d * n1d> tmp_y{};
            std::array<double, nq1d> x{};
            std::array<double, n1d> y{};

            for (int qy = 0; qy < nq1d; ++qy)
                for (int qx = 0; qx < nq1d; ++qx)
                {
                    for (int qz = 0; qz < nq1d; ++qz)
                        x[qz] = values_q[qidx_3d(qx, qy, qz)];
                    apply_dense_matvec<true>(B[2].data(), x.data(), y.data(), nq1d, n1d);
                    for (int az = 0; az < n1d; ++az)
                        tmp_z[qx + nq1d * (qy + nq1d * az)] = y[az];
                }

            for (int az = 0; az < n1d; ++az)
                for (int qx = 0; qx < nq1d; ++qx)
                {
                    for (int qy = 0; qy < nq1d; ++qy)
                        x[qy] = tmp_z[qx + nq1d * (qy + nq1d * az)];
                    apply_dense_matvec<true>(B[1].data(), x.data(), y.data(), nq1d, n1d);
                    for (int ay = 0; ay < n1d; ++ay)
                        tmp_y[qx + nq1d * (ay + n1d * az)] = y[ay];
                }

            for (int az = 0; az < n1d; ++az)
                for (int ay = 0; ay < n1d; ++ay)
                {
                    for (int qx = 0; qx < nq1d; ++qx)
                        x[qx] = tmp_y[qx + nq1d * (ay + n1d * az)];
                    apply_dense_matvec<true>(B[0].data(), x.data(), y.data(), nq1d, n1d);
                    for (int ax = 0; ax < n1d; ++ax)
                        v_bernstein[idx_3d(ax, ay, az)] = y[ax];
                }
        }
    }

    static void apply_mass(
        const LocalArray& u_spline,
        const std::array<Extract1D, dim>& C,
        const std::array<Basis1D, dim>& B,
        const QPointMatrix& D,
        LocalArray& v_spline)
    {
        // Computes v = C * B^T * D * B * C^T * u
        // where C is the tensor-product extraction operator and
        // B * C^T * u is evaluated by sum-factorization.
        LocalArray u_bernstein{};
        LocalArray v_bernstein{};
        QPointArray values_q{};
        QPointArray weighted_q{};

        apply_extraction_transpose(u_spline, C, u_bernstein);
        evaluate_values_bernstein(u_bernstein, B, values_q);
        apply_quadrature_operator(values_q, D, weighted_q);
        integrate_values_bernstein(weighted_q, B, v_bernstein);
        apply_extraction(v_bernstein, C, v_spline);
    }
};
