// Parallel distributed iterative solution of the matrix equation
// Ax = b
// where A is a sparse 3-(tridiagonal block) matrix
// using steepest descent
// uses openMPI
// by Krasimir Georgiev, 2013
#include <mpi.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cmath>
using namespace std;

// size of the block, number of processors, and the current processor
int n, p, k;

// tridiagonal matrix represenation, supporting vector multiplication
struct tridiag_matrix {
    double c[3];

    void mult_add(double* source, double* dest) {
        for (int i = 1; i <= n; ++i) {
            dest[i] += c[0] * source[i - 1]
                    +  c[1] * source[i    ]
                    +  c[2] * source[i + 1];
        }
    }
};

// distributed vector implementation - the blocks are distributed across
// different processors based on the block id remainder modulo number of
// processes
struct dvec {
    double* data;

    dvec() {
        alloc_data();
    }

    dvec(dvec& b) {
        data = b.data;
    }

    void alloc_data() {
        data = new double[(n/p + 2) * (n + 1) * sizeof(double)];
    }
    
    // clears the vector to zero
    void clear() {
        memset(data, 0, (n/p + 2) * (n + 1) * sizeof(double));
    }

    // returns the start of a block with a given index
    double* block(int i) {
        assert (0 <= i && i <= n/p + 1);
        return data + i * (n + 1);
    }

    // retrieves the data from previous (d = -1) or next (d = 1) process,
    // required for triblock matrix multiplication
    void get(int d, dvec& dest) {
        if (p != 1) {
            // parity local communication between adjacent processes
            if (k % 2 == 0) {
                send(d);
                recv(d, dest);
            } else {
                recv(d, dest);
                send(d);
            }
        } else {
            // one processor - avoid MPI Send / Recv
            memcpy(dest.block(1), block(1 + d), n * (n + 1) * sizeof(double));
        }
    }

    void send(int d) {
        MPI_Send(block((2*p - k - 1 + d) / p), n/p * (n + 1), MPI_DOUBLE, (k - d + p) % p, 0, MPI_COMM_WORLD);
    }

    void recv(int d, dvec& dest) {
        MPI_Recv(dest.block(1), n/p * (n + 1), MPI_DOUBLE, (k + d + p) % p, 0, MPI_COMM_WORLD, 0);
    }

    // dot product; result in all processes
    double operator *(dvec& b) {
        double p_result = 0.0, result = 0.0;

        for (int i = 0; i < (n/p + 2) * (n + 1); ++i) {
            p_result += data[i] * b.data[i];
        }

        MPI_Allreduce(&p_result, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        return result;
    }

    // subtract a scalar from all the components of the distributed vector
    void operator -=(double b) {
        for (int i = 1; i <= n/p; ++i) {
            for (int j = 1; j <= n; ++j) {
                block(i)[j] -= b;
            }
        }
    }

    // vector subtraction
    void operator -=(dvec& b) {
        for (int i = 1; i <= n/p; ++i) {
            for (int j = 1; j <= n; ++j) {
                block(i)[j] -= b.block(i)[j];
            }
        }
    }

    // scalar multiplication
    void operator *= (double b) {
        for (int i = 0; i < (n/p + 2) * (n + 1); ++i) {
            data[i] *= b;
        }
    }

    // abs-max vector norm (C - norm)
    double c_norm() {
        double p_result = 0.0, result = 0.0;
        for (int i = 0; i < (n/p + 2) * (n + 1); ++i) {
            p_result = max(p_result, fabs(data[i]));
        }

        MPI_Allreduce(&p_result, &result, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        return result;
    }

    // print a representation of the distributed vector
    void print() {
        printf("[%d]: ", k);
        for (int i = 1; i <= n/p; ++i) {
            for (int j = 1; j <= n; ++j) {
                printf("%lf ", block(i)[j]);
            }
            printf("| ");
        }
        printf("\n");
    }
};

// triblock matrix representation, supporting distributed vector multiplication
struct triblock_matrix {
    tridiag_matrix a, b, c;

    // distributed vector multiplication, v_prev and v_next are used to get the
    // pieces needed for multiplication located at different processes
    void mult(dvec& v_prev, dvec& v_curr, dvec& v_next, dvec& to) {
        v_curr.get(-1, v_prev);
        v_curr.get(+1, v_next);
        
        to.clear();
        
        for (int i = 1; i <= n/p; ++i) {
            a.mult_add(v_prev.block(i), to.block(i));
            b.mult_add(v_curr.block(i), to.block(i));
            c.mult_add(v_next.block(i), to.block(i));
        }
    }
};

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &k);
    
    double start_time = MPI_Wtime();

    // read the size of the block
    n = atoi(argv[1]);
    if (k == 0) {
        printf("n: %d\n", n);
    }
    double* a = new double[n*n];
    for (int i = 0; i < n*n; ++i) {
        a[i] = i;
    }
    
    triblock_matrix m = {{-2.0, -1.0, -2.0}, {-1.0, 12.0, -1.0}, {-2.0, -1.0, -2.0}};
    
    // implementation of the steepest descent iteration
    double b = 1.0;
    dvec x_curr, x_prev, x_next;
    x_curr.clear();
    dvec r_curr, r_prev, r_next, mr;
    double error = 1.0;
    int iters = 0;
    while (error > 1e-3) {
        // r = m x - b
        //if (k == 0 && iters % 100 == 0) {
        //    printf("iters: %3d, error: %lf\n", iters, error);
        //}
        ++iters;

        m.mult(x_prev, x_curr, x_next, r_curr);
        r_curr -= b;

        double al_num = r_curr * r_curr;
        m.mult(r_prev, r_curr, r_next, mr);
        double al_den = r_curr * mr;
        double al = al_num / al_den;
        r_curr *= al;
        x_curr -= r_curr;
        error = r_curr.c_norm();
    }

    double end_time = MPI_Wtime();

    if (k == 0) printf("final iters: %3d, error: %lf\ntime: %lf\n", iters, error, end_time - start_time);

    MPI_Finalize();
    return 0;
}
