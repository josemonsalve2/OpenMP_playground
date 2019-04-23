#include <iostream>
#include <vector>
#include <cmath>

#ifndef VERIFY
#define VERIFY 0
#endif

#include "matrix.hpp"

// INCLUDING THE DIFFERENT VERSIONS
#include "cpu_version.hpp"
#include "cpu_omp_version.hpp"
#include "gpu_omp_version.hpp"
#include "gpu_omp_2_version.hpp"



int main(int argc, char **argv) {
    uint32_t M,N,K;
    double alpha;
    double beta;
    
    // ARGUMENTS READING AND PARSING
    if (argc == 1) {
        cout << "=== GEMM Implementations tester ===" << endl << "=> Usage " << argv[0] << " [M] [N] [K] [alpha] [beta]" << endl;
    }
    if (argc < 4) {
        M = 10;
        N = 10;
        K = 10;
        alpha = 1.0;
        beta = 1.0;

        cout << "=> Using default parameters: M = " << M << " : N = " << N << " : K = " << K << " : alpla = " << alpha<< " : beta = " << beta << endl;

    } else if (argc == 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        cout << "=> Using default parameters: alpla = 1.0 " << " : beta = 1.0 " << endl;
        alpha = 1.0;
        beta = 1.0;
    } else if (argc == 6) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
        alpha = atoi(argv[4]);
        beta = atoi(argv[5]);
    }

#if VERIFY == 1
    // ASSUMES CPU IMPLEMENTATION IS CORRECT
    double * verificationC;
    verificationC = new double[M*N];
    gemm_cpu<double> verification(M,N,K);
    verification.mm_compute(alpha, beta);
    for (int j = 0; j < N; j ++)
        for (int i = 0; i < M; i++)
            verificationC[j*M + i] = verification.getC()[j*M + i];
#endif 

    // CONTAINER FOR ALL THE DIFF IMPLEMENTATIONS
    vector< gemm<double> * > myTesters;

    // POPULATING THE CONTAINER
    cout << "Initializing data" << endl;
    gemm_cpu<double> CPU(M,N,K);
    gemm_cpu_omp<double> CPU_OMP(CPU.getA(), CPU.getB(), CPU.getC(), M, N, K);
    gemm_gpu_omp<double> GPU_OMP(CPU.getA(), CPU.getB(), CPU.getC(), M, N, K);
    gemm_gpu_2_omp<double> GPU_OMP2(CPU.getA(), CPU.getB(), CPU.getC(), M, N, K);
    myTesters.push_back(&CPU);
    myTesters.push_back(&CPU_OMP);
    myTesters.push_back(&GPU_OMP);
    myTesters.push_back(&GPU_OMP2);


    // Calling all the computes 
    cout << "starting tests" << endl;
    for(auto it = myTesters.begin(); it != myTesters.end(); it++)    {
        gemm<double> * thisImpl = *it;
        thisImpl->time_compute(alpha, beta);
#if VERIFY == 1
        if (thisImpl->verify_against(verificationC)) {
            cout << "CORRECT IMPLEMENTATION " << thisImpl->getImpName() << endl; 
        }
        else {
            cout << "ERROR IN IMPLEMENTATION ..." << thisImpl->getImpName() << endl;
        }
#endif
    }

    return 0;
}
