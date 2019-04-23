template <class T>
class gemm_gpu_omp: public gemm<T>
{
public:
    // Constructor
    gemm_gpu_omp (uint32_t M, uint32_t N, uint32_t K) : gemm<T>(M,N,K, "GPU OMP VERSION") {  }
    gemm_gpu_omp (T* or_A, T* or_B, T* or_C, uint32_t M, uint32_t N, uint32_t K) : gemm<T>(or_A, or_B, or_C, M, N, K, "GPU OMP VERSION") {  }

    // MM Implementation
    void mm_compute (T alpha, T beta) {
        T* ptrA = this->A;
        T* ptrB = this->B;
        T* ptrC = this->C;
        uint32_t M = this->size_M;
        uint32_t N = this->size_N;
        uint32_t K = this->size_K;
        T tmp = 0;

        #pragma omp target teams distribute parallel for map(to: ptrA[0:N*K], ptrB[0:M*K]) map(tofrom: ptrC[0:M*N]) firstprivate(tmp) collapse(2)
        for (int i = 0; i < M; i ++)
        {
            for (int j = 0; j < N; j ++) 
            {
                ptrC[j*M + i] = beta * ptrC[j*M + i];
                #pragma omp simd reduction(+:tmp) simdlen(64) 
                for (int k = 0; k < K; k++)
                {
                    tmp += alpha * ptrA[j*K + k] * ptrB[k*M + i];
                }
                ptrC[j*M + i] += tmp;
            }
        }
    }
};

