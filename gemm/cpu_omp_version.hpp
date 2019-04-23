template <class T>
class gemm_cpu_omp: public gemm<T>
{
public:
    // Constructor
    gemm_cpu_omp (uint32_t M, uint32_t N, uint32_t K) : gemm<T>(M,N,K, "CPU OMP VERSION") {  }
    gemm_cpu_omp (T* or_A, T* or_B, T* or_C, uint32_t M, uint32_t N, uint32_t K) : gemm<T>(or_A, or_B, or_C, M, N, K, "CPU OMP VERSION") {  }

    // MM Implementation
    void mm_compute (T alpha, T beta) {
        #pragma omp parallel for collapse(2) num_threads(32)
        for (int i = 0; i < this->size_M; i ++) 
        {
            for (int j = 0; j < this->size_N; j ++) 
            {
                this->C[j*this->size_M + i] = beta * this->C[j*this->size_M + i];
                T tmp = 0;
                #pragma omp simd reduction(+:tmp)
                for (int k = 0; k < this->size_K; k++)
                {
                    tmp += alpha * this->A[j*this->size_K + k] * this->B[k*this->size_M + i];
                }
                this->C[j*this->size_M + i] += tmp;
            }
        }
    }
};

