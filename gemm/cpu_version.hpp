template <class T>
class gemm_cpu: public gemm<T>
{
public:
    // Constructor
    gemm_cpu (uint32_t M, uint32_t N, uint32_t K) : gemm<T>(M,N,K, "CPU VERSION") {  }

    // MM Implementation
    void mm_compute (T alpha, T beta) {
        for (int i = 0; i < this->size_M; i ++) 
        {
            for (int j = 0; j < this->size_N; j ++) 
            {
                this->C[j*this->size_M + i] = beta * this->C[j*this->size_M + i];
                for (int k = 0; k < this->size_K; k++)
                {
                    this->C[j*this->size_M + i] += alpha * this->A[j*this->size_K + k] * this->B[k*this->size_M + i];
                }
            }
        }
    }
};

