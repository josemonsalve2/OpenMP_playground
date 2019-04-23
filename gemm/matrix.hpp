// Definition of the base class for matrix multiplication.
// This class is later on inherited by the particular
// implementation.
#include <omp.h>
#include <string>
#include <iostream>
#include <cstdint>
#include <cstdlib>

#define RAND ((double) (rand()) / RAND_MAX)*1000

using namespace std;

template <class T> 
class gemm 
{
protected:
    T *A;
    T *B;
    T *C;

    uint32_t size_M, size_N, size_K;
    string implementation_name;

public:
    gemm(T* origin_A, T* origin_B, T* origin_C, const uint32_t M, const uint32_t N, const uint32_t K, string imp_name) {
        // Determine the matrix sizes
        uint32_t sizeA = K * N;
        uint32_t sizeB = M * K; 
        uint32_t sizeC = M * N;

        // Initi sizes 
        this->size_M = M;
        this->size_N = N; 
        this->size_K = K;

        // Allocating the memory
        A = new T[sizeA];
        B = new T[sizeB];
        C = new T[sizeC];
        implementation_name = imp_name;

        for (int i = 0; i < this->size_M; i ++) 
        {
            for (int j = 0; j < this->size_N; j ++) 
            {
                this->C[j*this->size_M + i] = origin_C[j*this->size_M + i];
                for (int k = 0; k < this->size_K; k++)
                {
                   this->A[j*this->size_K + k] = origin_A[j*this->size_K + k];
                   this->B[k*this->size_M + i] = origin_B[k*this->size_M + i];
                }
            }
        }
    }
    gemm(const uint32_t M, const uint32_t N, const uint32_t K, string imp_name) {
        // Determine the matrix sizes
        uint32_t sizeA = K * N;
        uint32_t sizeB = M * K; 
        uint32_t sizeC = M * N;

        // Initi sizes 
        this->size_M = M;
        this->size_N = N; 
        this->size_K = K;

        // Allocating the memory
        A = new T[sizeA];
        B = new T[sizeB];
        C = new T[sizeC];
        implementation_name = imp_name;

        // Ramdomly filling the matrices
#if VERIFY == 1
        srand(0);
#endif
        for (int i = 0; i < this->size_M; i ++) 
        {
            for (int j = 0; j < this->size_N; j ++) 
            {
                this->C[j*this->size_M + i] = static_cast<T>(RAND);
                for (int k = 0; k < this->size_K; k++)
                {
                   this->A[j*this->size_K + k] = static_cast<T>(RAND);
                   this->B[k*this->size_M + i] = static_cast<T>(RAND);
                }
            }
        }
    }

    virtual void mm_compute (const T alpha, const T beta) = 0;

    void time_compute(const T alpha, const T beta) {
        double start, end;
        for (int i = 0; i < 2; i++) 
        {
            start = omp_get_wtime();
            mm_compute(alpha, beta);
            end = omp_get_wtime();
            cout << "Execution time for implementation "<< implementation_name << " is " << end - start << endl;
        }
    }

    void print_matrices () 
    {
        cout << " Matrix A " << endl << "[";
        for (int j = 0; j < this->size_N; j ++) 
        {
            cout << "[";
            for (int k = 0; k < this->size_K; k++)
            {
               cout << this->A[j*this->size_K + k];
               if (k != this->size_K - 1 )
                   cout<<",";
            }
            cout << "]";
            if (j != this->size_N - 1)
                cout <<",";
        }
        cout << "]" << endl;

        cout << " Matrix B " << endl << "[";
        for (int k = 0; k < this->size_K; k ++) 
        {
            cout << "[";
            for (int i = 0; i < this->size_M; i++)
            {
               cout << this->B[k*this->size_M + i];
               if (i != this->size_M - 1 )
                   cout<<",";
            }
            cout << "]";
            if (k != this->size_K - 1)
                cout <<",";
        }
        cout << "]" << endl;

        cout << " Matrix C " << endl << "[";
        for (int j = 0; j < this->size_N; j++)
        {
            cout << "[";
            for (int i = 0; i < this->size_M; i ++) 
            {
               cout << this->C[j*this->size_M + i];
               if (i != this->size_M - 1 )
                   cout<<",";
            }
            cout << "]";
            if (j != this->size_N - 1)
                cout <<",";
        }
        cout << "]" << endl;
        
    }

    bool verify_against(T* goodC) 
    {
        for (int i = 0; i < size_N; i ++)
            for (int j = 0; j < size_M; j++)
                if (fabs(C[i*size_M + j] - goodC[i*size_M+j]) > 0.0001)
                    return false;
        return true;
    }

    T* getA() {
        return A;
    }

    T* getB() {
        return B;
    }

    T* getC() {
        return C;
    }

    string getImpName() {
        return implementation_name;
    }

    ~gemm() {
        delete A;
        delete B;
        delete C;
    }
};
