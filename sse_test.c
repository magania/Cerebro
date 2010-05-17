#include <xmmintrin.h>
#include <malloc.h>
#include <stdlib.h>
#include <iostream>

/* x = k*a; */
void mul(int size, float* x, float* a, float k){
	int nLoop = size/4;

	__m128 K = _mm_set_ps1(k);

    __m128* A = (__m128*) a;
    __m128* X = (__m128*) x;

    for(int i=0; i<nLoop; i++)
	 *X++ = _mm_mul_ps(*A++,K);
}

/* a = a+b */
void add(int size, float* a, float* b){
	int nLoop = size/4;

    __m128* A = (__m128*) a;
    __m128* B = (__m128*) b;

	for(int i=0; i<nLoop; i++)
		*A++ =  _mm_add_ps(*A, *B++);
}

/* x = k*(a-b) */
void del(int size, float* x, float* a, float* b, float k){
	int nLoop = size/4;

	__m128 K = _mm_set_ps1(k);
	__m128 AB;

    __m128* A = (__m128*) a;
    __m128* B = (__m128*) b;
    __m128* X = (__m128*) x;

	for(int i=0; i<nLoop; i++){
		 AB = _mm_sub_ps(*A++, *B++);
		 *X++ = _mm_mul_ps(AB, K);
	}
}

/* x = 0 */
void zero(int size, float *x){
	int nLoop = size/4;

	__m128 zero = _mm_set_ps1(0);
    __m128* X = (__m128*) x;

	for(int i=0; i<nLoop; i++)
		 *X++ = _mm_mul_ps(*X, zero);
}

void print(int size, float* array){
  for (int i=0; i<size; i++)
    std::cout << array[i] << ' ';
  std::cout << std::endl << std::endl;
}

int main(void){
    float **numbers = (float**) malloc( 2 * sizeof(float*));
    posix_memalign((void **) &numbers[0],16, 100 * sizeof(float));
    posix_memalign((void **) &numbers[1],16, 100 * sizeof(float));

    for (int i=0; i<100; i++)
      numbers[0][i] = i;
    for (int i=0; i<100; i++)
      numbers[1][i] = 2*i;

    print(100, numbers[0]);
    print(100, numbers[1]);
    del(100, numbers[0], numbers[1], numbers[0], 5);
    print(100, numbers[0]);
    zero(100, numbers[0]);
    print(100, numbers[0]);

    free(numbers[0]);
    free(numbers[1]);

}
