#include "constants.h"
#include <float.h>
#include "header.h"
#include <iostream>
double mean_opti(double a[])
{
    double m = 0.0;
    for (int i = 0; i < DATA_SIZE; i++) {
        m += a[i];
    }
    m /= DATA_SIZE;
    return m;
}
void dot_multiply_optimized(cmplx_type a[DATA_SIZE], cmplx_type b[DATA_SIZE])
{
    #pragma HLS array_partition variable=a complete
    #pragma HLS array_partition variable=b complete
    std::cout << "I am in dot-multiply optimized";
    loop_check :for (int i = 0; i < 2*DATA_SIZE; i++) {
        #pragma HLS unroll 
       // #pragma HLS unroll pipeline //skip_exit_check off=true
        CMPXDOTMUL(b[i],a[i]);                          
        
    }
}
cmplx_type cmpxdiv_opti(cmplx_type a, cmplx_type b) {
    cmplx_type a_conj, a_conj_divisor, result;
    CMPXCONJ(a_conj, a);
    CMPXMUL(a_conj_divisor, a, a_conj);
    result.real = b.real/a_conj_divisor.real; 
    result.imag = b.imag/a_conj_divisor.real; 
    return result;
}
void inverse_fft_opti(cmplx_type input[DATA_SIZE], cmplx_type output[DATA_SIZE]) 
{
    #pragma HLS array_partition variable=input complete
    #pragma HLS array_partition variable=output complete
    cmplx_type conju[2*DATA_SIZE];
    cmplx_type second_conj[2*DATA_SIZE];
    #pragma HLS array_partition variable=conju complete
    #pragma HLS array_partition variable=second_conj complete
    loop_inverse_conj:for(int i=0;i<2*DATA_SIZE;i++) 
    {
        #pragma HLS unroll 
        CMPXCONJ(conju[i],input[i]);
    }
    pease_fft(conju,second_conj);
    
    loop_inverse_conj2:for(int i=0;i<2*DATA_SIZE;i++) 
    {
        #pragma HLS unroll 
        CMPXCONJ(output[i],second_conj[i]);
        output[i].real = output[i].real/(2*DATA_SIZE);
        output[i].imag = output[i].imag/(2*DATA_SIZE);

    }

} 
void co_autocorrs_optimized(double y[DATA_SIZE], double z[DATA_SIZE])
{  
    std::cout << "AUTO COR" << std::endl; 
    double m, nFFT;
    m = mean_opti(y);
    //NFFT is not defined
    std::cout << "I am in auto corrs optimized";
    cmplx_type input[2 * DATA_SIZE];
    cmplx_type output[2 * DATA_SIZE];
    
   
    for (int i = 0; i < DATA_SIZE; i++) {
        
        input[i].real = y[i] - m;
        input[i].imag = 0.0; 
    }
    for (int i = DATA_SIZE; i < 2 * DATA_SIZE; i++) {
        input[i].real = 0.0;
        input[i].imag = 0.0; 
    }

    pease_fft(input, output);
    dot_multiply_optimized(output, input);
    inverse_fft_opti(input, output);
    cmplx_type divisor = output[0];    
    for (int i = 0; i < 2 * DATA_SIZE; i++) {        
        input[i] = cmpxdiv_opti(divisor, output[i]); // F[i] / divisor;
    }
   
 
    for (int i = 0; i < 2 * DATA_SIZE; i++) {       
        z[i] = input[i].real;
    }
}

int CO_FirstMin_ac_opti(double window[DATA_SIZE])
{
    // Removed NaN check
    double autocorrs[2 * DATA_SIZE];
    co_autocorrs_optimized(window, autocorrs);
    
    int minInd = DATA_SIZE;
    for(int i = 1; i < DATA_SIZE-1; i++)
    {
        if(autocorrs[i] < autocorrs[i-1] && autocorrs[i] < autocorrs[i+1])
        {
            minInd = i;
            break;
        }
    }    
    return minInd;
    
}
extern "C" void krnl_optimized(data_t* input, data_t* output) {
    //int arr_len = sizeof(input)/sizeof(input[0]);
    //std::cout <<"The length of the array is"<<arr_len;
   
    #pragma HLS INTERFACE mode=m_axi port=input
    #pragma HLS INTERFACE mode=m_axi port=output

    static data_t window[DATA_SIZE];
    data_t result;
    static int w = 0; 

    /* Reading from DDR*/
    for (int i = 0; i < DATA_SIZE; i++) {
        window[i] = input[i];
    }

   
    
    /* Feature Extraction */
    result = CO_FirstMin_ac_opti(window);
    //std::cout <<  << "\n"; result
    /* Writing to DDR */
    output[0] = result;
}