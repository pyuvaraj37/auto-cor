#include "constants.h"
#include <float.h>
#include "header.h"
#include <iostream>


double mean(double a[])
{
    double m = 0.0;
    for (int i = 0; i < DATA_SIZE; i++) {
        m += a[i];
    }
    m /= DATA_SIZE;
    return m;
}
cmplx_type cmpxdiv(cmplx_type a, cmplx_type b) {
    cmplx_type a_conj, a_conj_divisor, result;
    CMPXCONJ(a_conj, a);
    CMPXMUL(a_conj_divisor, a, a_conj);
    result.real = b.real/a_conj_divisor.real; 
    result.imag = b.imag/a_conj_divisor.real; 
    return result;
}

cmplx_type cmpxdiv_optimized(cmplx_type a, cmplx_type b) {
    cmplx_type a_conj, a_conj_divisor, result;
    CMPXDOTMUL(a_conj_divisor,a);
    result.real = b.real/a_conj_divisor.real; 
    result.imag = b.imag/a_conj_divisor.real; 
    return result;
}
void dot_multiply(cmplx_type a[DATA_SIZE], cmplx_type b[DATA_SIZE])
{
    std::cout << "I am in dot-multiply";
    for (int i = 0; i < DATA_SIZE; i++) {
        cmplx_type a_conj;  //
        CMPXCONJ(a_conj, a[i]);
        CMPXMUL(b[i], a[i], a_conj);
    }
    
}
void dot_multiply_optimized(cmplx_type a[DATA_SIZE], cmplx_type b[DATA_SIZE])
{
    std::cout << "I am in dot-multiply optimized";
    for (int i = 0; i < DATA_SIZE; i++) {
        #pragma HLS pipeline
       // #pragma HLS unroll pipeline //skip_exit_check off=true
        CMPXDOTMUL(b[i],a[i]);
                           
        
    }  

}



void co_autocorrs_optimized(double y[DATA_SIZE], double z[DATA_SIZE])
{   
    std::cout << "AUTO COR" << std::endl; 
    double m, nFFT;
    m = mean(y);
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
    pease_fft(input, output);

    cmplx_type divisor = output[0];
    
    for (int i = 0; i < 2 * DATA_SIZE; i++) {
        #pragma HLS unroll
        input[i] = cmpxdiv_optimized(divisor, output[i]); // F[i] / divisor;
    }


    for (int i = 0; i < 2 * DATA_SIZE; i++) {  
        z[i] = input[i].real;
    }

}

void co_autocorrs(double y[DATA_SIZE], double z[DATA_SIZE])
{   
    std::cout << "AUTO COR" << std::endl; 
    double m, nFFT;
    m = mean(y);
    std::cout << "I am in auto corrs";
    
    
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
    dot_multiply(output, input);
    pease_fft(input, output);

    cmplx_type divisor = output[0];
    
    for (int i = 0; i < 2 * DATA_SIZE; i++) {
        input[i] = cmpxdiv(divisor, output[i]); // F[i] / divisor;
    }


    for (int i = 0; i < 2 * DATA_SIZE; i++) {
  
        z[i] = input[i].real;
    }

}


int CO_FirstMin_ac(double window[DATA_SIZE],bool optimized)
{
    // Removed NaN check
    double autocorrs[2 * DATA_SIZE];
    if(optimized) co_autocorrs_optimized(window, autocorrs);
    else co_autocorrs(window, autocorrs);
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

extern "C" void krnl(data_t* input, data_t* output,bool optimized) {
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
    result = CO_FirstMin_ac(window,optimized);
    //std::cout <<  << "\n"; result
    /* Writing to DDR */
    output[0] = result;
}
