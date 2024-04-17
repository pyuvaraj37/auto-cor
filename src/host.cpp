#include "host.h"
#include <stdio.h>
#include "constants.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>
#include <tgmath.h>
#include <complex>

// #if __cplusplus

#include <complex>
typedef std::complex< double > cplx;

// #else

// #include <complex.h>

// #if defined(__GNUC__) || defined(__GNUG__)
// typedef double complex cplx;
// #elif defined(_MSC_VER)
// typedef _Dcomplex cplx;
// #endif
// #endif

#define pow2(x) (1 << x)

double mean(const double a[], const int size)
{
    double m = 0.0;
    for (int i = 0; i < size; i++) {
        m += a[i];
    }
    m /= size;
    return m;
}

cplx _Cmulcc(cplx x, cplx y) {
    /*double a = x._Val[0];
    double b = x._Val[1];

    double c = y._Val[0];
    double d = y._Val[1];

    cplx result = { (a * c - b * d), (a * d + c * b) };
        */
    return x*y;
}

cplx _Cminuscc(cplx x, cplx y) {
    //cplx result = { x._Val[0] - y._Val[0], x._Val[1] - y._Val[1] };
    return x - y;
}

cplx _Caddcc(cplx x, cplx y) {
    // cplx result = { x._Val[0] + y._Val[0], x._Val[1] + y._Val[1] };
    return x + y;
}

cplx _Cdivcc(cplx x, cplx y) {
    
    double a = real(x);
    double b = imag(x);

    double c = real(y);
    double d = imag(y);

    cplx result;
    result.real((a*c + b*d) / (c*c + d*d));
    result.imag((b*c - a*d)/(c*c + d*d));
    
    return result;
        
    // return x / y;
}

void twiddles(cplx a[], int size)
{

    double PI = 3.14159265359;

    for (int i = 0; i < size; i++) {
        // cplx tmp = { 0, -PI * i / size };
	    cplx tmp = {0.0, -PI * i / size };
        a[i] = exp(tmp);
        //a[i] = cexp(-I * M_PI * i / size);
    }
}

static void _fft(cplx a[], cplx out[], int size, int step, cplx tw[])
{   
    if (step < size) {
        _fft(out, a, size, step * 2, tw);
        _fft(out + step, a + step, size, step * 2, tw);

        for (int i = 0; i < size; i += 2 * step) {
            //cplx t = tw[i] * out[i + step];
            cplx t = _Cmulcc(tw[i], out[i + step]);
            a[i / 2] = _Caddcc(out[i], t);
            a[(i + size) / 2] = _Cminuscc(out[i], t);
        }
    }
}

void fft(cplx a[], int size, cplx tw[])
{
    cplx * out = (cplx*) malloc(size * sizeof(cplx));
    memcpy(out, a, size * sizeof(cplx));
    _fft(a, out, size, 1, tw);
    free(out);
}

int nextpow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

void dot_multiply(cplx a[], cplx b[], int size)
{
    for (int i = 0; i < size; i++) {
        a[i] = _Cmulcc(a[i], conj(b[i]));
    }
}


double * co_autocorrs(const double y[], const int size)
{
    double m, nFFT;
    m = mean(y, size);
    nFFT = nextpow2(size) << 1;
    
    cplx * F = (cplx*) malloc(nFFT * 2 * sizeof *F);
    cplx * tw = (cplx*) malloc(nFFT * 2 * sizeof *tw);
    for (int i = 0; i < size; i++) {
        
        cplx tmp = { y[i] - m, 0.0 };
        F[i] = tmp;
        
    }
    for (int i = size; i < nFFT; i++) {
        
        cplx tmp = { 0.0, 0.0 };
        F[i] = tmp;

    }
    //size = nFFT;
    
    twiddles(tw, nFFT);
    fft(F, nFFT, tw);
    dot_multiply(F, F, nFFT);
    fft(F, nFFT, tw);
    cplx divisor = F[0];
    for (int i = 0; i < nFFT; i++) {
        F[i] = _Cdivcc(F[i], divisor); // F[i] / divisor;
    }
    
    double * out = (double*) malloc(nFFT * 2 * sizeof(out));
    for (int i = 0; i < nFFT; i++) {
        out[i] = real(F[i]);
    }
    free(F);
    free(tw);
    return out;
}


int CO_FirstMin_ac(const double y[], const int size)
{
    
    // NaN check
    // for(int i = 0; i < size; i++)
    // {
    //     if(isnan(y[i]))
    //     {
    //         return 0;
    //     }
    // }
    
    double * autocorrs = co_autocorrs(y, size);
    
    int minInd = size;
    for(int i = 1; i < size-1; i++)
    {
        if(autocorrs[i] < autocorrs[i-1] && autocorrs[i] < autocorrs[i+1])
        {
            minInd = i;
            break;
        }
    }
    
    free(autocorrs);
    
    return minInd;
    
}


int main(int argc, char** argv) {


    if (argc != 2) {
        std::cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
        return EXIT_FAILURE;
    }
   
    clock_t htod, dtoh, comp, comp_opti,dtoh_opti,htod_opti;


    /*====================================================CL===============================================================*/
    std::string binaryFile = argv[1];
    cl_int err;
    cl::Context context;
    cl::Kernel krnl1, krnl2, krnl3;
    cl::CommandQueue q;
   
    auto devices = get_xil_devices();
    auto fileBuf = read_binary_file(binaryFile);
    cl::Program::Binaries bins{{fileBuf.data(), fileBuf.size()}};
    bool valid_device = false;
    for (unsigned int i = 0; i < devices.size(); i++) {
        auto device = devices[i];
        OCL_CHECK(err, context = cl::Context(device, nullptr, nullptr, nullptr, &err));
        OCL_CHECK(err, q = cl::CommandQueue(context, device, 0, &err));
        std::cout << "Trying to program device[" << i << "]: " << device.getInfo<CL_DEVICE_NAME>() << std::endl;
        cl::Program program(context, {device}, bins, nullptr, &err);
        if (err != CL_SUCCESS) {
            std::cout << "Failed to program device[" << i << "] with xclbin file!\n";
        } else {
            std::cout << "Device[" << i << "]: program successful!\n";
            std::cout << "Setting CU(s) up..." << std::endl;
            OCL_CHECK(err, krnl1 = cl::Kernel(program, "krnl", &err));
            OCL_CHECK(err, krnl2 = cl::Kernel(program, "krnl", &err));
            valid_device = true;
            break; // we break because we found a valid device
        }
    }
    if (!valid_device) {
        std::cout << "Failed to program any device found, exit!\n";
        exit(EXIT_FAILURE);
    }


    /*====================================================INIT INPUT/OUTPUT VECTORS===============================================================*/
    std::vector<data_t, aligned_allocator<data_t> > input(DATA_SIZE);
    std::vector<data_t, aligned_allocator<data_t> > input_opti(DATA_SIZE);
    data_t *input_sw = (data_t*) malloc(sizeof(data_t) * (DATA_SIZE));
    std::vector<data_t, aligned_allocator<data_t> > output_hw(1);
    std::vector<data_t, aligned_allocator<data_t> > output_hw_opti(1);
    data_t *output_sw = (data_t*) malloc(sizeof(data_t) * (1));


    for (int i = 0; i < DATA_SIZE; i++) {
        input[i] = data[i];
        input_opti[i] = data[i];
        input_sw[i] = data[i];
    }

    /*====================================================SW VERIFICATION===============================================================*/


    output_sw[0] = CO_FirstMin_ac(input_sw, DATA_SIZE);



    /*====================================================Setting up kernel I/O===============================================================*/

    /* INPUT BUFFERS */
    OCL_CHECK(err, cl::Buffer buffer_input(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * DATA_SIZE, input.data(), &err));  
    OCL_CHECK(err, cl::Buffer buffer_input_optimized(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * DATA_SIZE, input.data(), &err));  


    /* OUTPUT BUFFERS */
    OCL_CHECK(err, cl::Buffer buffer_output(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 1, output_hw.data(), &err));
    OCL_CHECK(err, cl::Buffer buffer_output_optimized(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 1, output_hw_opti.data(), &err));


    /* SETTING INPUT PARAMETERS */
    OCL_CHECK(err, err = krnl1.setArg(0, buffer_input));
    OCL_CHECK(err, err = krnl1.setArg(1, buffer_output));
    OCL_CHECK(err, err = krnl1.setArg(2, false));

    OCL_CHECK(err, err = krnl2.setArg(0, buffer_input_optimized));
    OCL_CHECK(err, err = krnl2.setArg(1, buffer_output_optimized));
    OCL_CHECK(err, err = krnl2.setArg(2, true));

    /*====================================================KERNEL===============================================================*/
    /* HOST -> DEVICE DATA TRANSFER*/
    std::cout << "HOST -> DEVICE [Kernel-1]" << std::endl;
    htod = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input}, 0 /* 0 means from host*/)); 
    q.finish();
    htod = clock() - htod;

      /*STARTING KERNEL(S)*/
    std::cout << "STARTING KERNEL - 1(S)" << std::endl;
    comp = clock();
    OCL_CHECK(err, err = q.enqueueTask(krnl1));
    q.finish();
    comp = clock() - comp;
    std::cout << "KERNEL - 1(S) FINISHED" << std::endl;

     /*DEVICE -> HOST DATA TRANSFER*/
    std::cout << "HOST <- DEVICE [Kernel-1]" << std::endl;
    dtoh = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    dtoh = clock() - dtoh;
    std::cout<<"\n==================================================================================================\n";
    
    
    std::cout << "HOST -> DEVICE [Kernel-2]" << std::endl;
    htod_opti = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_optimized}, 0 /* 0 means from host*/)); 
    q.finish();
    htod_opti = clock() - htod_opti;
  

    std::cout << "STARTING KERNEL - 2(S)" << std::endl;
    comp_opti = clock();
    OCL_CHECK(err, err = q.enqueueTask(krnl2));
    q.finish();
    comp_opti = clock() - comp_opti;
    std::cout << "KERNEL - 2(S) FINISHED" << std::endl;

    
    std::cout << "HOST <- DEVICE [Kernel-2]" << std::endl;
    dtoh_opti = clock();
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_optimized}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
    dtoh_opti = clock() - dtoh_opti;

    /*====================================================VERIFICATION & TIMING===============================================================*/

    //printf("Host -> Device : %lf ms\n", 1000.0 * htod/CLOCKS_PER_SEC);
    printf("Computation without optimization : %lf ms\n",  1000.0 * comp/CLOCKS_PER_SEC);
    printf("Computation with optimization : %lf ms\n",  1000.0 * comp_opti/CLOCKS_PER_SEC);
    //printf("Device -> Host : %lf ms\n", 1000.0 * dtoh/CLOCKS_PER_SEC);
    //printf("==========================To check:-================================\n");
    



    std::cout << std::endl; 
   
    bool match = true;

    std::cout << "HW:" << output_hw[0] << " SW:" << output_sw[0] << std::endl;
    if (output_hw[0] != output_sw[0]) match = false;     //std::cout << output_hw_opti[0] << std::endl;
    
    std::cout << std::endl << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    free(output_sw);
    free(input_sw);


    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}


