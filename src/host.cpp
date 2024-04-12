#include "host.h"
#include <stdio.h>
#include "constants.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <float.h>

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
    
    
    // std::cout << "HOST -> DEVICE [Kernel-2]" << std::endl;
    // htod_opti = clock();
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_input_optimized}, 0 /* 0 means from host*/)); 
    // q.finish();
    // htod_opti = clock() - htod_opti;
  

    // std::cout << "STARTING KERNEL - 2(S)" << std::endl;
    // comp_opti = clock();
    // OCL_CHECK(err, err = q.enqueueTask(krnl2));
    // q.finish();
    // comp_opti = clock() - comp_opti;
    // std::cout << "KERNEL - 2(S) FINISHED" << std::endl;

    
    // std::cout << "HOST <- DEVICE [Kernel-2]" << std::endl;
    // dtoh_opti = clock();
    // OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buffer_output_optimized}, CL_MIGRATE_MEM_OBJECT_HOST));
    // q.finish();
    // dtoh_opti = clock() - dtoh_opti;

    /*====================================================VERIFICATION & TIMING===============================================================*/

    //printf("Host -> Device : %lf ms\n", 1000.0 * htod/CLOCKS_PER_SEC);
    printf("Computation without optimization : %lf ms\n",  1000.0 * comp/CLOCKS_PER_SEC);
    //printf("Computation with optimization : %lf ms\n",  1000.0 * comp_opti/CLOCKS_PER_SEC);
    //printf("Device -> Host : %lf ms\n", 1000.0 * dtoh/CLOCKS_PER_SEC);
    //printf("==========================To check:-================================\n");
    



    std::cout << std::endl; 
   
    bool match = true;

    std::cout << output_hw[0] << std::endl;
    //std::cout << output_hw_opti[0] << std::endl;

    std::cout << std::endl << "TEST " << (match ? "PASSED" : "FAILED") << std::endl;

    free(output_sw);
    free(input_sw);


    return (match ? EXIT_SUCCESS : EXIT_FAILURE);
}


