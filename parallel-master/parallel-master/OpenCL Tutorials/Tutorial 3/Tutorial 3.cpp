#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <vector>



#include <CL/cl.hpp>
#include "Utils.h"
#include <fstream>
#include <string>
#include <conio.h>



using namespace std;



void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	
	
	



	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;

	for (int i = 1; i < argc; i++) {
		if ((strcmp(argv[i], "-p") == 0) && (i < (argc - 1))) { platform_id = atoi(argv[++i]); }
		else if ((strcmp(argv[i], "-d") == 0) && (i < (argc - 1))) { device_id = atoi(argv[++i]); }
		else if (strcmp(argv[i], "-l") == 0) { std::cout << ListPlatformsDevices() << std::endl; }
		else if (strcmp(argv[i], "-h") == 0) { print_help(); }
	}

		try {
			//Part 2 - host operations

			//2.1 Select computing devices
			cl::Context context = GetContext(platform_id, device_id);

			//display the selected device
			std::cout << "Runinng on " << GetPlatformName(platform_id) << ", " << GetDeviceName(platform_id, device_id) << std::endl;

			//create a queue to which we will push commands for the device
			cl::CommandQueue queue(context);

			//2.2 Load & build the device code
			cl::Program::Sources sources;

			AddSources(sources, "my_kernels3.cl");

			cl::Program program(context, sources);

			//build and debug the kernel code
			try {
				program.build();
			}
			catch (const cl::Error& err) {
				std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
				std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
				std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(context.getInfo<CL_CONTEXT_DEVICES>()[0]) << std::endl;
				throw err;
			}

			typedef int mytype;

			//Part 4 - memory allocation
			//host - input
			std::vector<mytype> temperature;//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!
			

			string location;
			int year;
			int month;
			int day;
			int time;
			float temps;



			ifstream myfile;
			myfile.open("temp_lincolnshire_short.txt");

			while (!myfile.eof())
			{
				myfile >> location >> year >> month >> day >> time >> temps;
				temperature.push_back((int)(temps * 10.0f));
			}

			myfile.close();


			int sizeTemp = (int)temperature.size();

			std::vector<mytype> tempsAvg = temperature;
			std::vector<mytype> tempsMin = temperature;
			std::vector<mytype> tempsMax = temperature;
			//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
			//if the total input length is divisible by the workgroup size
			//this makes the code more efficient
			size_t local_size = 1024;

			size_t padding_size = temperature.size() % local_size;

			//if the input vector is not a multiple of the local_size
			//insert additional neutral elements (0 for addition) so that the total will not be affected
			if (padding_size) {
				//create an extra vector with neutral values
				std::vector<int> temperature_ext(local_size - padding_size, 0); //average
                tempsAvg.insert(tempsAvg.end(), temperature_ext.begin(), temperature_ext.end()); 


				std::vector<int> temperature_extMin1(local_size - padding_size, INT_MIN);

				tempsMax.insert(tempsMax.end(), temperature_extMin1.begin(), temperature_extMin1.end()); //min


				std::vector<int> temperature_ext2(local_size - padding_size, INT_MAX);

				tempsMin.insert(tempsMin.end(), temperature_ext2.begin(), temperature_ext2.end()); //max
				

				
			}

			size_t input_elements = tempsAvg.size();//number of input elements
			size_t input_size = tempsAvg.size()*sizeof(mytype);//size in bytes
			size_t nr_groups = input_elements / local_size;

			//host - output
			std::vector<mytype> B(1);
			size_t output_size = B.size()*sizeof(mytype);//size in bytes

			//device - buffers
			cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

			//Part 5 - device operations

			//5.1 copy array A to and initialise other arrays on device memory
			

			//5.2 Setup and execute all kernels (i.e. device code)
		 

			 //5.3 Copy the result from device to host

			////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &tempsMin[0]);
			queue.enqueueFillBuffer(buffer_B, INT_MAX, 0, output_size);

			cl::Kernel kernel_1 = cl::Kernel(program, "get_min_temp"); //min
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); //call all kernels in a sequence
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); //5.3 Copy the result from device to host
			std::cout << "Minimum temperature:" << (float)B[0] / 10.f << std::endl;

			//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &tempsMax[0]);
			queue.enqueueFillBuffer(buffer_B, INT_MIN, 0, output_size);
			
			kernel_1 = cl::Kernel(program, "get_max_temp"); //add all of the temperatures together
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); //call all kernels in a sequence
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]); //5.3 Copy the result from device to host
			
			std::cout << "Maximum temperature:" << (float)B[0] / 10.f << std::endl;
			

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &tempsAvg[0]);
			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory
			kernel_1 = cl::Kernel(program, "reduce_add_4"); //add all of the temperatures together
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); //call all kernels in a sequence
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			std::cout << "Average = " << (float)B[0] / (float)sizeTemp / 10.0f << std::endl;
			
			

			/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);
			kernel_1 = cl::Kernel(program, "hist_simple"); //add all of the temperatures together
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			//kernel_1.setArg(2, 5);              //cl::Local(local_size*sizeof(mytype)));//local memory size

			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size)); //call all kernels in a sequence
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			std::cout << "Hist = " <<B[0];
			
			
			
			
			
		}
		catch (cl::Error err) {
			std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		}
		getch();
		return 0;
}

