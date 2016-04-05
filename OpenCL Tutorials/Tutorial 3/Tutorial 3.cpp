#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#define __CL_ENABLE_EXCEPTIONS
#include <iostream>
#include <vector>
#include <CL/cl.hpp>
#include "Utils.h"
#include <fstream>
#include <string>
#include <conio.h>
#include <algorithm>
#include <functional>


using namespace std;



void print_help() {
	std::cerr << "Application usage:" << std::endl;

	std::cerr << "  -p : select platform " << std::endl;
	std::cerr << "  -d : select device" << std::endl;
	std::cerr << "  -l : list all platforms and devices" << std::endl;
	std::cerr << "  -h : print this message" << std::endl;
}

int main(int argc, char **argv) {
	std::vector<string> station;
	std::vector<int>    year;
	std::vector<int>    date;
	std::vector<int>    day;
	std::vector<int>    time;
	std::vector<int> temperature;
	string data = "";
	
	
	string userInput;
	int counter = 0;



	//Part 1 - handle command line options such as device selection, verbosity, etc.
	int platform_id = 0;
	int device_id = 0;
	ifstream myfile;


		//cout << "Enter the loaction of the file" << endl;
		//std::cin >> userInput;
		myfile.open("temp_lincolnshire_short.txt"); //, std::ofstream::out, std::ofstream::app);
		

		if (myfile.is_open()) {

			cout << "Reading in data from file" << endl;
		

			while (myfile >> data) {

				switch (counter) {

				case 0:
					station.push_back(data);
					counter++;
					break;
				case 1:
					year.push_back(stoi(data));
					counter++;
					break;

				case 2:
					date.push_back(stoi(data));
					counter++;
					break;

				case 3:
					day.push_back(stoi(data));
					counter++;
					break;

				case 4:
					time.push_back(stoi(data));
					counter++;
					break;

				case 5:
					temperature.push_back(stoi(data));
					
					counter++;
					counter = 0;
					break;
				}
				
				
			}
			cout << temperature.size();
			std::transform(temperature.begin(), temperature.end(), temperature.begin(),
				std::bind1st(std::multiplies<float>(), 10.0f)); //multiply all the vector elements by 10

			}   myfile.close();
			float temps = temperature.size();
			
			
		
		//detect any potential exceptions
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
			//std::vector<mytype> A(, 60);//allocate 10 elements with an initial value 1 - their sum is 10 so it should be easy to check the results!

			//the following part adjusts the length of the input vector so it can be run for a specific workgroup size
			//if the total input length is divisible by the workgroup size
			//this makes the code more efficient
			size_t local_size = 1024;

			size_t padding_size = temperature.size() % local_size;

			//if the input vector is not a multiple of the local_size
			//insert additional neutral elements (0 for addition) so that the total will not be affected
			if (padding_size) {
				//create an extra vector with neutral values
				std::vector<int> temp_ext(local_size - padding_size, 0);
				//append that extra vector to our input
				temperature.insert(temperature.end(), temp_ext.begin(), temp_ext.end());
			}

			size_t input_elements = temperature.size();//number of input elements
			size_t input_size = temperature.size()*sizeof(mytype);//size in bytes
			size_t nr_groups = input_elements / local_size;

			//host - output
			std::vector<mytype> B(input_elements);
			size_t output_size = B.size()*sizeof(mytype);//size in bytes

			//device - buffers
			cl::Buffer buffer_A(context, CL_MEM_READ_ONLY, input_size);
			cl::Buffer buffer_B(context, CL_MEM_READ_WRITE, output_size);

			//Part 5 - device operations

			//5.1 copy array A to and initialise other arrays on device memory
			queue.enqueueWriteBuffer(buffer_A, CL_TRUE, 0, input_size, &temperature[0]);
			queue.enqueueFillBuffer(buffer_B, 0, 0, output_size);//zero B buffer on device memory

			//5.2 Setup and execute all kernels (i.e. device code)
			cl::Kernel kernel_1 = cl::Kernel(program, "reduce_add_4");
			kernel_1.setArg(0, buffer_A);
			kernel_1.setArg(1, buffer_B);
			kernel_1.setArg(2, cl::Local(local_size*sizeof(mytype)));//local memory size

					//call all kernels in a sequence
			queue.enqueueNDRangeKernel(kernel_1, cl::NullRange, cl::NDRange(input_elements), cl::NDRange(local_size));

			//5.3 Copy the result from device to host
			queue.enqueueReadBuffer(buffer_B, CL_TRUE, 0, output_size, &B[0]);

			std::cout << "A = " << temperature[0] << std::endl;
			std::cout << "Average = " << (float)(B[0] / (float)temps) / 10.0f << std::endl;
			
			
		}
		catch (cl::Error err) {
			std::cerr << "ERROR: " << err.what() << ", " << getErrorString(err.err()) << std::endl;
		}
		getch();
		return 0;
}

