// Name: Kyle Earp
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information about the GPU(s) in your system. 
 Also, and this is the fun part, be prepared to explain what each piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines

// Global variables

// Function prototypes
void cudaErrorCheck(const char*, int);

void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}

int main()
{
	cudaDeviceProp prop;

	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);
	printf(" You have %d GPUs in this machine\n", count);
	
	for (int i=0; i < count; i++) {
		cudaGetDeviceProperties(&prop, i);
		cudaErrorCheck(__FILE__, __LINE__);
		printf(" ---General Information for device %d ---\n", i);
		printf("Name: %s\n", prop.name); //ASCII string identifying the device
		printf("Compute capability: %d.%d\n", prop.major, prop.minor); //Compute capability of the device (major and minor revision numbers)
		printf("Clock rate: %d\n", prop.clockRate); //Clock frequency in kilohertz
		printf("Device copy overlap: ");
		if (prop.deviceOverlap) printf("Enabled\n"); //Device can concurrently copy memory between host and device while executing a kernel
		else printf("Disabled\n"); //Device cannot concurrently copy memory between host and device
		printf("Kernel execution timeout : ");
		if (prop.kernelExecTimeoutEnabled) printf("Enabled\n"); //Kernel execution time limit is enabled
		else printf("Disabled\n"); //Kernel execution time limit is disabled
		printf(" ---Memory Information for device %d ---\n", i);
		printf("Total global mem: %ld\n", prop.totalGlobalMem); //Total amount of global memory available on the device in bytes
		printf("Total constant Mem: %ld\n", prop.totalConstMem); //Total amount of constant memory available on the device in bytes
		printf("Max mem pitch: %ld\n", prop.memPitch); //Maximum pitch in bytes allowed by memory copy functions: memory pitch is the number of bytes between the starting addresses of consecutive rows in a 2D memory allocation, or the stride
		printf("Texture Alignment: %ld\n", prop.textureAlignment); //Alignment requirement for textures, textures must be aligned to this number of bytes; textures: specialized memory region on the GPU
		printf(" ---MP Information for device %d ---\n", i);
		printf("Multiprocessor count : %d\n", prop.multiProcessorCount); //Number of SMs on the GPU
		printf("Shared mem per mp: %ld\n", prop.sharedMemPerBlock); // Shared memory available per block in bytes
		printf("Registers per mp: %d\n", prop.regsPerBlock); // Maximum number of 32-bit registers available to a thread block
		printf("Threads in warp: %d\n", prop.warpSize); // Warp size in threads, number of threads that execute simultaneously as a single unit
		printf("Max threads per block: %d\n", prop.maxThreadsPerBlock); // How many threads can be in a single block
		printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]); // Maximum size of each dimension of a block (must multiply to maxThreadsPerBlock)
		printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]); // Maximum size of each dimension of a grid (must multiply to 16,000 something)
		printf(" ---Other Information for device %d ---\n", i);
		printf("16-byte unique identifier: %02x%02x%02x%02x-%02x%02x%02x%02x-%02x%02x%02x%02x-%02x%02x%02x%02x\n",
			prop.uuid.bytes[0], prop.uuid.bytes[1], prop.uuid.bytes[2], prop.uuid.bytes[3],
			prop.uuid.bytes[4], prop.uuid.bytes[5], prop.uuid.bytes[6], prop.uuid.bytes[7],
			prop.uuid.bytes[8], prop.uuid.bytes[9], prop.uuid.bytes[10], prop.uuid.bytes[11],
			prop.uuid.bytes[12], prop.uuid.bytes[13], prop.uuid.bytes[14], prop.uuid.bytes[15]); // 16-byte unique identifier for the device
		printf("8-byte locally unique identifier (undefined on TCC and non-Windows platforms): %s\n", prop.luid); //Not applicable
		printf("LUID device node mask (undefined on TCC and non-Windows platforms): %u\n", prop.luidDeviceNodeMask); // Not applicable
		printf("Pitch alignment requirement for texture regerences bound to pitched memory: %zu\n", prop.texturePitchAlignment); // Alignment requirement for textures bound to pitched memory
		printf("Device is ");
		if (prop.integrated) printf("integrated\n"); // Device is integrated with the host
		else printf("discrete\n"); // Device is discrete from the host
		printf("Device can");
		if(!prop.canMapHostMemory) printf("not"); // Device cannot map host memory
		printf(" map host memory with cudaHostAlloc/cudaHostGetDevicePointer\n"); // Device can map host memory
		printf("Compute mode: "); // Compute mode of the device
		if (prop.computeMode == cudaComputeModeDefault) printf("Default - Device is not restricted and multiple threads can use cudaSetDevice() with this device\n");
		else if (prop.computeMode == cudaComputeModeProhibited) printf("Prohibited - No threads can use cudaSetDevice()\n");
		else if (prop.computeMode == cudaComputeModeExclusiveProcess) printf("Exclusive Process - Many threads in one process will be able to use cudaSetDevice()\n");
		printf("Maximum 1D texture size: %d\n", prop.maxTexture1D); // Maximum size of a 1D texture
		printf("Maximum 1D mipmapped texture size: %d\n", prop.maxTexture1D); // Maximum size of a 1D mipmapped texture
		printf("Maximum 1D texture size for textures bound to linear memory: %d\n", prop.maxTexture1DLinear); // Maximum size of a 1D texture bound to linear memory
		printf("Maximum 2D texture dimensions: (%d, %d)\n", prop.maxTexture2D[0], prop.maxTexture2D[1]); // Maximum size of a 2D texture
		printf("Maximum 2D mipmapped texture dimensions: (%d, %d)\n", prop.maxTexture2DMipmap[0], prop.maxTexture2DMipmap[1]); // Maximum size of a 2D mipmapped texture
		printf("Maximum 2D texture dimensions if texture bound to linear memory: (%d, %d)\n", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1]); // Maximum size of a 2D texture bound to linear memory
		printf("Maximum 2D texture dimensions if texture gather operations have to be performed: (%d, %d)\n", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]); // Maximum size of a 2D texture for texture gather operations
		printf("Maximum 3D texture dimensions: (%d, %d, %d)\n", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]); // Maximum size of a 3D texture
		printf("Maximum alternate 3D texture dimensions: (%d, %d, %d)\n", prop.maxTexture3DAlt[0], prop.maxTexture3DAlt[1], prop.maxTexture3DAlt[2]); // Maximum size of an alternate 3D texture
		printf("Maximum Cubemap texture dimensions: %d\n", prop.maxTextureCubemap); // Maximum size of a Cubemap texture
		printf("Maximum 1D layered texture dimensions: (%d, %d)\n", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]); // Maximum size of a 1D layered texture
		printf("Maximum 2D layered texture dimensions: (%d, %d, %d)\n", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]); // Maximum size of a 2D layered texture
		printf("Maximum Cubemap layered texture dimensions: (%d, %d)\n", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]); // Maximum size of a Cubemap layered texture
		printf("Maximum 1D surface size: %d\n", prop.maxSurface1D); // Maximum size of a 1D surface
		printf("Maximum 2D surface dimensions: (%d, %d)\n", prop.maxSurface2D[0], prop.maxSurface2D[1]); // Maximum size of a 2D surface
		printf("Maximum 3D surface dimensions: (%d, %d, %d)\n", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]); // Maximum size of a 3D surface
		printf("Maximum 1D layered surface dimensions: (%d, %d)\n", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]); // Maximum size of a 1D layered surface
		printf("Maximum 2D layered surface dimensions: (%d, %d, %d)\n", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]); // Maximum size of a 2D layered surface
		printf("Maximum Cubemap surface dimensions: %d\n", prop.maxSurfaceCubemap); // Maximum size of a Cubemap surface
		printf("Maximum Cubemap layered surface dimensions: (%d, %d)\n", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]); // Maximum size of a Cubemap layered surface
		printf("Surface alignment: %zu\n", prop.surfaceAlignment); // Alignment requirement for surfaces
		printf("Concurrent kernel execution: ");
		if (prop.concurrentKernels) printf("Yes\n"); // Device can execute multiple kernels concurrently
		else printf("No\n"); // Device cannot execute multiple kernels concurrently
		printf("ECC support: ");
		if (prop.ECCEnabled) printf("Enabled\n"); // Device has ECC support
		else printf("Disabled\n"); // Device does not have ECC support
		printf("PCI bus ID: %d\n", prop.pciBusID); // PCI bus ID of the device
		printf("PCI device ID: %d\n", prop.pciDeviceID); // PCI device ID of the device
		printf("PCI domain ID: %d\n", prop.pciDomainID); // PCI domain ID of the device
		printf("TCC driver: ");
		if (prop.tccDriver) printf("Yes\n"); // Device is using TCC driver
		else printf("No\n"); // Device is not using TCC driver
		printf("Async engine: ");
		if (prop.asyncEngineCount == 1) printf("Device can concurrently copy memory between host and device while executing a kernel\n"); // Device can concurrently copy memory between host and device in one direction
		else if (prop.asyncEngineCount == 2) printf("Device can concurrently copy memory between host and device in both directions\n"); // Device can concurrently copy memory between host and device in both directions
		else printf("Device cannot concurrently copy memory between host and device\n"); // Device cannot concurrently copy memory between host and device
		printf("Unified address space: ");
		if (prop.unifiedAddressing) printf("Yes\n"); // Device shares a unified address space with the host
		else printf("No\n"); // Device does not share a unified address space with the host
		printf("Peak memory clock frequency in kilohertz: %d\n", prop.memoryClockRate); // Peak memory clock frequency in kilohertz
		printf("Memory bus width in bits: %d\n", prop.memoryBusWidth); // Memory bus width in bits
		printf("L2 cache size in bytes: %d\n", prop.l2CacheSize); // Size of L2 cache in bytes
		printf("Maximum persisting lines size in bytes: %d\n", prop.persistingL2CacheMaxSize); // L2 cache's maximum persisting lines size in bytes
		printf("Maximum resident threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor); // Maximum number of threads that can reside on a multiprocessor
		printf("Stream priorities supported: ");
		if (prop.streamPrioritiesSupported) printf("Yes\n"); // Device supports stream priorities
		else printf("No\n"); // Device does not support stream priorities
		printf("Global L1 cache supported: ");
		if (prop.globalL1CacheSupported) printf("Yes\n"); // Device supports global L1 cache
		else printf("No\n"); // Device does not support global L1 cache
		printf("Local L1 cache supported: ");
		if (prop.localL1CacheSupported) printf("Yes\n"); // Device supports local L1 cache
		else printf("No\n"); // Device does not support local L1 cache
		printf("Maximum amount of shared momory available to a multiprocessor in bytes, this amount is shared by all thread blocks simultaneously resident on a multiprocessor: %ld\n", prop.sharedMemPerMultiprocessor); // Maximum amount of shared memory available to a multiprocessor in bytes
		printf("Maximum number of 32-bit registers available to a multiprocessor: %d\n", prop.regsPerMultiprocessor); // Maximum number of 32-bit registers available to a multiprocessor
		printf("Allocating managed memory is ");
		if (prop.managedMemory) printf("supported\n"); // Device supports allocating managed memory on the device
		else printf("not supported\n"); // Device does not support allocating managed memory on the device
		printf("Device is ");
		if (prop.isMultiGpuBoard) printf("multi-GPU board\n"); // Device is a multi-GPU board
		else printf("single-GPU board\n"); // Device is a single-GPU board
		printf("Multi-GPU board group ID: %d\n", prop.multiGpuBoardGroupID); // Multi-GPU board group ID
		printf("Device and host ");
		if (prop.hostNativeAtomicSupported) printf("support native atomic operations\n"); // Device and host support native atomic operations
		else printf("do not support native atomic operations\n"); // Device and host do not support native atomic operations
		printf("Ratio of single precision performance (in floating-point operations per second) to double precision performance: %d\n", prop.singleToDoublePrecisionPerfRatio); // Ratio of single precision performance to double precision performance
		printf("Device supports coherently accessing pageable memory from the device: ");
		if (prop.pageableMemoryAccess) printf("Yes\n"); // Device can coherently access pageable memory from the device
		else printf("No\n"); // Device cannot coherently access pageable memory from the device
		printf("Device can coherently acces managed memory concurrently with the CPU: ");
		if (prop.concurrentManagedAccess) printf("Yes\n"); // Device can coherently access managed memory concurrently with the CPU
		else printf("No\n"); // Device cannot coherently access managed memory concurrently with the CPU
		printf("Compute Preemption: ");
		if (prop.computePreemptionSupported) printf("Yes\n"); // Device supports compute preemption
		else printf("No\n"); // Device does not support compute preemption
		printf("Device can access host registered memory at the same virtual address as the CPU: ");
		if (prop.canUseHostPointerForRegisteredMem) printf("Yes\n"); // Device can access host registered memory at the same virtual address as the CPU
		else printf("No\n"); // Device cannot access host registered memory at the same virtual address as the CPU
		printf("Device supports launching cooperative kernels via cudaLaunchCooperativeKernel: ");
		if (prop.cooperativeLaunch) printf("Yes\n"); // Device supports launching cooperative kernels via cudaLaunchCooperativeKernel
		else printf("No\n"); // Device does not support launching cooperative kernels via cudaLaunchCooperativeKernel
		printf("Maximum shared memory per block usable by special opt in: %ld\n", prop.sharedMemPerBlockOptin); // Maximum shared memory per block usable by special opt in
		printf("Device accesses pageable memory via the host\'s page tables: ");
		if (prop.pageableMemoryAccessUsesHostPageTables) printf("Yes\n"); // Device accesses pageable memory via the host's page tables
		else printf("No\n"); // Device does not access pageable memory via the host's page tables
		printf("Host can directly access managed memory on the device without migration: ");
		if (prop.directManagedMemAccessFromHost) printf("Yes\n"); // Host can directly access managed memory on the device without migration
		else printf("No\n"); // Host cannot directly access managed memory on the device without migration
		printf("Maximum number of thread blocks that can reside on a multiprocessor: %d\n", prop.maxBlocksPerMultiProcessor); // Maximum number of thread blocks that can reside on a multiprocessor
		printf("Maximum value of cudaAccessPolicyWindow::num_bytes for the current device: %d\n", prop.accessPolicyMaxWindowSize); // Maximum value of cudaAccessPolicyWindow::num_bytes for the current device
		printf("Shared memory reserved by CUDA driver per block in bytes: %ld\n", prop.reservedSharedMemPerBlock); // Shared memory reserved by CUDA driver per block in bytes
		printf("\n");
	}	
	return(0);
}

