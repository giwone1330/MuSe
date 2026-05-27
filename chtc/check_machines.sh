condor_status -constraint 'CUDAGlobalMemoryMb>0' -af Name CUDADeviceName CUDACapability CUDAGlobalMemoryMb > condor_machines.txt
