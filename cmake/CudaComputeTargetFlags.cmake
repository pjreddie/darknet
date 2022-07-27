#
#  Compute target flags macros by Anatoly Baksheev
# 
#  Usage in CmakeLists.txt:
#   	include(CudaComputeTargetFlags.cmake)
#		APPEND_TARGET_ARCH_FLAGS() 

#compute flags macros
MACRO(CUDA_COMPUTE_TARGET_FLAGS arch_bin arch_ptx cuda_nvcc_target_flags)
	string(REGEX REPLACE "\\." "" ARCH_BIN_WITHOUT_DOTS "${${arch_bin}}")
	string(REGEX REPLACE "\\." "" ARCH_PTX_WITHOUT_DOTS "${${arch_ptx}}")
								
	set(cuda_computer_target_flags_temp "") 
	
	# Tell NVCC to add binaries for the specified GPUs
	string(REGEX MATCHALL "[0-9()]+" ARCH_LIST "${ARCH_BIN_WITHOUT_DOTS}")
	foreach(ARCH IN LISTS ARCH_LIST)
		if (ARCH MATCHES "([0-9]+)\\(([0-9]+)\\)")
			# User explicitly specified PTX for the concrete BIN					
			set(cuda_computer_target_flags_temp ${cuda_computer_target_flags_temp} -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})					
		else()					
			# User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN                				
			set(cuda_computer_target_flags_temp ${cuda_computer_target_flags_temp} -gencode arch=compute_${ARCH},code=sm_${ARCH})					
		endif()
	endforeach()
				
	# Tell NVCC to add PTX intermediate code for the specified architectures
	string(REGEX MATCHALL "[0-9]+" ARCH_LIST "${ARCH_PTX_WITHOUT_DOTS}")
	foreach(ARCH IN LISTS ARCH_LIST)				
		set(cuda_computer_target_flags_temp ${cuda_computer_target_flags_temp} -gencode arch=compute_${ARCH},code=compute_${ARCH})				
	endforeach()	
							
	set(${cuda_nvcc_target_flags} ${cuda_computer_target_flags_temp})		
ENDMACRO()

MACRO(APPEND_TARGET_ARCH_FLAGS)
	set(cuda_nvcc_target_flags "")
	CUDA_COMPUTE_TARGET_FLAGS(CUDA_ARCH_BIN CUDA_ARCH_PTX cuda_nvcc_target_flags)		
	if (cuda_nvcc_target_flags)
		message(STATUS "CUDA NVCC target flags: ${cuda_nvcc_target_flags}")
		list(APPEND CUDA_NVCC_FLAGS ${cuda_nvcc_target_flags})
	endif()
ENDMACRO()