#pragma once
#include <cuda_runtime.h>

#include <src/utils/disable_copy.h>
#include <src/utils/hpc_helper.h>

template<typename T>
struct CudaArray : DisableCopyAllowMove
{
    cudaArray* m_cudaArray{};
    uint3 m_dim{};

    explicit CudaArray(uint3 const& _dim)
        : m_dim(_dim)
    {
        cudaExtent extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();
        checkCudaErrors(cudaMalloc3DArray(&m_cudaArray, &channelDesc, extent, cudaArraySurfaceLoadStore));
    }

    ~CudaArray()
    {
        checkCudaErrors(cudaFreeArray(m_cudaArray));
    }

    cudaArray* getArray() const
    {
        return m_cudaArray;
    }

    void copyIn(T const *_data)
    {
        cudaMemcpy3DParms copyParams{};
        copyParams.srcPtr = make_cudaPitchedPtr((void*) _data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copyParams.dstArray = m_cudaArray;
        copyParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copyParams.kind = cudaMemcpyHostToDevice;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }

    void copyOut(T *_data)
    {
        cudaMemcpy3DParms copyParams{};
        copyParams.srcArray = m_cudaArray;
        copyParams.dstPtr = make_cudaPitchedPtr((void*) _data, m_dim.x * sizeof(T), m_dim.x, m_dim.y);
        copyParams.extent = make_cudaExtent(m_dim.x, m_dim.y, m_dim.z);
        copyParams.kind = cudaMemcpyDeviceToHost;
        checkCudaErrors(cudaMemcpy3D(&copyParams));
    }
};

/**
 * cudaBoundaryModeZero: 
 *   当访问 surface 时，如果坐标超出边界，将返回零值。
 *   这种模式常用于避免错误，同时确保超出边界的读取操作不引发异常。
 * cudaBoundaryModeClamp: 
 *   超出边界的访问会被“钳制”到合法的边界值。
 *   例如，如果访问的坐标小于0，则会使用0，如果大于最大边界，则使用最大合法坐标。
 *   这种模式确保了访问始终位于合法范围内。
 * cudaBoundaryModeTrap: 
 *   当访问超出边界时，会触发运行时错误（trap）。
 *   这种模式会显式地报告边界违规，适合调试和确保程序不会超出内存限制的场景。
*/
// ================== 表面对象 ==================
// 访问者模式，用于访问 surface, 是作为指向资源的弱引用, 可以随意拷贝
// 将数据的读写访问方法放置在访问者类中，而不是在 surface 类中，可以更好地封装数据访问逻辑，并提高代码的可读性和可维护性。
template<typename T>
struct CudaSurfaceAccessor
{
    cudaSurfaceObject_t m_cudaSuface;

    template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ T read(int x, int y, int z) const
    {
        return surf3Dread<T>(m_cudaSuface, x * sizeof(T), y, z, mode);
    }

    template<cudaSurfaceBoundaryMode mode = cudaBoundaryModeTrap>
    __device__ __forceinline__ void write(T value, int x, int y, int z) const
    {
        surf3Dwrite(value, m_cudaSuface, x * sizeof(T), y, z, mode);
    }
};

template<typename T>
struct CudaSurface : CudaArray<T>
{
    cudaSurfaceObject_t m_cudaSurface;  

    explicit CudaSurface(uint3 const& _dim)
        : CudaArray<T>(_dim)
    {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaArray<T>::getArray();
        checkCudaErrors(cudaCreateSurfaceObject(&m_cudaSurface, &resDesc));
    }

    ~CudaSurface()
    {
        checkCudaErrors(cudaDestroySurfaceObject(m_cudaSurface));
    }

    cudaSurfaceObject_t getSurface() const
    {
        return m_cudaSurface;
    }

    CudaSurfaceAccessor<T> accessorSurface() const
    {
        return {m_cudaSurface};
    }
};

// ================== 表面对象 ==================

// ================== 纹理对象 ==================
template<typename T>
struct CudaTextureAccessor
{
    cudaTextureObject_t m_cudaTexture;

    __device__ __forceinline__ T sample(float x, float y, float z) const
    {
        return tex3D<T>(m_cudaTexture, x, y, z);
    }
};

template<typename T>
struct CudaTexture : CudaSurface<T>
{
    struct Parameters
    {
        cudaTextureAddressMode addressMode{cudaAddressModeBorder};
        cudaTextureFilterMode filterMode{cudaFilterModeLinear};
        cudaTextureReadMode readMode{cudaReadModeElementType};
        bool normalizedCoords{false};
    };

    cudaTextureObject_t m_cudaTexture;

    explicit CudaTexture(uint3 const& _dim, Parameters const &_args = {})
        : CudaSurface<T>(_dim)
    {
        cudaResourceDesc resDesc{};
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = CudaArray<T>::getArray();

        cudaTextureDesc texDesc{};
        texDesc.addressMode[0] = _args.addressMode;
        texDesc.addressMode[1] = _args.addressMode;
        texDesc.addressMode[2] = _args.addressMode;
        texDesc.filterMode = _args.filterMode;
        texDesc.readMode = _args.readMode;
        texDesc.normalizedCoords = _args.normalizedCoords;

        checkCudaErrors(cudaCreateTextureObject(&m_cudaTexture, &resDesc, &texDesc, nullptr));
    }

    ~CudaTexture()
    {
        checkCudaErrors(cudaDestroyTextureObject(m_cudaTexture));
    }

    cudaTextureObject_t getTexture() const
    {
        return m_cudaTexture;
    }

    CudaTextureAccessor<T> accessorTexture() const
    {
        return {m_cudaTexture};
    }
};


// ================== 纹理对象 ==================