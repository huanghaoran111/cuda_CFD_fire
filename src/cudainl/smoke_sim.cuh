#pragma once
#include <memory>
#include <vector>
#include <iostream>

#include <src/utils/disable_copy.h>
#include <src/cudainl/cuda_array.cuh>

// ============== 对流部分-函数声明 ================

__global__ void advect_kernel(
    CudaTextureAccessor<float4> texVel, 
    CudaSurfaceAccessor<float4> surLoc, 
    CudaSurfaceAccessor<char> sufBound, 
    unsigned int n
);


template <class T>
__global__ void resample_kernel(
    CudaSurfaceAccessor<float4> sufLoc, 
    CudaTextureAccessor<T> texClr,
    CudaSurfaceAccessor<T> sufClrNext,
    unsigned int n
)
{
    // TODO: 根据对流后的位置重新采样
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;

    float4 loc = sufLoc.read(x, y, z);
    T clr = texClr.sample(loc.x, loc.y, loc.z);
    sufClrNext.write(clr, x, y, z);
}


__global__ void heatup_kernel(
    CudaSurfaceAccessor<float4> sufVel, 
    CudaSurfaceAccessor<float> sufTmp, 
    CudaSurfaceAccessor<float> sufClr,
    CudaSurfaceAccessor<char> sufBound,
    float tmpAmbient,
    float heatRate,
    float clrRate,
    unsigned int n
);

__global__ void decay_kernel(
    CudaSurfaceAccessor<float> sufTmp, 
    CudaSurfaceAccessor<float> sufTmpNext, 
    CudaSurfaceAccessor<char> sufBound, 
    float ambientRate, 
    float decayRate, 
    unsigned int n
);

// ============== 对流部分-函数声明 ================

// ============== 投影部分-函数声明 ================
__global__ void divergence_kernel(
    CudaSurfaceAccessor<float4> sufLoc, 
    CudaSurfaceAccessor<float> sufDiv, 
    CudaSurfaceAccessor<char> sufBound,
    unsigned int n
);

template<int phase>
__global__ void rbgs_kernel(
    CudaSurfaceAccessor<float> sufPre,
    CudaSurfaceAccessor<float> sufDiv,
    unsigned int n
);

__global__ void residual_kernel(
    CudaSurfaceAccessor<float> sufRes,
    CudaSurfaceAccessor<float> sufPre,
    CudaSurfaceAccessor<float> sufDiv,
    unsigned int n
);

__global__ void restrict_kernel(
    CudaSurfaceAccessor<float> sufPreNext, 
    CudaSurfaceAccessor<float> sufPre, 
    unsigned int n
);

__global__ void fillzero_kernel(
    CudaSurfaceAccessor<float> sufPre,
    unsigned int n
);

__global__ void prolongate_kernel(
    CudaSurfaceAccessor<float> sufPreNext,
    CudaSurfaceAccessor<float> sufPre,
    unsigned int n
);

__global__ void subgradient_kernel(
    CudaSurfaceAccessor<float> sufPre,
    CudaSurfaceAccessor<float4> sufVel,
    CudaSurfaceAccessor<char> sufBound,
    unsigned int n
);

// ============== 投影部分-函数声明 ================

// ============== 误差计算 ================
__global__ void sumloss_kernel(
    CudaSurfaceAccessor<float> sufDiv, 
    float *sum, 
    unsigned int n
);
// ============== 误差计算 ===============

// ============== 边界条件 ===============

// ============== 边界条件 ===============

struct SmokeSim : DisableCopyDisableMove
{
    unsigned int n;
    std::unique_ptr<CudaSurface<float4>> loc;
    std::unique_ptr<CudaTexture<float4>> vel;
    std::unique_ptr<CudaTexture<float4>> velNext;
    std::unique_ptr<CudaTexture<float>> clr;
    std::unique_ptr<CudaTexture<float>> clrNext;
    std::unique_ptr<CudaTexture<float>> tmp;
    std::unique_ptr<CudaTexture<float>> tmpNext;

    std::unique_ptr<CudaSurface<char>> bound;
    std::unique_ptr<CudaSurface<float>> div;
    std::unique_ptr<CudaSurface<float>> pre;
    std::vector<std::unique_ptr<CudaSurface<float>>> res;
    std::vector<std::unique_ptr<CudaSurface<float>>> res2;
    std::vector<std::unique_ptr<CudaSurface<float>>> err2;
    std::vector<unsigned int> sizes;

    explicit SmokeSim(unsigned int _n, unsigned int _n0 = 16)
        : n(_n)
        , loc(std::make_unique<CudaSurface<float4>>(uint3{n, n, n}))
        , vel(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
        , velNext(std::make_unique<CudaTexture<float4>>(uint3{n, n, n}))
        , clr(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
        , clrNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
        , tmp(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
        , tmpNext(std::make_unique<CudaTexture<float>>(uint3{n, n, n}))
        , div(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
        , pre(std::make_unique<CudaSurface<float>>(uint3{n, n, n}))
        , bound(std::make_unique<CudaSurface<char>>(uint3{n, n, n}))
    {
        fillzero_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(pre->accessorSurface(), n);

        unsigned int tn;
        for(tn = n; tn >= _n0; tn /= 2)
        {
            res.push_back(std::make_unique<CudaSurface<float>>(uint3{tn, tn, tn}));
            res2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            err2.push_back(std::make_unique<CudaSurface<float>>(uint3{tn/2, tn/2, tn/2}));
            sizes.push_back(tn);
        }
    }

    void smooth(CudaSurface<float> *v, CudaSurface<float> *f, unsigned int lev, int times = 4)
    {
        unsigned int tn = sizes[lev];
        for(int step = 0; step < times; step++)
        {
            rbgs_kernel<0><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
                v->accessorSurface(),
                f->accessorSurface(),
                tn
            );
            rbgs_kernel<1><<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
                v->accessorSurface(),
                f->accessorSurface(),
                tn
            );
        }
    }

    void vcycle(unsigned int lev, CudaSurface<float> *v, CudaSurface<float> *f)
    {
        if(lev >= sizes.size())
        {
            unsigned int tn = sizes.back();
            smooth(v, f, lev);
            return;
        }
        auto *r = res[lev].get();
        auto *r2 = res2[lev].get();
        auto *e2 = err2[lev].get();
        unsigned int tn = sizes[lev];
        smooth(v, f, lev);
        residual_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
            r->accessorSurface(),
            v->accessorSurface(),
            f->accessorSurface(),
            tn
        );
        restrict_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
            r2->accessorSurface(),
            r->accessorSurface(),
            tn / 2
        );
        fillzero_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
            e2->accessorSurface(),
            tn / 2
        );
        vcycle(lev + 1, e2, r2);
        prolongate_kernel<<<dim3((tn + 7) / 8, (tn + 7) / 8, (tn + 7) / 8), dim3(8, 8, 8)>>>(
            v->accessorSurface(),
            e2->accessorSurface(),
            tn / 2
        );
        smooth(v, f, lev);
    }

    void projection()
    {
        heatup_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessorSurface(),
            tmp->accessorSurface(),
            clr->accessorSurface(),
            bound->accessorSurface(),
            0.05f,
            0.018f,
            0.004f,
            n
        );
        divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessorSurface(),
            div->accessorSurface(),
            bound->accessorSurface(),
            n
        );
        vcycle(0, pre.get(), div.get());
        subgradient_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            pre->accessorSurface(),
            vel->accessorSurface(),
            bound->accessorSurface(),
            n
        );
        
    }

    void advection()
    {
        advect_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessorTexture(),
            loc->accessorSurface(),
            bound->accessorSurface(),
            n
        );
        
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            loc->accessorSurface(),
            vel->accessorTexture(),
            velNext->accessorSurface(),
            n
        );
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            loc->accessorSurface(),
            clr->accessorTexture(),
            clrNext->accessorSurface(),
            n
        );
        resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            loc->accessorSurface(),
            tmp->accessorTexture(),
            tmpNext->accessorSurface(),
            n
        );
        vel.swap(velNext);
        clr.swap(clrNext);
        tmp.swap(tmpNext);
        // resample_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
        //     loc->accessorSurface(),
        //     tmp->accessorTexture(),
        //     tmpNext->accessorSurface(),
        //     n
        // );
        decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            tmp->accessorSurface(), 
            tmpNext->accessorSurface(), 
            bound->accessorSurface(), 
            std::exp(-0.5f), 
            std::exp(-0.0003f), 
            n
        );
        decay_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            clr->accessorSurface(), 
            clrNext->accessorSurface(), 
            bound->accessorSurface(), 
            std::exp(-0.05f), 
            std::exp(-0.003f), 
            n
        );
        
        tmp.swap(tmpNext);
        clr.swap(clrNext);
        // tmp.swap(tmpNext);
    }

    void step(int times = 16)
    {
        for(int step = 0; step < times; step++)
        {
            projection();
            advection();
        }
    }

    float calc_loss()
    {
        // TODO: 计算未消除的散度，查看是否为不可压缩流
        divergence_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            vel->accessorSurface(),
            div->accessorSurface(),
            bound->accessorSurface(),
            n
        );

        float *sum;
        checkCudaErrors(cudaMalloc(&sum, sizeof(float)));
        checkCudaErrors(cudaMemset(sum, 0, sizeof(float)));
        sumloss_kernel<<<dim3((n + 7) / 8, (n + 7) / 8, (n + 7) / 8), dim3(8, 8, 8)>>>(
            div->accessorSurface(),
            sum,
            n
        );
        float cpu;
        checkCudaErrors(cudaMemcpy(&cpu, sum, sizeof(float), cudaMemcpyDeviceToHost));
        checkCudaErrors(cudaFree(sum));
        return cpu;
    }
};

__global__ void advect_kernel(
    CudaTextureAccessor<float4> texVel,
    CudaSurfaceAccessor<float4> surLoc,
    CudaSurfaceAccessor<char> sufBound,
    unsigned int n
)
{
    // TODO: 三线性插值实现半拉格朗日对流(计算对流后的位置-RK3)
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;

    auto sample = [](CudaTextureAccessor<float4> tex, float3 loc) -> float3 {
        float4 vel = tex.sample(loc.x, loc.y, loc.z);
        return make_float3(vel.x, vel.y, vel.z);
    };

    float3 loc = make_float3(x + 0.5f, y + 0.5f, z + 0.5f);
    if(sufBound.read(x, y, z) >= 0)
    {
        float3 vel1 = sample(texVel, loc);
        float3 vel2 = sample(texVel, loc - 0.5f * vel1);
        float3 vel3 = sample(texVel, loc - 0.75f * vel2);
        loc -= (2.f / 9.f) * vel1 + (1.f / 3.f) * vel2 + (4.f / 9.f) * vel3;
    }
    surLoc.write(make_float4(loc.x, loc.y, loc.z, 0.f), x, y, z);
}



__global__ void divergence_kernel(
    CudaSurfaceAccessor<float4> sufVel, 
    CudaSurfaceAccessor<float> sufDiv,
    CudaSurfaceAccessor<char> sufBound,
    unsigned int n
)
{
    // TODO: 计算速度的散度
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;
    if (sufBound.read(x, y, z) < 0) {
        sufDiv.write(0.f, x, y, z);
        return;
    }

    float vxp = sufVel.read<cudaBoundaryModeClamp>(x + 1, y, z).x;
    float vyp = sufVel.read<cudaBoundaryModeClamp>(x, y + 1, z).y;
    float vzp = sufVel.read<cudaBoundaryModeClamp>(x, y, z + 1).z;
    float vxn = sufVel.read<cudaBoundaryModeClamp>(x - 1, y, z).x;
    float vyn = sufVel.read<cudaBoundaryModeClamp>(x, y - 1, z).y;
    float vzn = sufVel.read<cudaBoundaryModeClamp>(x, y, z - 1).z;
    float div = (vxp - vxn + vyp - vyn + vzp - vzn) / 2.f;
    sufDiv.write(div, x, y, z);
}

__global__ void subgradient_kernel(
    CudaSurfaceAccessor<float> sufPre,
    CudaSurfaceAccessor<float4> sufVel,
    CudaSurfaceAccessor<char> sufBound,
    unsigned int n
)
{
    // TODO: 计算速度(速度减去压强的梯度)
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;
    if(sufBound.read(x, y, z) < 0)
        return;

    float pxn = sufPre.read<cudaBoundaryModeZero>(x - 1, y, z);
    float pyn = sufPre.read<cudaBoundaryModeZero>(x, y - 1, z);
    float pzn = sufPre.read<cudaBoundaryModeZero>(x, y, z - 1);
    float pxp = sufPre.read<cudaBoundaryModeZero>(x + 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeZero>(x, y + 1, z);
    float pzp = sufPre.read<cudaBoundaryModeZero>(x, y, z + 1);
    float4 vel = sufVel.read(x, y, z);
    vel.x -= (pxp - pxn) / 2.f;
    vel.y -= (pyp - pyn) / 2.f;
    vel.z -= (pzp - pzn) / 2.f;
    sufVel.write(vel, x, y, z);
}

__global__ void sumloss_kernel(
    CudaSurfaceAccessor<float> sufDiv, 
    float *sum, 
    unsigned int n
)
{
    // TODO: 计算sum的误差
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;

    float div = sufDiv.read(x, y, z);
    atomicAdd(sum, div * div);
}

template<int phase>
__global__ void rbgs_kernel(
    CudaSurfaceAccessor<float> sufPre,
    CudaSurfaceAccessor<float> sufDiv,
    unsigned int n
)
{
    // TODO: 红黑高斯
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;
    if((x + y + z) % 2 != phase)
        return;

    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float div = sufDiv.read(x, y, z);

    float preNext = (pxp + pxn + pyp + pyn + pzp + pzn - div) * (1.f / 6.f);
    sufPre.write(preNext, x, y, z);
}


__global__ void residual_kernel(
    CudaSurfaceAccessor<float> sufRes,
    CudaSurfaceAccessor<float> sufPre,
    CudaSurfaceAccessor<float> sufDiv,
    unsigned int n
)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;
    
    float pxp = sufPre.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float pxn = sufPre.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float pyp = sufPre.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float pyn = sufPre.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float pzp = sufPre.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float pzn = sufPre.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float pre = sufPre.read(x, y, z);
    float div = sufDiv.read(x, y, z);

    float res = pxp + pxn + pyp + pyn + pzp + pzn - pre * 6.f - div;
    sufRes.write(res, x, y, z);
}

__global__ void restrict_kernel(
    CudaSurfaceAccessor<float> sufPreNext, 
    CudaSurfaceAccessor<float> sufPre, 
    unsigned int n
)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;

    float ooo = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2, z * 2);
    float ioo = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2, z * 2);
    float oio = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2 + 1, z * 2);
    float iio = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2 + 1, z * 2);
    float ooi = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2, z * 2 + 1);
    float ioi = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2, z * 2 + 1);
    float oii = sufPre.read<cudaBoundaryModeClamp>(x * 2, y * 2 + 1, z * 2 + 1);
    float iii = sufPre.read<cudaBoundaryModeClamp>(x * 2 + 1, y * 2 + 1, z * 2 + 1);

    float preNext = (ooo + ioo + oio + iio + ooi + ioi + oii + iii);
    sufPreNext.write(preNext, x, y, z);
}

__global__ void fillzero_kernel(
    CudaSurfaceAccessor<float> sufPre,
    unsigned int n
)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;

    sufPre.write(0.f, x, y, z);
}

__global__ void prolongate_kernel(
    CudaSurfaceAccessor<float> sufPreNext,
    CudaSurfaceAccessor<float> sufPre,
    unsigned int n
)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n)
        return;

    float preDelta = sufPre.read(x, y, z);
#pragma unroll
    for(int dz = 0; dz < 2; dz++)
    {
#pragma unroll
        for(int dy = 0; dy < 2; dy++)
        {
#pragma unroll
            for(int dx = 0; dx < 2; dx++)
            {
                float preNext = sufPreNext.read<cudaBoundaryModeZero>(x * 2 + dx, y * 2 + dy, z * 2 + dz);
                preNext += preDelta;
                sufPre.write<cudaBoundaryModeZero>(preNext, x * 2 + dx, y * 2 + dy, z * 2 + dz);
            }
        }
    }
}

__global__ void heatup_kernel(
    CudaSurfaceAccessor<float4> sufVel, 
    CudaSurfaceAccessor<float> sufTmp, 
    CudaSurfaceAccessor<float> sufClr,
    CudaSurfaceAccessor<char> sufBound,
    float tmpAmbient,
    float heatRate,
    float clrRate,
    unsigned int n
)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;

    if(x >= n || y >= n || z >= n) return;
    if(sufBound.read(x, y, z) < 0) return;

    float4 vel = sufVel.read(x, y, z);
    float tmp = sufTmp.read(x, y, z);
    float clr = sufClr.read(x, y, z);
    vel.z += heatRate * (tmp - tmpAmbient);
    vel.z -= clr * clrRate;
    sufVel.write(vel, x, y, z);
}

__global__ void decay_kernel(CudaSurfaceAccessor<float> sufTmp, CudaSurfaceAccessor<float> sufTmpNext, CudaSurfaceAccessor<char> sufBound, float ambientRate, float decayRate, unsigned int n) {
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    if (x >= n || y >= n || z >= n) return;
    if (sufBound.read(x, y, z) < 0) return;

    float txp = sufTmp.read<cudaBoundaryModeClamp>(x + 1, y, z);
    float typ = sufTmp.read<cudaBoundaryModeClamp>(x, y + 1, z);
    float tzp = sufTmp.read<cudaBoundaryModeClamp>(x, y, z + 1);
    float txn = sufTmp.read<cudaBoundaryModeClamp>(x - 1, y, z);
    float tyn = sufTmp.read<cudaBoundaryModeClamp>(x, y - 1, z);
    float tzn = sufTmp.read<cudaBoundaryModeClamp>(x, y, z - 1);
    float tmpAvg = (txp + typ + tzp + txn + tyn + tzn) * (1 / 6.f);
    float tmpNext = sufTmp.read(x, y, z);
    tmpNext = (tmpNext * ambientRate + tmpAvg * (1.f - ambientRate)) * decayRate;
    sufTmpNext.write(tmpNext, x, y, z);
}