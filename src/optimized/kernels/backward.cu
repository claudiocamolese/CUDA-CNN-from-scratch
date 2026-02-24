#include <cuda_runtime.h>

/**
 * @brief Backward pass of the fused Softmax + Cross-Entropy loss.
 *
 * Computes the gradient of the loss with respect to the input logits.
 *
 * Tensor shapes:
 *   prob        → [batchSize, numClasses]
 *   gradLogits  → [batchSize, numClasses]
 *   labels      → [batchSize]
 *
 * Parallelization strategy:
 *   - One thread computes exactly one gradient element (b, c).
 *   - Linear index mapping is used to convert a 1D thread index
 *     into (sampleIdx, classIdx).
 *
 * Optimization techniques used:
 *
 *   One-Thread-Per-Element Mapping:
 *      Each CUDA thread computes one output gradient element.
 *      This eliminates race conditions and avoids synchronization.
 *
 *   Memory Coalescing:
 *      prob and gradLogits are stored in contiguous row-major layout,
 *      enabling coalesced global memory access across threads.
 *
 *   Implicit One-Hot Encoding:
 *      The ground-truth vector is not explicitly stored as a one-hot tensor.
 *      Instead, it is reconstructed using a conditional check,
 *      reducing memory usage and bandwidth.
 *
 * No shared memory or inter-thread communication is required.
 *
 * @param gradLogits  Output gradient dL/dlogits.
 *                    Flattened array of size batchSize * numClasses.
 *
 * @param prob        Softmax probabilities from forward pass.
 *                    Flattened array of size batchSize * numClasses.
 *
 * @param labels      Ground-truth class indices.
 *                    Array of size batchSize.
 *
 * @param batchSize   Number of samples in the mini-batch.
 *
 * @param numClasses  Number of output classes.
 */
__global__ void SoftmaxCrossBackward(float* gradLogits, const float* prob, const int* labels, int batchSize, int numClasses)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batchSize * numClasses) return;

    int sampleIdx = idx / numClasses;
    int classes = idx % numClasses;
    int label = labels[sampleIdx];

    float gt = (classes == label) ? 1.0f : 0.0f;
    gradLogits[idx] = prob[idx] - gt;
}


/**
 * @brief Backward pass for Fully Connected (Dense) layer parameters.
 *
 * Computes gradients of:
 *
 *   1. Weight matrix W  → gradW
 *   2. Bias vector  b   → gradB
 *
 * Tensor shapes:
 *
 *   in        → [batchSize, inFeatures]
 *   gradOut   → [batchSize, outFeatures]
 *   gradW     → [inFeatures, outFeatures]
 *   gradB     → [outFeatures]
 *
 * Parallelization strategy:
 *
 *   - Parameters (weights + biases) are flattened into a single
 *     linear index space.
 *   - Each thread block is responsible for a TILE_SIZE group
 *     of parameters.
 *   - Threads cooperate to compute partial sums over the batch dimension.
 *
 * Optimization techniques used:
 *
 *   Parameter Space Parallelization:
 *      The kernel parallelizes across parameter indices instead
 *      of batch samples, allowing independent computation of each
 *      weight and bias gradient.
 *
 *   Strided Batch Accumulation:
 *      Threads accumulate partial sums across the batch using
 *      a stride equal to blockDim.x, improving parallel workload
 *      distribution.
 *
 *   Shared Memory Reduction:
 *      Partial sums are stored in shared memory and reduced using
 *      a tree-based parallel reduction to obtain the final gradient
 *      for each parameter.
 *
 *   Fused Weight and Bias Gradient Computation:
 *      Both weight and bias gradients are computed in a single kernel,
 *      reducing kernel launch overhead.
 *
 *
 * @param gradOut     Upstream gradient from next layer.
 *                    Shape: [batchSize * outFeatures]
 *
 * @param in          Input activations to the FC layer.
 *                    Shape: [batchSize * inFeatures]
 *
 * @param gradW       Output gradient for weights.
 *                    Shape: [inFeatures * outFeatures]
 *
 * @param gradB       Output gradient for bias.
 *                    Shape: [outFeatures]
 *
 * @param batchSize   Number of samples in the mini-batch.
 *
 * @param inFeatures  Number of input features.
 *
 * @param outFeatures Number of output neurons.
 */
__global__ void FCParamBackward(const float* gradOut, const float* in, float* gradW, float* gradB, int batchSize, int inFeatures, int outFeatures)
{
    const int TILE_SIZE = 16;
    int totalW = inFeatures * outFeatures;
    int totalParams = totalW + outFeatures;
    int paramIdx = blockIdx.x * TILE_SIZE + threadIdx.y;
    if(paramIdx >= totalParams) return;

    float sum = 0.0f;
    bool isWeight = (paramIdx < totalW);
    int featureIdx = 0;
    int outputIdx = 0;
    
    if(isWeight){
        featureIdx = paramIdx / outFeatures;
        outputIdx = paramIdx % outFeatures;
    } else {
        outputIdx = paramIdx - totalW;
    }

    int stride = blockDim.x;
    
    for(int b = threadIdx.x; b < batchSize; b += stride){
        if(isWeight) sum += in[b * inFeatures + featureIdx] * gradOut[b * outFeatures + outputIdx];
        else sum += gradOut[b * outFeatures + outputIdx];
    }

    __shared__ float sdata[32][TILE_SIZE];
    sdata[threadIdx.x][threadIdx.y] = sum;
    __syncthreads();

    // tree reduction
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(threadIdx.x < s) sdata[threadIdx.x][threadIdx.y] += sdata[threadIdx.x + s][threadIdx.y];
        __syncthreads();
    }

    if(threadIdx.x == 0){
        if(isWeight) gradW[paramIdx] = sdata[0][threadIdx.y];
        else gradB[paramIdx - totalW] = sdata[0][threadIdx.y];
    }
}

/**
 * @brief Backward pass for Fully Connected layer input gradients.
 *
 * Computes the gradient of the loss with respect to the input activations
 * of a Fully Connected (Dense) layer.
 *
 * Tensor shapes:
 *
 *   gradOut  → [batchSize, outFeatures]
 *   W        → [inFeatures, outFeatures]
 *   gradIn   → [batchSize, inFeatures]
 *
 * Parallelization strategy:
 *
 *   - One thread computes exactly one output element gradIn[b, k].
 *   - A linear index is mapped to (batch index, input feature index).
 *
 * Optimization techniques used:
 *
 *   One-Thread-Per-Output-Element Mapping:
 *      Each thread independently computes one (b, k) entry of gradIn.
 *      No synchronization or shared memory is required.
 *
 *   Loop Unrolling:
 *      '#pragma unroll' hints the compiler to unroll the loop over
 *      output features to reduce loop overhead and improve ILP
 *      (Instruction-Level Parallelism).
 * 
 *  Fused Multiply-Add (FMA):
 *      Uses 'fmaf(a, b, c)' which computes (a * b + c) in a single
 *      instruction, improving both performance and numerical precision.
 *
 * Memory layout is row-major for all tensors.
 *
 * @param gradOut     Upstream gradient from next layer.
 *                    Shape: [batchSize * outFeatures]
 *
 * @param w           Weight matrix of the FC layer.
 *                    Shape: [inFeatures * outFeatures]
 *
 * @param gradIn      Output gradient with respect to inputs.
 *                    Shape: [batchSize * inFeatures]
 *
 * @param batchSize   Number of samples in the mini-batch.
 *
 * @param inFeatures  Number of input features.
 *
 * @param outFeatures Number of output neurons.
 */
__global__ void FCGradBackward(const float* gradOut, const float* w, float* gradIn, int batchSize, int inFeatures, int outFeatures)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= batchSize * inFeatures) return;

    int batchIdx = idx / inFeatures;
    int featureIdx = idx % inFeatures;
    float sumVal = 0.0f;
    int baseGrad = batchIdx * outFeatures;
    int baseW = featureIdx * outFeatures;

    #pragma unroll
    for(int feature = 0; feature < outFeatures; feature++)
        sumVal = fmaf(gradOut[baseGrad + feature], w[baseW + feature], sumVal);

    gradIn[idx] = sumVal;
}

/**
 * @brief Backward pass of the ReLU activation function.
 *
 * Computes the gradient of the loss with respect to the input tensor
 * of a ReLU activation layer.
 *
 * Tensor shapes:
 *
 *   x        → [n]
 *   gradOut  → [n]
 *   gradIn   → [n]
 *
 * Parallelization strategy:
 *
 *   - One thread computes exactly one element i.
 *   - Linear indexing is used over the flattened tensor.
 *
 * Optimization techniques used:
 *
 *   One-Thread-Per-Element Mapping:
 *      Each thread independently computes one gradient value.
 *      No synchronization or shared memory is required.
 *
 *   Read-Only Cache Usage (__ldg):
 *      The input tensor x is loaded using __ldg(), enabling
 *      the use of the GPU read-only data cache for improved
 *      memory access efficiency when x is not modified.
 *
 * Memory layout is assumed to be contiguous (row-major flattening
 * for multi-dimensional tensors).
 *
 * @param gradOut  Upstream gradient from next layer.
 *                 Size: n
 *
 * @param x        Input tensor from forward pass (pre-activation values).
 *                 Size: n
 *
 * @param gradIn   Output gradient with respect to x.
 *                 Size: n
 *
 * @param n        Total number of elements in the tensor.
 */
__global__ void ReLUBackward(const float* gradOut, const float* x, float* gradIn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        float x_val = __ldg(&x[i]);
        gradIn[i] = gradOut[i] * (x_val > 0.0f);
    }
}


/**
 * @brief Backward pass of a 2D Max Pooling layer (no overlap).
 *
 * Computes the gradient of the loss with respect to the input of a
 * max-pooling layer by routing each upstream gradient value only to
 * the position that produced the maximum during the forward pass.
 *
 * Tensor shapes:
 *
 *   convOut     → [batchSize, inChannels, inH, inW]
 *   gradFlat    → [batchSize, inChannels, outH, outW]
 *   gradConvOut → [batchSize, inChannels, inH, inW]
 *
 * Parallelization strategy:
 *
 *   - One thread computes the backward contribution of one pooled
 *     output element (b, c, rOut, cOut).
 *   - Each thread scans its corresponding pooling window to locate
 *     the maximum value and writes the upstream gradient to that
 *     position in gradConvOut.
 *
 * @param convOut      Output of the convolution layer before pooling.
 *                     Shape: [batchSize, inChannels, inH, inW]
 *
 * @param gradFlat     Upstream gradient from next layer (flattened pooling output).
 *                     Shape: [batchSize, inChannels, outH, outW]
 *
 * @param gradConvOut  Output gradient w.r.t. convOut (input of pooling).
 *                     Shape: [batchSize, inChannels, inH, inW]
 *
 * @param batchSize    Number of samples in the batch.
 *
 * @param inChannels   Number of feature maps (channels).
 *
 * @param inH          Height of the convolution output (before pooling).
 *
 * @param inW          Width of the convolution output (before pooling).
 *
 * @param poolSize     Spatial size of the pooling window (kernel size).
 *
 * @param outH         Height of the pooling output.
 *
 * @param outW         Width of the pooling output.
 */
__global__ void MaxPollBackward(const float* convOut, const float* gradFlat, float* gradConvOut, int batchSize, int inChannels, int inH, int inW, int poolSize, int outH, int outW)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * inChannels * outH * outW;
    if(index >= total) return;

    int batchIdx = index / (inChannels * outH * outW);
    int residual = index % (inChannels * outH * outW);
    int channel = residual / (outH * outW);
    int residual2 = residual % (outH * outW);
    int output_row = residual2 / outW;
    int output_col = residual2 % outW;

    int input_row = output_row * poolSize;
    int input_col = output_col * poolSize;

    int convStride = inW;
    int base = batchIdx * (inChannels * inH * inW) + channel * inH * inW + input_row * convStride + input_col;

    float maxVal = convOut[base];
    int maxIdx = 0;
    
    for(int pool_row = 0; pool_row < poolSize; pool_row++){
        for(int pool_col = 0; pool_col < poolSize; pool_col++){
            
            int row = input_row + pool_row;
            int col = input_col + pool_col;
            
            if(row < inH && col < inW){
                float val = convOut[batchIdx * (inChannels * inH * inW) + channel * inH * inW + row * inW + col];
                
                if(val > maxVal){
                    maxVal = val;
                    maxIdx = pool_row * poolSize + pool_col;
                }
            }
        }
    }

    int max_row = input_row + maxIdx / poolSize;
    int max_col = input_col + maxIdx % poolSize;
    gradConvOut[batchIdx * (inChannels * inH * inW) + channel * inH * inW + max_row * inW + max_col] = gradFlat[index];
}



/**
 * @brief Backward pass for convolution layer parameters (weights and biases).
 *
 * Computes gradients with respect to:
 *   - Convolution weights (gradW)
 *   - Convolution biases (gradB)
 *
 *
 * Tensor shapes:
 *
 *   in           → [batchSize, inChannels, inH, inW]
 *   gradConvOut  → [batchSize, outChannels, outH, outW]
 *   gradW        → [outChannels, inChannels, kH, kW]
 *   gradB        → [outChannels]
 *
 * Parallelization strategy:
 *
 *   - Each CUDA block processes one parameter (weight or bias).
 *   - Threads inside a warp collaboratively accumulate partial sums
 *     over batch and spatial output positions.
 *
 * Optimization techniques used:
 *
 *   Warp-Level Reduction (__shfl_down_sync):
 *      Partial sums are reduced using warp shuffle instructions,
 *      avoiding shared memory and synchronization barriers.
 *
 *   Strided Loop Over Batch × Spatial Domain:
 *      Each thread accumulates contributions spaced by warpSize,
 *      ensuring load balancing across threads.
 *
 * @param in            Input tensor to convolution layer.
 * @param gradConvOut   Gradient w.r.t. convolution output.
 * @param gradW         Output gradient w.r.t. convolution weights.
 * @param gradB         Output gradient w.r.t. convolution biases.
 * @param batchSize     Number of samples in batch.
 * @param inChannels    Number of input channels.
 * @param inH           Input height.
 * @param inW           Input width.
 * @param outChannels   Number of convolution filters.
 * @param kH            Kernel height.
 * @param kW            Kernel width.
 * @param outH          Output height.
 * @param outW          Output width.
 */
__global__ void ConvWeightBackward(const float* in, const float* gradConvOut, float* gradW, float* gradB, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW)
{
    int totalW = outChannels * inChannels * kH * kW;
    int totalParams = totalW + outChannels;
    int paramIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(paramIdx >= totalParams) return;

    float PartialSum = 0.0f;
    int warpSize = 32;

    if(paramIdx < totalW){
        int filter = paramIdx / (inChannels * kH * kW);
        int residual = paramIdx % (inChannels * kH * kW);
        int channel = residual / (kH * kW);
        int kernel_row = (residual % (kH * kW)) / kW;
        int kernel_col = residual % kW;
        int N = batchSize * outH * outW;

        for(int i = threadIdx.x; i < N; i += warpSize){
            int batchIdx = i / (outH * outW);
            int res = i % (outH * outW);
            int output_row = res / outW;
            int output_col = res % outW;

            float grad = gradConvOut[batchIdx * (outChannels * outH * outW) + filter * (outH * outW) + output_row * outW + output_col];
            float input = in[batchIdx * (inChannels * inH * inW) + channel * (inH * inW) + (output_row + kernel_row) * inW + (output_col + kernel_col)];
            
            PartialSum += input * grad;
        }
    } else {
        int filter = paramIdx - totalW;
        int N = batchSize * outH * outW;

        for(int i = threadIdx.x; i < N; i += warpSize){
            
            int batchIdx = i / (outH * outW);
            int res = i % (outH * outW);
            int output_row = res / outW;
            int output_col = res % outW;

            PartialSum += gradConvOut[batchIdx * (outChannels * outH * outW) + filter * (outH * outW) + output_row * outW + output_col];
        }
    }

    unsigned int mask = 0xffffffff;
    for(int thr = warpSize/2; thr > 0; thr /= 2)
        PartialSum += __shfl_down_sync(mask, PartialSum, thr);

    if(threadIdx.x == 0){
        if(paramIdx < totalW) gradW[paramIdx] = PartialSum;
        else gradB[paramIdx - totalW] = PartialSum;
    }
}



/**
 * @brief Backward pass for convolution with respect to input (gradIn).
 *
 * Computes the gradient of the loss with respect to the input of a convolutional layer.
 * For each input element, it sums over all contributions from the upstream gradients
 * weighted by the corresponding kernel weights.

 * Tensor shapes:
 *
 *   gradOut   → [batchSize, outChannels, outH, outW]   (upstream gradient)
 *   w         → [outChannels, inChannels, kH, kW]      (convolution weights)
 *   gradIn    → [batchSize, inChannels, inH, inW]      (gradient w.r.t input)
 *
 * Parallelization strategy:
 *
 *   - Each thread computes one element of gradIn: indexed by (b, c, r, col)
 *   - blockIdx.x  → batch index
 *   - threadIdx.z → input channel
 *   - threadIdx.y → row of input
 *   - threadIdx.x → column of input
 *
 * Optimization techniques used:
 *
 *   - None specific beyond one-thread-per-input-element mapping
 *     (no shared memory, no warp-level reduction)
 *   - FMA (fmaf) used for numerically stable and efficient multiply-add
 *
 * @param gradOut      Upstream gradient tensor (∂L/∂Y).
 * @param w            Convolution weights tensor.
 * @param gradIn       Output gradient w.r.t input.
 * @param batchSize    Number of samples in batch.
 * @param inChannels   Number of input channels.
 * @param inH          Input height.
 * @param inW          Input width.
 * @param outChannels  Number of output channels (filters).
 * @param kH           Kernel height.
 * @param kW           Kernel width.
 * @param outH         Output height.
 * @param outW         Output width.
 */
__global__ void ConvLayerBackward(const float* gradOut, const float* w, float* gradIn, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW)
{
    int batchIdx = blockIdx.x;
    if(batchIdx >= batchSize) return;

    int channel = threadIdx.z;   // input channel
    int row = threadIdx.y;
    int col = threadIdx.x;

    if(channel >= inChannels || row >= inH || col >= inW) return;

    float sum = 0.0f;

    for(int filter = 0; filter < outChannels; filter++){
        for(int kernel_row = 0; kernel_row < kH; kernel_row++){
            for(int kernel_col = 0; kernel_col < kW; kernel_col++){
                
                int output_row = row - kernel_row;
                int output_col = col - kernel_col;
                
                if(output_row >= 0 && output_row < outH && output_col >= 0 && output_col < outW){
                    
                    float grad = gradOut[batchIdx * (outChannels * outH * outW) + filter * (outH * outW) + output_row * outW + output_col];
                    float wv = w[filter * (inChannels * kH * kW) + channel * (kH * kW) + kernel_row * kW + kernel_col];
                    
                    sum = fmaf(grad, wv, sum);
                }
            }
        }
    }

    gradIn[batchIdx * (inChannels * inH * inW) + channel * (inH * inW) + row * inW + col] = sum;
}


/**
 * @brief Stochastic Gradient Descent (SGD) parameter update kernel.
 *
 * Updates model parameters directly on the GPU using the gradients
 * computed during the backward pass.

 * Parallelization strategy:
 *
 *   - Each thread updates a single parameter `param[i]`.
 *   - `blockIdx.x` and `threadIdx.x` determine the global index `i`.
 *
 * Optimization techniques used:
 *
 *   - One-thread-per-parameter mapping: avoids thread conflicts and synchronization.
 *   - In-place update directly on the GPU (no host-device transfer needed).
 *
 * @param param  Array of parameters to update.
 * @param grad   Array of corresponding gradients.
 * @param lr     Learning rate scalar controlling update step size.
 * @param n      Total number of parameters to update.
 * @return __global__  CUDA kernel executed in parallel on the GPU.
 */
__global__ void SGDBackward(float* param, const float* grad, float lr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i< n) param[i] -= lr * grad[i];
}
