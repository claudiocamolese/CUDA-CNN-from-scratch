#include <cuda_runtime.h>

/**
 * @brief Performs the forward pass of a Convolution layer with Bias addition
 *        and ReLU activation fused into a single CUDA kernel.
 *
 * Supports configurable stride and zero-padding.
 *
 * Memory layout (row-major):
 *  - in  : [batchSize, inChannels, inH, inW]
 *  - w   : [outChannels, inChannels, kH, kW]
 *  - b   : [outChannels]
 *  - out : [batchSize, outChannels, outH, outW]
 *
 * Optimization Techniques Used:
 *
 * Kernel Fusion (Conv + Bias + ReLU):
 *    Convolution, bias addition, and ReLU activation are executed in a single kernel to:
 *      - Reduce global memory traffic
 *      - Avoid intermediate tensor writes
 *      - Improve cache locality
 *      - Reduce kernel launch overhead
 *
 * 3D Grid Mapping:
 *    - blockIdx.x/y → spatial coordinates (outW, outH)
 *    - blockIdx.z   → encodes batch index and filter index
 *    This maximizes parallel coverage of output space.
 *
 * Boundary Checks for Padding:
 *    Conditional checks avoid illegal memory accesses
 *    while implicitly implementing zero-padding.
 *
 *
 * @param in            Pointer to input tensor in device memory
 * @param w             Pointer to convolution filters in device memory
 * @param b             Pointer to bias vector in device memory
 * @param out           Pointer to output tensor in device memory
 * @param batchSize     Number of samples in the batch (B)
 * @param inChannels    Number of input channels (Cin)
 * @param inH           Input height
 * @param inW           Input width
 * @param outChannels   Number of output channels / filters (Cout)
 * @param kH            Kernel height
 * @param kW            Kernel width
 * @param outH          Output height
 * @param outW          Output width
 * @param stride        Convolution stride (default = 1)
 * @param padding       Zero-padding size applied to input (default = 0)
 *
 * @return void
 */
__global__ void ConvReluForward(const float* input, const float* w, const float* b, float* out, int batchSize, int inChannels, int inH, int inW, int outChannels, int kH, int kW, int outH, int outW, int stride=1, int padding=0)
{
    // compute global coordinates
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batchIdx = blockIdx.z / outChannels;
    int filterIdx = blockIdx.z % outChannels;

    if (out_x >= outW || out_y >= outH || batchIdx >= batchSize) return;

    float result = 0.0f;

    for (int channel = 0; channel < inChannels; channel++){
        for (int row_kernel = 0; row_kernel < kH; row_kernel++){
            for (int col_kernel = 0; col_kernel < kW; col_kernel++){
                int in_x = out_x * stride + col_kernel - padding;
                int in_y = out_y * stride + row_kernel - padding;
                
                if(in_x >= 0 && in_x < inW && in_y >= 0 && in_y < inH){
                    float inVal = input[batchIdx * (inChannels * inH * inW) + channel * (inH * inW) + in_y * inW + in_x];
                    float wVal = w[filterIdx * (inChannels * kH * kW) + channel * (kH * kW) + row_kernel * kW + col_kernel];
                    result += inVal * wVal;
                }
            }
        }
    }

    result += b[filterIdx];

    // ReLU passage
    if(result < 0.0f) result = 0.0f;

    out[batchIdx * (outChannels * outH * outW) + filterIdx * (outH * outW) + out_y * outW + out_x] = result;
}




/**
 * @brief Performs Max Pooling followed by implicit Flatten in a single CUDA kernel.
 *
 * The output tensor is written in flattened linear memory layout:
 * index ∈ [0, batchSize * inChannels * outH * outW)
 *
 * Memory layout (row-major):
 *  - in  : [batchSize, inChannels, inH, inW]
 *  - out : [batchSize, inChannels, outH, outW] (flattened storage)
 *
 * Assumes non-overlapping pooling regions (stride = poolSize).
 *
 * Optimization Techniques Used:
 *
 * Kernel Fusion (MaxPool + Flatten):
 *    Pooling and flattening are combined into a single kernel.
 *    This:
 *      - Avoids an additional reshape kernel
 *      - Reduces global memory traffic
 *      - Eliminates intermediate writes
 *
 * Linear Index Decomposition:
 *    A single linear thread index is decomposed into:
 *      - batch index (b)
 *      - channel index (c)
 *      - pooled spatial coordinates (output_row, cOut)
 *    This simplifies parallel mapping and guarantees full coverage.
 *
 * Bounds Checking:
 *    Ensures safe memory access when input dimensions
 *    are not perfectly divisible by poolSize.
 *
 *
 * @param in         Pointer to input tensor in device memory
 * @param out        Pointer to pooled + flattened output tensor
 * @param batchSize  Number of samples in the batch (B)
 * @param inChannels Number of input channels (Cin)
 * @param inH        Input height
 * @param inW        Input width
 * @param poolSize   Pooling window size (assumed square)
 * @param outH       Output height after pooling
 * @param outW       Output width after pooling
 *
 * @return void
 */
__global__ void MaxPoolFlattenForward(const float* in, float* out, int batchSize, int inChannels, int inH, int inW, int poolSize, int outH, int outW)
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

    //int convStride = inW;
    float maxVal = -1e30f;
    for(int pool_row = 0; pool_row < poolSize; pool_row++){
        for(int pool_col = 0; pool_col < poolSize; pool_col++){
            
            int rowIdx = input_row + pool_row;
            int colIdx = input_col + pool_col;
            if(rowIdx < inH && colIdx < inW){
                float val = in[batchIdx * (inChannels * inH * inW) + channel * (inH * inW) + rowIdx * inW + colIdx];
                if(val > maxVal) maxVal = val;
            }
        }
    }

    out[index] = maxVal;
}



/**
 * @brief Performs the forward pass of a Fully Connected (Dense) layer
 *        using tiled matrix multiplication with shared memory.
 *
 * Each thread computes a single output element:
 * out[row, col]
 *
 * Optimization Techniques Used:
 *
 * Tiled Matrix Multiplication:
 *    The input matrix and weight matrix are divided into TILE_SIZE × TILE_SIZE submatrices (tiles).
 *    This reduces redundant global memory accesses.
 *
 * Shared Memory Usage:
 *    Submatrices of 'in' and 'w' are loaded into shared memory (A and B) before computation.
 *    This:
 *      - Exploits fast on-chip memory
 *      - Improves memory bandwidth efficiency
 *      - Increases data reuse within a thread block
 *
 * Cooperative Thread Loading:
 *    Threads within a block collaboratively load tiles into shared memory, maximizing memory coalescing.
 *
 * Loop Tiling Over Input Features:
 *    The dot product is computed incrementally over tiles (numTiles), enabling scalability for large feature sizes.
 *
 *  Loop Unrolling:
 *    '#pragma unroll' reduces loop overhead and may improve instruction-level parallelism.
 *
 * One-Thread-Per-Output-Element Strategy:
 *    Each thread computes exactly one output element, eliminating race conditions and synchronization complexity beyond tile synchronization.
 *
 * Boundary Handling:
 *    Conditional checks prevent out-of-bounds accesses when dimensions are not multiples of TILE_SIZE.
 *
 *
 * @param in            Pointer to input tensor in device memory
 * @param w             Pointer to weight matrix in device memory
 * @param b             Pointer to bias vector in device memory
 * @param out           Pointer to output tensor in device memory
 * @param batchSize     Number of samples in the batch (B)
 * @param inFeatures    Number of input features (Fin)
 * @param outFeatures   Number of output features (Fout)
 *
 * @return void
 */
__global__ void FCForward(const float* in, const float* w, const float* b, float* out, int batchSize, int inFeatures, int outFeatures)
{
    const int TILE_SIZE = 16;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y; // batch
    int col = blockIdx.x * TILE_SIZE + threadIdx.x; // output feature

    __shared__ float A[TILE_SIZE][TILE_SIZE];
    __shared__ float B[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    int numTiles = (inFeatures + TILE_SIZE - 1)/TILE_SIZE;

    for(int tile = 0; tile < numTiles; tile++){
        int tiled_col = tile * TILE_SIZE + threadIdx.x;
        
        if(row < batchSize && tiled_col < inFeatures)
            A[threadIdx.y][threadIdx.x] = in[row * inFeatures + tiled_col];
        else
            A[threadIdx.y][threadIdx.x] = 0.0f;

        int tiled_row = tile * TILE_SIZE + threadIdx.y;
        
        if(tiled_row < inFeatures && col < outFeatures)
            B[threadIdx.y][threadIdx.x] = w[tiled_row * outFeatures + col];
        else
            B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();
        #pragma unroll
        for(int tile = 0; tile < TILE_SIZE; tile++)
            sum += A[threadIdx.y][tile] * B[tile][threadIdx.x];
        __syncthreads();
    }

    if(row < batchSize && col < outFeatures)
        out[row * outFeatures + col] = sum + b[col];
}





/**
 * @brief Computes Softmax probabilities and Cross-Entropy loss (fused)
 *        for a batch of logits.
 *
 * Outputs:
 *  - outProb : softmax probabilities [batchSize, numClasses]
 *  - outLoss : per-sample cross-entropy loss [batchSize]
 *
 * Memory layout (row-major):
 *  - logits  : [batchSize, numClasses]
 *  - labels  : [batchSize]
 *  - outProb : [batchSize, numClasses]
 *  - outLoss : [batchSize]
 *
 * Optimization Techniques Used:
 *
 * Kernel Fusion (Softmax + Cross-Entropy):
 *    Softmax computation and cross-entropy loss are performed
 *    in a single kernel. This:
 *      - Eliminates intermediate global memory writes
 *      - Reduces memory bandwidth usage
 *      - Reduces kernel launch overhead
 *
 * One-Thread-Per-Sample Strategy:
 *    Each thread processes one batch element.
 *    This avoids inter-thread synchronization and ensures
 *    independence across samples.
 *
 * Numerical Stability Trick:
 *    The maximum logit is subtracted before exponentiation:
 *        exp(logit - maxLogit)
 *    This prevents overflow and improves floating-point stability.
 *
 * Fast Approximate Exponential:
 *    Uses __expf() instead of expf() for faster computation on GPU, trading slight precision loss for performance.
 *
 * In-Place Temporary Storage:
 *    The outProb buffer temporarily stores unnormalized exponentials before normalization, avoiding extra buffers.
 *
 * Log Epsilon Protection:
 *    A small constant (1e-10f) is added inside log() to prevent log(0) and improve numerical robustness.
 *
 *
 * @param logits      Pointer to input logits in device memory
 * @param labels      Pointer to ground-truth class indices
 * @param outLoss     Pointer to per-sample loss output
 * @param outProb     Pointer to softmax probabilities output
 * @param batchSize   Number of samples in the batch (B)
 * @param numClasses  Number of classes (C)
 *
 * @return void
 */
__global__ void SoftmaxCrossForward(const float* logits, const int* labels, float* outLoss, float* outProb, int batchSize, int numClasses)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= batchSize) return;

    int offset = i * numClasses;

    // max logit (stabilità numerica)
    float maxLogit = -1e30f;
    for(int classes = 0; classes < numClasses; classes++){
        float valthr = logits[offset + classes];
        if(valthr > maxLogit) maxLogit = valthr;
    }

    // exp + sum
    float sumExp = 0.0f;
    for(int classes = 0; classes < numClasses; classes++){
        float exp = __expf(logits[offset + classes] - maxLogit);
        outProb[offset + classes] = exp;  
        sumExp += exp;
    }

    // normalizzazione
    for(int classes = 0; classes < numClasses; classes++)
        outProb[offset + classes] /= sumExp;

    // cross-entropy
    int label = labels[i];
    float prob = outProb[offset + label];
    outLoss[i] = -logf(prob + 1e-10f);
}
