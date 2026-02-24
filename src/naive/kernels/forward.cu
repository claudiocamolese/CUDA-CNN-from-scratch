#include <cuda_runtime.h>


/**
 * @brief Generic forward pass for convolution (supports multiple channels)
 * 
 * Each thread computes a single element of the convolution output.
 * Supports batching, multiple input channels and multiple output filters.
 * 
 * @param input_tensor         Pointer to input tensor [batchSize, inChannels, inRows, inCols]
 * @param weights              Pointer to filter weights [outChannels, inChannels, filterSize, filterSize]
 * @param bias                 Pointer to bias for each filter [outChannels]
 * @param output_tensor        Pointer to output tensor [batchSize, outChannels, outRows, outCols]
 * @param batchSize            Number of images in the batch
 * @param inChannels           Number of input channels
 * @param outChannels          Number of output filters / channels
 * @param inRows               Height of the input
 * @param inCols               Width of the input
 * @param filterSize           Size of the filter (assumes square filters)
 */
__global__ void ConvForward(const float* input_tensor, const float* weights, const float* bias, float* output_tensor, int batchSize, int inChannels, int outChannels, int inRows, int inCols, int filterSize){
    // dimension of the output without padding
    int outRows = inRows - filterSize + 1;
    int outCols = inCols - filterSize + 1;

    // One thread for each output element
    int total = batchSize * outChannels * outRows * outCols;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if(index >= total) return;

    /* 
        Decode the thread idx in 4D coordinate [B, C, H, W].
        To undestand which element to compute, first extract the image idx in the batch (batchIdx), then we need to extract
        the filter idx and the position in the image.
    */
    int batchIdx = index / (outChannels * outRows * outCols);  // index of the image in the batch
    int residual = index % (outChannels * outRows * outCols);
    int filter = residual / (outRows * outCols);  // index of the filter
    int residual2 = residual % (outRows * outCols);
    int output_row = residual2 / outCols; // index in the row dimension
    int output_col = residual2 % outCols; // index in the col dimension

    float val = 0.0f;
    for(int channel= 0; channel < inChannels; channel++){
        for(int kernel_row= 0; kernel_row < filterSize; kernel_row++){
            for(int kernel_col= 0; kernel_col< filterSize; kernel_col++){

                // index position with respect to the kernel position 
                int inRow = output_row + kernel_row;
                int inCol = output_col + kernel_col;

                float input = input_tensor[batchIdx * (inChannels*inRows*inCols) + channel * (inRows*inCols) + inRow * inCols + inCol];
                float kernel_weight = weights[filter * (inChannels*filterSize*filterSize) + channel * (filterSize*filterSize) + kernel_row * filterSize + kernel_col];
                
                // Compute the scalar product between the input window and the kernel 
                val += input * kernel_weight;
            }
        }
    }
    val += bias[filter];
    output_tensor[index] = val;
}



/**
 * @brief ReLU kernel. Each thread index controls the value of its element and set it to zero if negative 
 * 
 * @param input_tensor         Pointer to the input tensor  [batchSize, inChannels, inRows, inCols]
 * @param n                    Number of elements in the input tensor
 */
__global__ void ReLUForward(float* input_tensor, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        if(input_tensor[idx]<0.0f){
            input_tensor[idx]= 0.0f;
        }
    }
}


/**
 * @brief Max Pooling Forward Kernel. Take a poolsize x poolsize window of the input, saves the maximum 
 * value and its position (this will be used for the backword process).
 *
 * @param input         Pointer to input tensor [batch, channels, inRows, inCols]
 * @param output        Pointer to output tensor [batch, channels, outRows, outCols]
 * @param maxIdx        Pointer to array storing max positions (for backward pass)
 * @param batchSize     Number of samples in batch
 * @param channels      Number of feature maps / channels
 * @param inRows        Input height
 * @param inCols        Input width
 * @param poolSize      Pooling window size (assumes square)
 */
__global__ void MaxPoolForward(const float* input, float* output, int* maxIdx, int batchSize, int channels, int inRows, int inCols, int poolSize)
{
    int outRows = inRows / poolSize;
    int outCols = inCols / poolSize;
    int total = batchSize * channels * outRows * outCols; // total number of output elements

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total) return;

    // Decode the thread idx in 4D coordinate [B, C, H, W].
    int batchIdx = idx / (channels * outRows * outCols);
    int residual = idx % (channels * outRows * outCols);
    int filter = residual / (outRows * outCols);
    int residual2 = residual % (outRows * outCols);
    int output_row = residual2 / outCols;
    int output_col = residual2 % outCols;

    // Every output is a related to the input window [row_start : row_start + poolSize, col_start : col_start + poolSize]
    int row_start = output_row * poolSize;
    int col_start = output_col * poolSize;

    float maxVal = -1e30f;
    int maxPos = 0;

    for(int i= 0; i < poolSize; i++){
        for(int j= 0; j < poolSize; j++){
            
            int rowIdx = row_start + i;
            int colIdx = col_start + j;

            if(rowIdx < inRows && colIdx < inCols){

                int inIdx = batchIdx * (channels*inRows*inCols) + filter * (inRows*inCols) + rowIdx * inCols + colIdx;
                float val = input[inIdx];
                
                // Save the max value and its position
                if(val > maxVal){
                    maxVal = val;
                    maxPos = inIdx;
                }
            }
        }
    }

    output[idx] = maxVal;
    maxIdx[idx] = maxPos;
}


/**
 * @brief Flatten a 4D tensor from [batch, channels, height, width] to 2D [batch, channels*height*width].
 *
 * @param input_tensor          Pointer to the input tensor (shape: batchSize * inChannels * inRows * inCols)
 * @param output_tensor         Pointer to the flattened output tensor (shape: batchSize * inChannels*inRows*inCols)
 * @param batchSize             Number of samples in the batch
 * @param inChannels            Number of channels in the input tensor
 * @param inRows                Number of rows per channel in the input tensor
 * @param inCols                Number of columns per channel in the input tensor
 */
__global__ void FlattenForward(const float* input_tensor, float* output_tensor, int batchSize, int inChannels, int inRows, int inCols)
{
    int total = batchSize * inChannels * inRows * inCols;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= total) return;

    int batchIdx = idx / (inChannels * inRows * inCols);
    int residual = idx % (inChannels * inRows * inCols);
    int filter = residual / (inRows * inCols);
    int residual2 = residual % (inRows * inCols);
    int output_row = residual2 / inCols;
    int output_col = residual2 % inCols;

    int flatIdx = filter * (inRows * inCols) + output_row * inCols + output_col; // allows to pass from [filters][rows][cols] to flatIdx
    output_tensor[batchIdx * (inChannels * inRows * inCols) + flatIdx] = input_tensor[idx];
}



/**
 * @brief Fully connected layer forward pass.
 *
 * Each thread computes one output neuron for a specific batch element and class.
 * The kernel performs a dot product between the flattened input vector and the
 * corresponding column of the weight matrix, then adds the bias.
 *
 * @param input_tensor   Pointer to the flattened input tensor
 *                       (shape: [batch_size, flatten_size])
 * @param w              Pointer to the weight matrix
 *                       (shape: [flatten_size, num_classes])
 * @param b              Pointer to the bias vector
 *                       (shape: [num_classes])
 * @param output_tensor  Pointer to the output tensor
 *                       (shape: [batch_size, num_classes])
 * @param batch_size     Number of samples in the batch
 * @param num_classes    Number of output classes (neurons)
 * @param flatten_size   Number of input features per sample
 */
__global__ void FullyConnectedForward(const float* input_tensor, const float* w, const float* b, float* output_tensor, int batch_size, int num_classes, int flattent_size){
    
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch_size * num_classes;
    if (idx >= total) return;

    int batchIdx = idx / num_classes; 
    int classIdx = idx % num_classes;

    // compute the dot product of the fully connected layer
    float sumVal = 0.0f;
    for(int k= 0; k < flattent_size; k++){
        sumVal += input_tensor[batchIdx * flattent_size + k] * w[k * num_classes + classIdx];
    }

    // add the bias 
    sumVal += b[classIdx];

    output_tensor[idx] = sumVal;
}



/**
 * @brief Softmax + Cross-Entropy forward pass.
 *
 * This kernel computes the softmax probabilities and the cross entropy loss
 * for each sample in the batch. One thread processes one sample.
 *
 * For numerical stability, the maximum logit is subtracted before computing
 * the exponentials.
 *
 * Outputs:
 * - A probability distribution over classes for each sample
 * - The scalar cross entropy loss for each sample
 *
 * @param logits     Pointer to input logits (shape: [batch_size, num_classes])
 * @param labels     Pointer to ground-truth labels (shape: [batch_size])
 * @param outLoss    Pointer to output loss values (shape: [batch_size])
 * @param PredProb    Pointer to output softmax probabilities (shape: [batch_size, num_classes])
 * @param batchSize  Number of samples in the batch
 * @param num_classes Number of classes
 */
__global__ void SoftmaxCrossEntropyForward(const float* logits, const int* labels, float* outLoss, float* PredProb, int batchSize, int num_classes)
    {

        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= batchSize) return;

        // Find the max in order to avoid NaN / Inf
        float thr = -1e30f;
        for(int classIdx = 0; classIdx < num_classes; classIdx++){
            float val = logits[i * num_classes + classIdx];
            if(val > thr) thr = val;
        }

        // Compute the softmax 
        float ExpSum = 0.0f;
        for(int classIdx = 0; classIdx < num_classes; classIdx++){
            float Exp = expf(logits[i *num_classes + classIdx] - thr);
            ExpSum += Exp;
        }

        // Compute the predicted label
        int labelIdx = labels[i];
        float lossVal = 0.0f;
        for(int classIdx = 0; classIdx < num_classes; classIdx++){
            
            float Exp = expf(logits[i * num_classes + classIdx] - thr);
            float prob = Exp / ExpSum;
            
            PredProb[ i *num_classes + classIdx] = prob;
            
            if(classIdx == labelIdx){
                lossVal = -logf(prob + 1e-10f); // avoid numerical instability
            }
        }

        outLoss[i] = lossVal;
    }