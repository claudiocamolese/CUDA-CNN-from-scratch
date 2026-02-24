#include <cuda_runtime.h>


/**
 * @brief Backward pass of Softmax with Cross-Entropy loss (fused)
 *
 * Computes the gradient with respect to the output logits:
 *
 * \f[
 * \frac{\partial L}{\partial z_c} = p_c - ground_truth
 * \f]
 *
 * where:
 * - \f$p_c\f$ is the predicted probability from softmax
 * - \f$y_c\f$ is the one-hot encoded ground-truth label
 *
 * This kernel assumes that the softmax has already been computed in the forward
 * pass and that `prob` contains normalized probabilities.
 *
 * @param[out] grad_logits  Gradient w.r.t. logits, shape (batch_size, num_classes)
 * @param[in]  prob         Softmax probabilities, shape (batch_size, num_classes)
 * @param[in]  labels       Ground-truth labels, shape (batch_size)
 * @param[in]  batch_size   Number of samples in the batch
 * @param[in]  num_classes  Number of classes
 */
__global__ void SoftmaxCrossEntropyBackward(float* grad_logits, const float* prob, const int* labels, int batch_size, int num_classes){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= batch_size * num_classes) return;
    int sampleIdx = idx / num_classes;
    int classes = idx % num_classes;
    int label = labels[sampleIdx];

    float p = prob[idx];
    float ground_truth = (classes == label) ? 1.0f : 0.0f;
    grad_logits[idx] = p - ground_truth;
}

/**
 * @brief Backward pass of a fully connected (dense) layer
 *
 * Computes gradients with respect to:
 * - weights:  gradW  (shape: flatten_size × num_classes)
 * - biases:   gradB  (shape: num_classes)
 *
 * Given:
 * - input tensor `in` of shape (batch_size, flatten_size)
 * - upstream gradient `grad_out` of shape (batch_size, num_classes)
 *
 * The gradients are computed as:
 *
 * \f[
 * \frac{\partial L}{\partial W_{k,c}} = \sum_{b} x_{b,k} \cdot \frac{\partial L}{\partial y_{b,c}}
 * \f]
 *
 * \f[
 * \frac{\partial L}{\partial b_c} = \sum_{b} \frac{\partial L}{\partial y_{b,c}}
 * \f]
 *
 * @param[in]  grad_out      Gradient from the next layer,
 *                           shape (batch_size, num_classes)
 * @param[in]  in            Input of the fully connected layer,
 *                           shape (batch_size, flatten_size)
 * @param[out] gradW         Gradient w.r.t. weights,
 *                           shape (flatten_size, num_classes)
 * @param[out] gradB         Gradient w.r.t. biases,
 *                           shape (num_classes)
 * @param[in]  batch_size    Number of samples in the batch
 * @param[in]  num_classes   Number of output neurons (classes)
 * @param[in]  flatten_size  Number of input features per sample
 *
 * @note
 * - Threads with idx < flatten_size * num_classes compute gradW
 * - Threads with idx >= flatten_size * num_classes compute gradB
 * - This kernel performs a reduction over the batch dimension
 * - No atomic operations are required since each parameter is handled
 *   by exactly one thread
 */
__global__ void FullyConnectedLayerBackward(const float* grad_out, const float* input_fc, float* gradW, float* gradB, int batch_size, int num_classes, int flatten_size){
    // Each thread handles one element in [FLATTEN_SIZE*NUM_CLASSES] for gradW
    // or up to NUM_CLASSES for gradB. We'll handle them separately:
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalW = flatten_size * num_classes;
    int totalParams = totalW + num_classes;

    if(idx >= totalParams) return;

    if(idx < totalW) {
        // gradW
        int featuresIdx = idx / num_classes;   // which input index
        int classes = idx % num_classes;   // which class
        float sumVal = 0.0f;
        
        for(int batch = 0; batch < batch_size; batch++){
            float grad_output = grad_out[batch * num_classes + classes];
            float input = input_fc[batch * flatten_size + featuresIdx];
            sumVal += input * grad_output;
        }
        gradW[idx] = sumVal;
    } else {
        // gradB
        int classes = idx - totalW;
        float sumVal = 0.0f;
        for(int batch = 0; batch < batch_size; batch++){
            sumVal += grad_out[batch * num_classes + classes];
        }
        gradB[classes] = sumVal;
    }
}


/**
 * @brief Computes gradient with respect to the input of a fully connected layer
 *
 * This kernel computes the backward pass for the input tensor of a fully
 * connected (dense) layer.
 *
 * Given:
 * - gradOut: gradient from the next layer, shape (batchSize, NUM_CLASSES)
 * - w:       weight matrix, shape (FLATTEN_SIZE, NUM_CLASSES)
 *
 * It computes:
 *
 * \f[
 * \frac{\partial L}{\partial x_{b,k}} =
 * \sum_{c=0}^{C-1}
 * \frac{\partial L}{\partial y_{b,c}} \cdot W_{k,c}
 * \f]
 *
 * where:
 * - b is the batch index
 * - k is the input feature index
 * - c is the class index
 *
 * @param[in]  gradOut   Gradient from the next layer,
 *                       shape (batchSize, NUM_CLASSES)
 * @param[in]  w         Fully connected layer weights,
 *                       shape (FLATTEN_SIZE, NUM_CLASSES)
 * @param[out] gradIn    Gradient w.r.t. input of the FC layer,
 *                       shape (batchSize, FLATTEN_SIZE)
 * @param[in]  batchSize Number of samples in the batch
 *
 * @note
 * - Each thread computes one element gradIn[b, k]
 * - The kernel performs a reduction over the NUM_CLASSES dimension
 * - This kernel assumes NUM_CLASSES and FLATTEN_SIZE are compile time constants
 */
__global__ void FullyConnectedBackward(const float* gradOut, const float* w, float* gradIn, int batch_size, int num_classes, int flatten_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * flatten_size;

    if (idx >= total)
        return;

    int batchIdx = idx / flatten_size;   // batch index
    int featuresIdx = idx % flatten_size;   // input feature index

    float sumVal = 0.0f;
    for (int classes = 0; classes < num_classes; classes++) {
        sumVal += gradOut[batchIdx * num_classes + classes] * w[featuresIdx * num_classes + classes];
    }

    gradIn[idx] = sumVal;
}

/**
 * @brief Unflattens gradients from a flattened tensor back to pooled feature maps
 *
 * This kernel reverses the flatten operation applied after max pooling.
 * It maps gradients from shape:
 *
 *   [batchSize, FLATTEN_SIZE]
 *
 * back to:
 *
 *   [batchSize, NUM_FILTERS, POOL_OUT_ROWS, POOL_OUT_COLS]
 *
 * @param[in]  gradFlat     Gradient of the flattened tensor
 * @param[out] gradPoolOut  Gradient w.r.t. pooled output feature maps
 * @param[in]  batchSize   Number of samples in the batch
 */
__global__ void FlattenBackward(const float* gradFlat, float* gradPoolOut, int batchSize, int num_filters, int pool_out_rows, int pool_out_cols, int flatten_size)
{
    // Inverse of flatten: [B, FLATTEN_SIZE] -> [B, NUM_FILTERS, 12, 12]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * num_filters * pool_out_rows * pool_out_cols;
    if(idx >= total) return;

    int batchIdx = idx / (num_filters * pool_out_rows * pool_out_cols);
    int residual = idx % (num_filters * pool_out_rows * pool_out_cols);
    int filter = residual / (pool_out_rows * pool_out_cols);
    int residual2 = residual % (pool_out_rows * pool_out_cols);
    int output_row = residual2 / pool_out_cols;
    int output_col = residual2 % pool_out_cols;

    int flatIdx = filter * (pool_out_rows * pool_out_cols) + output_row * (pool_out_cols) + output_col;
    gradPoolOut[idx] = gradFlat[batchIdx * flatten_size + flatIdx];
}




/**
 * @brief Backward pass for max pooling layer
 *
 * Propagates gradients from the pooled output back to the input feature map
 * using the stored max indices from the forward pass.
 *
 * @param[in]  gradOut   Gradient from the next layer
 * @param[out] gradIn    Gradient w.r.t. pooling input
 * @param[in]  maxIdx    Indices of maximum values selected during forward pass
 * @param[in]  batchSize Number of samples in the batch
 *
 * @note
 * - Uses atomicAdd because multiple pooled outputs may map to the same input
 */
__global__ void MaxPoolBackward(const float* gradOut, float* gradIn, const int* maxIdx, int batch_size, int num_filters, int pool_out_rows, int pool_out_cols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * num_filters * pool_out_rows * pool_out_cols;
    if(idx >= total) return;

    float val = gradOut[idx];
    int input_pos = maxIdx[idx];
    atomicAdd(&gradIn[input_pos], val);
}


/**
 * @brief Backward pass for ReLU activation
 *
 * Computes gradient through ReLU non-linearity:
 *
 *   gradIn = gradOut if x > 0, else 0
 *
 * @param[in]  gradOut Gradient from next layer
 * @param[in]  x       Input to ReLU during forward pass
 * @param[out] gradIn  Gradient w.r.t. ReLU input
 * @param[in]  n       Total number of elements
 */
__global__ void ReLUBackward(const float* gradOut, const float* x, float* gradIn, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        gradIn[i] = (x[i] > 0.0f) ? gradOut[i] : 0.0f;
    }
}

/**
 * @brief Computes gradients of convolution weights and biases
 *
 * Calculates:
 * - Gradient of convolution filters (gradW)
 * - Gradient of convolution biases (gradB)
 *
 * using the input tensor and gradients from convolution output.
 *
 * @param[in]  in           Input images, shape [batchSize, IMAGE_ROWS, IMAGE_COLS]
 * @param[in]  gradConvOut  Gradient from convolution output
 * @param[out] gradW        Gradient w.r.t. convolution weights
 * @param[out] gradB        Gradient w.r.t. convolution biases
 * @param[in]  batchSize   Number of samples in the batch
 *
 * @note
 * - Each thread computes either one weight or one bias gradient
 */
__global__ void ConvLayerBackward(
    const float* input,        // [B, inChannels, inRows, inCols]
    const float* gradOut,      // [B, outChannels, outRows, outCols]
    float* gradW,              // [outChannels, inChannels, filterSize, filterSize]
    float* gradB,              // [outChannels]
    int batchSize, int inChannels, int outChannels, int inRows, int inCols, int outRows, int outCols, int filterSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalParams = outChannels * inChannels * filterSize * filterSize + outChannels;
    if(idx >= totalParams) return;

    if(idx < outChannels * inChannels * filterSize * filterSize) {
        int residual = idx;
        int filter = residual / (inChannels * filterSize * filterSize);
        residual = residual % (inChannels * filterSize * filterSize);
        int channel = residual / (filterSize * filterSize);
        residual = residual % (filterSize * filterSize);
        int row_kernel = residual / filterSize;
        int col_kernel = residual % filterSize;

        float sumVal = 0.0f;
        
        for(int b = 0; b < batchSize; b++){
            for(int output_row = 0; output_row < outRows; output_row++){
                for(int output_col = 0; output_col < outCols; output_col++){
                    float grad_output = gradOut[b * (outChannels * outRows * outCols) + filter * (outRows * outCols) + output_row * outCols + output_col];
                    float inp = input[b * (inChannels * inRows * inCols) + channel * (inRows * inCols) + (output_row + row_kernel) * inCols + (output_col + col_kernel)];
                    sumVal += grad_output * inp;
                }
            }
        }
        gradW[idx] = sumVal;
    } else {
        int filter = idx - outChannels * inChannels * filterSize * filterSize;
        float sumVal = 0.0f;

        for(int batch = 0; batch < batchSize; batch++){
            for(int output_row = 0; output_row < outRows; output_row++){
                for(int output_col = 0; output_col < outCols; output_col++){
                    sumVal += gradOut[batch * (outChannels * outRows * outCols) + filter * (outRows * outCols)+ output_row * outCols + output_col];
                }
            }
        }
        gradB[filter] = sumVal;
    }
}


/**
 * @brief Computes gradient with respect to convolution input
 *
 * Backpropagates gradients from convolution output to the input image
 * by convolving the output gradients with flipped filters.
 *
 * @param[in]  gradConvOut Gradient from convolution output
 * @param[in]  w           Convolution filters
 * @param[out] gradIn      Gradient w.r.t. convolution input
 * @param[in]  batchSize  Number of samples in the batch
 */
__global__ void ConvBackward(
    const float* gradOut,  // [B, outChannels, outRows, outCols]
    const float* weight,   // [outChannels, inChannels, filterSize, filterSize]
    float* gradIn,         // [B, inChannels, inRows, inCols]
    int batchSize, int inChannels, int outChannels, int inRows, int inCols, int outRows, int outCols, int filterSize)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * inChannels * inRows * inCols;
    if(idx >= total) return;

    int batchIdx = idx / (inChannels * inRows * inCols);
    int residual = idx % (inChannels * inRows * inCols);
    int channel = residual / (inRows * inCols);
    residual = residual % (inRows * inCols);
    int row = residual / inCols;
    int col = residual % inCols;

    float sumVal = 0.0f;
    for(int filter = 0; filter < outChannels; filter++){
        for(int row_kernel = 0; row_kernel < filterSize; row_kernel++){
            for(int col_kernel = 0; col_kernel < filterSize; col_kernel++){
                int output_row = row - row_kernel;
                int output_col = col - col_kernel;
                
                if(output_row >= 0 && output_row < outRows && output_col >= 0 && output_col < outCols){
                    float out_grad = gradOut[batchIdx * (outChannels * outRows * outCols) + filter * (outRows * outCols) + output_row * outCols + output_col];
                    float out_w = weight[filter * (inChannels* filterSize * filterSize) + channel * (filterSize * filterSize) + row_kernel * filterSize + col_kernel];
                    sumVal += out_grad * out_w;
                }
            }
        }
    }
    gradIn[idx] = sumVal;
}


/**
 * @brief Performs SGD parameter update
 *
 * Updates parameters using Stochastic Gradient Descent:
 *
 *   param = param - lr * grad
 *
 * @param[in,out] param Parameter array to update
 * @param[in]     grad  Gradient array
 * @param[in]     lr    Learning rate
 * @param[in]     n     Number of parameters
 */
__global__ void SGDBackward(float* param, const float* grad, float lr, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
        param[i] -= lr * grad[i];
    }
}

