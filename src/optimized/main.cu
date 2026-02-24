/*
    This code implements a fully custom CUDA-based Convolutional Neural Network
    optimized for GPU execution. No external deep learning libraries (cuDNN, cuBLAS)
    are used; all kernels are written from scratch.

    Network architecture:
        (Conv + ReLU) →
        (Conv + ReLU) →
        MaxPool →
        Flatten →
        Fully Connected →
        Softmax

    ------------------------------------------------------------------------
    Optimization Techniques Used
    ------------------------------------------------------------------------

    Kernel Fusion
       - Convolution + ReLU are fused into a single kernel (ConvReluForward)
         to reduce global memory traffic and eliminate intermediate writes.
       - Softmax qnd Cross-Entropy loss are fused to avoid storing intermediate
         softmax outputs and reduce memory bandwidth usage.

    Tiled Matrix Multiplication (Fully Connected Layer)
       - The FC layer uses a tiled GEMM implementation.
       - Shared memory tiles (16x16) are used to improve memory reuse.
       - Reduces redundant global memory accesses.
       - Loop unrolling is applied to increase instruction-level parallelism.

    Shared Memory Usage
       - Used in convolution and FC kernels to cache tiles of data.
       - Reduces global memory latency and improves bandwidth utilization.

    Triple Buffering with CUDA Streams
       - Three independent host-device buffers are used.
       - cudaMemcpyAsync + cudaStream allow overlap of:
            • CPU data preparation
            • Host-to-device transfers
            • GPU computation
       - Minimizes GPU idle time.

    Pinned (Page-Locked) Host Memory
       - cudaMallocHost is used for faster and asynchronous transfers.

    Asynchronous Execution Pipeline
       - Data transfer and kernel execution occur in separate CUDA streams.
       - Synchronization is only performed when strictly necessary.

    Memory Coalescing Strategy
       - Data is stored in contiguous row-major layout.
       - Thread indexing ensures coalesced global memory accesses.

    One-Thread-Per-Output Mapping
       - Each thread computes one output element in convolution,
         pooling, and FC layers.
       - Eliminates race conditions and simplifies synchronization.

    Numerical Stability in Softmax
       - Maximum logit subtraction before exponentiation.
       - Small epsilon added inside log to avoid log(0).
       - Fast intrinsic __expf() used for performance.

    Lightweight SGD Update Kernel
        - Parameter updates performed directly on device.
        - Avoids unnecessary host-device transfers.
*/

#include "config.h"
#include "utils/load_data.h"
#include "utils/printing.h"
#include "kernels/forward.h"
#include "kernels/backward.h"
#include <cuda_runtime.h>
#include <iostream>


/**
 * @brief Checks the result of a CUDA runtime call and exits on error.
 *
 * @param err The cudaError_t returned by a CUDA API call.
 * @param msg Optional custom message describing the context of the call.
 */

inline void CudaCheck(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s (%s)\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[]) {
    
    srand(21); // for reproducibility

    /*
        Allocate space for train and test set, images are 28x28  
    */
    float* host_TrainImages = (float*)malloc(TRAIN_IMAGES * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    float* h_testImages  = (float*)malloc(TEST_IMAGES  * IMAGE_ROWS * IMAGE_COLS * sizeof(float));
    int*   host_TrainLabels = (int*)malloc(TRAIN_IMAGES * sizeof(int));
    int*   h_testLabels  = (int*)malloc(TEST_IMAGES  * sizeof(int));

    if (!host_TrainImages || !h_testImages || !host_TrainLabels || !h_testLabels) {
        printf("Memory allocation failed\n");
        return 1;
    }

    /* 
        Load the dataset (MNIST or FASHION), MNIST is the default one.
    */
    std::string dataset = "mnist"; // default

    if (argc == 2) {
        dataset = argv[1];
    }

    std::string train_images = "../datasets/" + dataset + "/train-images.idx3-ubyte";
    std::string test_images ="../datasets/" + dataset + "/t10k-images.idx3-ubyte";
    std::string train_labels = "../datasets/" + dataset + "/train-labels.idx1-ubyte";
    std::string test_labels = "../datasets/" + dataset + "/t10k-labels.idx1-ubyte";

    load_images(train_images.c_str(), host_TrainImages, TRAIN_IMAGES);
    load_images(test_images.c_str(), h_testImages, TEST_IMAGES);
    load_labels(train_labels.c_str(), host_TrainLabels, TRAIN_IMAGES);
    load_labels(test_labels.c_str(), h_testLabels, TEST_IMAGES);

    printf("Network --> (CONV(1, %d, %d), RELU) + (CONV(%d, %d, %d), RELU) + MaxPool(%d) + Flatten + FC + SoftMax\n", FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FIRST_OUTPUT_CHANNELS, SECOND_OUTPUT_CHANNELS, FILTER_SIZE, POOL_SIZE);
    printf("Epochs: %d, Learning rate: %.2f, Batch size: %d\n", EPOCHS, LEARNING_RATE, BATCH_SIZE);
    printf("Block Size: %d\n", BLOCK_SIZE);


    // --------------------
    //  Initialization
    // --------------------
    
    // Convolutional layers
    float *d_conv1W, *d_conv1B, *d_conv1Out;
    float *d_conv2W, *d_conv2B, *d_conv2Out;
    
    // Pooling
    float *d_poolOut;
    int   *d_poolIdx;
    
    // Flatten + Fully Connected
    float *d_flat, *d_fcW, *d_fcB, *d_fcOut;
    
    // Prob, loss, gradients
    float *d_prob, *d_loss;
    float *d_grad_fcW, *d_grad_fcB;
    float *d_grad_fcOut, *d_grad_flat;
    float *d_grad_conv1Out, *d_grad_conv1W, *d_grad_conv1B;
    float *d_grad_conv2Out, *d_grad_conv2W, *d_grad_conv2B;
    float *d_grad_poolOut, *d_grad_in;

    /* 
    Triple-buffered batch data

    Three separate buffers are allocated on both the CPU (pinned memory)
    and the GPU for batch data (images and labels). The goal is to allow asynchronous 
    data transfers from CPU to GPU using CUDA streams, so that while the GPU processes 
    one batch, the CPU can prepare the next batch and copy it into an available buffer, 
    avoiding idle time.

    - device_TrainImages[i] and device_Labels[i]: pointers to the buffers in device (GPU) memory
    - host_PinnedImages[i] and host_PinnedLabels[i]: pointers to the buffers in pinned CPU memory
    - stream[i]: CUDA stream associated with each buffer, allowing asynchronous memcpy 
      and kernel execution in parallel without blocking the GPU.

    NUM_BUFFERS = 3 (triple buffering) is used to maximize overlap between data transfers 
    and computation, improving the training pipeline efficiency. Higher NUM_BUFFERS is not useful.
*/

    size_t imageBytesPerBatch = BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float);
    float* device_TrainImages[NUM_BUFFERS];
    int*   device_Labels[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaMalloc(&device_TrainImages[i], imageBytesPerBatch));
        CudaCheck(cudaMalloc(&device_Labels[i], BATCH_SIZE * sizeof(int)));
    }

    cudaStream_t stream[NUM_BUFFERS];
    float* host_PinnedImages[NUM_BUFFERS];
    int*   host_PinnedLabels[NUM_BUFFERS];

    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaStreamCreate(&stream[i]));
        CudaCheck(cudaMallocHost((void**)&host_PinnedImages[i], imageBytesPerBatch));
        CudaCheck(cudaMallocHost((void**)&host_PinnedLabels[i], BATCH_SIZE * sizeof(int)));
    }

    // Dimensions
    int conv1W_size = FIRST_OUTPUT_CHANNELS  * FIRST_INPUT_CHANNELS  * FILTER_SIZE * FILTER_SIZE;
    int conv2W_size = SECOND_OUTPUT_CHANNELS * SECOND_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE;   

    // -------------------------
    // CUDA memory allocation
    // -------------------------
    CudaCheck(cudaMalloc(&d_conv1W, conv1W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv1Out, BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_conv2W, conv2W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_conv2Out, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_poolOut, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_poolIdx, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(int)));

    CudaCheck(cudaMalloc(&d_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));
    CudaCheck(cudaMalloc(&d_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_fcB, NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_fcOut, BATCH_SIZE * NUM_CLASSES * sizeof(float)));

    CudaCheck(cudaMalloc(&d_prob, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_loss, BATCH_SIZE * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcOut, BATCH_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_flat, BATCH_SIZE * FLATTEN_SIZE * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_fcB, NUM_CLASSES * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_conv1Out, BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv1W, conv1W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_conv2Out, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv2W, conv2W_size * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float)));

    CudaCheck(cudaMalloc(&d_grad_poolOut, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS * sizeof(float)));
    CudaCheck(cudaMalloc(&d_grad_in, BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS * sizeof(float)));

    
    /* ------------------------------------------------------------
        He weights initialization on host and copied to device
        ------------------------------------------------------------
        
        He initialization, is better suited for layers that use ReLU 
        activation functions since it mitigates the exploding gradient issue.
        The layer weights are initialized in the range [-limit, +limit] while the bias are initialized to 0.
        The limit is W ~ U(- sqrt(6/n), sqrt(6/n)) where n is the number of input neurons to the layer.
    */

    // Initilization of the first Conv layer (Conv1) 
    float* h_conv1W = (float*)malloc(conv1W_size * sizeof(float)); // layer weights
    float* h_conv1B = (float*)malloc(FIRST_OUTPUT_CHANNELS * sizeof(float)); // bias weights
    int fan_in_conv1 = FIRST_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE; // total input number for each neuron, use to scale the weights
    auto he_rand_conv1 = [fan_in_conv1]() {
        float limit = sqrtf(6.0f / fan_in_conv1); // He initilization
        return limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f); // casual number in the range [-limit, + limit]
    };
    for (int i = 0; i < conv1W_size; i++) h_conv1W[i] = he_rand_conv1();
    for (int i = 0; i < FIRST_OUTPUT_CHANNELS; i++) h_conv1B[i] = 0.0f;
    CudaCheck(cudaMemcpy(d_conv1W, h_conv1W, conv1W_size * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(d_conv1B, h_conv1B, FIRST_OUTPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv1W); 
    free(h_conv1B);

    // Initilization of the second Conv layer (Conv2)
    float* h_conv2W = (float*)malloc(conv2W_size * sizeof(float));
    float* h_conv2B = (float*)malloc(SECOND_OUTPUT_CHANNELS * sizeof(float));
    int fan_in_conv2 = SECOND_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE;
    auto he_rand_conv2 = [fan_in_conv2]() {
        float limit = sqrtf(6.0f / fan_in_conv2);
        return limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    };
    for (int i = 0; i < conv2W_size; i++) h_conv2W[i] = he_rand_conv2();
    for (int i = 0; i < SECOND_OUTPUT_CHANNELS; i++) h_conv2B[i] = 0.0f;
    CudaCheck(cudaMemcpy(d_conv2W, h_conv2W, conv2W_size * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(d_conv2B, h_conv2B, SECOND_OUTPUT_CHANNELS * sizeof(float), cudaMemcpyHostToDevice));
    free(h_conv2W); 
    free(h_conv2B);

    // Initilization of the FullyConnected layer (FC)
    float* h_fcW = (float*)malloc(FLATTEN_SIZE * NUM_CLASSES * sizeof(float));
    float* h_fcB = (float*)malloc(NUM_CLASSES * sizeof(float));
    int fan_in_fc = FLATTEN_SIZE;
    auto he_rand_fc = [fan_in_fc]() {
        float limit = sqrtf(6.0f / fan_in_fc);
        return limit * (2.0f * ((float)rand() / RAND_MAX) - 1.0f);
    };
    for (int i = 0; i < FLATTEN_SIZE * NUM_CLASSES; i++) h_fcW[i] = he_rand_fc();
    for (int i = 0; i < NUM_CLASSES; i++) h_fcB[i] = 0.0f;
    CudaCheck(cudaMemcpy(d_fcW, h_fcW, FLATTEN_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    CudaCheck(cudaMemcpy(d_fcB, h_fcB, NUM_CLASSES * sizeof(float), cudaMemcpyHostToDevice));
    free(h_fcW); 
    free(h_fcB);

    // Record time for training and testing
    cudaEvent_t startEvent, stopEvent;
    CudaCheck(cudaEventCreate(&startEvent));
    CudaCheck(cudaEventCreate(&stopEvent));
    CudaCheck(cudaEventRecord(startEvent, 0));

    printf("Starting training for %d epochs:\n", EPOCHS);

    for (int epoch= 0; epoch< EPOCHS; epoch++){
        
        float epoch_loss = 0.0f;
        int uploadedBuffers = (NUM_BATCHES < NUM_BUFFERS) ? NUM_BATCHES : NUM_BUFFERS; // correctly manage last batches

        for(int buffer= 0; buffer < uploadedBuffers ; buffer++){
            
            int BatchSize = buffer * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS);
            
            memcpy(host_PinnedImages[buffer], &host_TrainImages[BatchSize], imageBytesPerBatch);
            memcpy(host_PinnedLabels[buffer], &host_TrainLabels[buffer * BATCH_SIZE], BATCH_SIZE * sizeof(int));
            
            // Copy data without blocking the CPU, using the straming CUDA
            CudaCheck(cudaMemcpyAsync(device_TrainImages[buffer], host_PinnedImages[buffer], imageBytesPerBatch, cudaMemcpyHostToDevice, stream[buffer]));
            CudaCheck(cudaMemcpyAsync(device_Labels[buffer], host_PinnedLabels[buffer], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream[buffer]));
        }
        
        for(int batch = 0; batch < NUM_BATCHES; batch++){

            print_progress(batch + 1, NUM_BATCHES, epoch, EPOCHS);

            int CurrentBuffer = batch % NUM_BUFFERS;
            int NextBuffer = (batch + 1) % NUM_BUFFERS;

            // Before using the current buffer data, must wait that the asynchronize batch copy on the GPU is complete.
            CudaCheck(cudaStreamSynchronize(stream[CurrentBuffer]));

            // CONV1 + RELU
            {
            dim3 blockDim(16, 16, 1);
            dim3 gridDim((CONV1_OUT_COLS + 16 - 1)/16, (CONV1_OUT_ROWS + 16 - 1)/16, BATCH_SIZE * FIRST_OUTPUT_CHANNELS);

            size_t SharedMemorySize = (16 + FILTER_SIZE - 1) * (16 + FILTER_SIZE - 1) * sizeof(float); // to reduce global memory acess

            ConvReluForward<<<gridDim, blockDim, SharedMemorySize, stream[CurrentBuffer]>>>(
                device_TrainImages[CurrentBuffer], d_conv1W, d_conv1B, d_conv1Out, 
                BATCH_SIZE, FIRST_INPUT_CHANNELS, IMAGE_ROWS, IMAGE_COLS,
                FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV1_OUT_ROWS, CONV1_OUT_COLS);
            }

            // Conv2 + ReLU
            {
            dim3 blockDim(16, 16, 1);
            dim3 gridDim((CONV2_OUT_COLS + 16 - 1)/16, (CONV2_OUT_ROWS + 16 - 1)/16, BATCH_SIZE * SECOND_OUTPUT_CHANNELS);

            size_t SharedMemorySize = (16 + FILTER_SIZE - 1) * (16 + FILTER_SIZE - 1) * sizeof(float);

            ConvReluForward<<<gridDim, blockDim, SharedMemorySize, stream[CurrentBuffer]>>>(
                d_conv1Out, d_conv2W, d_conv2B, d_conv2Out, 
                BATCH_SIZE, SECOND_INPUT_CHANNELS, CONV1_OUT_ROWS, CONV1_OUT_COLS,
                SECOND_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV2_OUT_ROWS, CONV2_OUT_COLS);
            }


            // Max Pooling + Flatten
            {
            int totalFlatten = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
            int gridDim = (totalFlatten + BLOCK_SIZE - 1) / BLOCK_SIZE;

            MaxPoolFlattenForward<<<gridDim, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                d_conv2Out, d_flat,
                BATCH_SIZE,SECOND_OUTPUT_CHANNELS, CONV2_OUT_ROWS, CONV2_OUT_COLS, POOL_SIZE, POOL_OUT_ROWS, POOL_OUT_COLS);
            }

            // Fully Connected forward (tiled GEMM)
            {
            dim3 blockDim(16, 16, 1);
            dim3 gridDim((NUM_CLASSES + 16 - 1)/16, (BATCH_SIZE + 16 - 1)/16);
            int sharedMemBytes = 2 * (16*16) * sizeof(float);

            FCForward<<<gridDim, blockDim, sharedMemBytes, stream[CurrentBuffer]>>>(
                d_flat, d_fcW, d_fcB, d_fcOut,
                BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);
            }

            // -------------------------------
            // Softmax + Cross Entropy Loss
            // -------------------------------
            {
            int gridDim = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            size_t sharedMemBytes = NUM_CLASSES * sizeof(float); // per expVals

            SoftmaxCrossForward<<<gridDim, BLOCK_SIZE, sharedMemBytes, stream[CurrentBuffer]>>>(
                d_fcOut, device_Labels[CurrentBuffer], d_loss, d_prob,
                BATCH_SIZE, NUM_CLASSES);
            }

            // ---------------------------------------------------------------------------
            // Softmax + Cross-Entropy backward  -> d_grad_fcOut
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * NUM_CLASSES;
                int gridDim = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                SoftmaxCrossBackward<<<gridDim, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_fcOut, d_prob, device_Labels[CurrentBuffer],
                    BATCH_SIZE, NUM_CLASSES);
            }

            // ---------------------------------------------------------------------------
            // Fully Connected backward (grad W, B)
            // ---------------------------------------------------------------------------
            {
                int totalParams = FLATTEN_SIZE * NUM_CLASSES + NUM_CLASSES;
                dim3 blockDim(32, 16);
                dim3 gridDim((totalParams + 15) / 16);

                FCParamBackward<<<gridDim, blockDim, 0, stream[CurrentBuffer]>>>(
                    d_grad_fcOut,d_flat, d_grad_fcW, d_grad_fcB,
                    BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);
            }

            // ---------------------------------------------------------------------------
            // Fully Connected backward (grad input -> d_grad_flat)
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * FLATTEN_SIZE;
                int gridDim = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                FCGradBackward<<<gridDim, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_fcOut, d_fcW, d_grad_flat,
                    BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);
            }

            // ---------------------------------------------------------------------------
            // MaxPool backward -> d_grad_conv2Out
            // ---------------------------------------------------------------------------
            {
                cudaMemsetAsync(d_grad_conv2Out, 0, BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS * sizeof(float), stream[CurrentBuffer]);

                int total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
                int gridDim = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                MaxPollBackward<<<gridDim, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_conv2Out, d_grad_flat,d_grad_conv2Out,
                    BATCH_SIZE, SECOND_OUTPUT_CHANNELS, CONV2_OUT_ROWS, CONV2_OUT_COLS, POOL_SIZE, POOL_OUT_ROWS, POOL_OUT_COLS);
            }

            // ---------------------------------------------------------------------------
            // 5) ReLU backward (Conv2)
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * CONV2_OUT_ROWS * CONV2_OUT_COLS;
                int gridDim = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                ReLUBackward<<<gridDim, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_conv2Out, d_conv2Out, d_grad_conv2Out, total);
            }

            // ---------------------------------------------------------------------------
            // Conv2 backward (grad W, B)
            // ---------------------------------------------------------------------------
            {
                int totalParams = SECOND_OUTPUT_CHANNELS * SECOND_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE + SECOND_OUTPUT_CHANNELS;
                int gridDim = (totalParams + 31) / 32;

                ConvWeightBackward<<<gridDim, 32, 0, stream[CurrentBuffer]>>>(
                    d_conv1Out, d_grad_conv2Out, d_grad_conv2W, d_grad_conv2B,
                    BATCH_SIZE, SECOND_INPUT_CHANNELS, CONV1_OUT_ROWS, CONV1_OUT_COLS, SECOND_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV2_OUT_ROWS, CONV2_OUT_COLS);
            }

            // ---------------------------------------------------------------------------
            // Conv2 backward (grad input -> d_grad_conv1Out)
            // ---------------------------------------------------------------------------
            {
                dim3 blockDim(16, 16, FIRST_OUTPUT_CHANNELS);
                dim3 gridDim(BATCH_SIZE);

                ConvLayerBackward<<<gridDim, blockDim, 0, stream[CurrentBuffer]>>>(
                    d_grad_conv2Out, d_conv2W, d_grad_conv1Out,
                    BATCH_SIZE, FIRST_OUTPUT_CHANNELS, CONV1_OUT_ROWS, CONV1_OUT_COLS, SECOND_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV2_OUT_ROWS, CONV2_OUT_COLS);
            }

            // ---------------------------------------------------------------------------
            // ReLU backward (Conv1)
            // ---------------------------------------------------------------------------
            {
                int total = BATCH_SIZE * FIRST_OUTPUT_CHANNELS * CONV1_OUT_ROWS * CONV1_OUT_COLS;
                int gridDim = (total + BLOCK_SIZE - 1) / BLOCK_SIZE;

                ReLUBackward<<<gridDim, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(
                    d_grad_conv1Out, d_conv1Out, d_grad_conv1Out, total);
            }

            // ---------------------------------------------------------------------------
            // Conv1 backward (grad W, B)
            // ---------------------------------------------------------------------------
            {
                int totalParams = FIRST_OUTPUT_CHANNELS * FIRST_INPUT_CHANNELS * FILTER_SIZE * FILTER_SIZE + FIRST_OUTPUT_CHANNELS;
                int gridDim = (totalParams + 31) / 32;

                ConvWeightBackward<<<gridDim, 32, 0, stream[CurrentBuffer]>>>(
                    device_TrainImages[CurrentBuffer],d_grad_conv1Out, d_grad_conv1W, d_grad_conv1B,
                    BATCH_SIZE, FIRST_INPUT_CHANNELS, IMAGE_ROWS, IMAGE_COLS, FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV1_OUT_ROWS, CONV1_OUT_COLS);
            }

            // ---------------------------------------------------------------------------
            // SGD update (Conv1, Conv2, FC)
            // ---------------------------------------------------------------------------
            {
                auto SGD = [&](float* p, float* g, int n){
                    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
                    SGDBackward<<<grid, BLOCK_SIZE, 0, stream[CurrentBuffer]>>>(p, g, LEARNING_RATE, n);
                };

                // Conv1
                SGD(d_conv1W, d_grad_conv1W, conv1W_size);
                SGD(d_conv1B, d_grad_conv1B, FIRST_OUTPUT_CHANNELS);

                // Conv2
                SGD(d_conv2W, d_grad_conv2W, conv2W_size);
                SGD(d_conv2B, d_grad_conv2B, SECOND_OUTPUT_CHANNELS);

                // FC
                SGD(d_fcW, d_grad_fcW, FLATTEN_SIZE * NUM_CLASSES);
                SGD(d_fcB, d_grad_fcB, NUM_CLASSES);
            }

            if(batch + 1 < NUM_BATCHES){

                int nextOffset = (batch + 1) * BATCH_SIZE * (IMAGE_ROWS * IMAGE_COLS);
                
                memcpy(host_PinnedImages[NextBuffer], &host_TrainImages[nextOffset], imageBytesPerBatch);
                memcpy(host_PinnedLabels[NextBuffer], &host_TrainLabels[(batch + 1) * BATCH_SIZE], BATCH_SIZE * sizeof(int));
                
                CudaCheck(cudaMemcpyAsync(device_TrainImages[NextBuffer], host_PinnedImages[NextBuffer], imageBytesPerBatch, cudaMemcpyHostToDevice, stream[NextBuffer]));
                CudaCheck(cudaMemcpyAsync(device_Labels[NextBuffer], host_PinnedLabels[NextBuffer], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice, stream[NextBuffer]));
            }
 
            CudaCheck(cudaStreamSynchronize(stream[CurrentBuffer]));
            float h_loss[BATCH_SIZE];
            CudaCheck(cudaMemcpy(h_loss, d_loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
            float batchLoss = 0.f;
            
            for (int i = 0; i < BATCH_SIZE; i++){
                batchLoss += h_loss[i];
            }
            
            epoch_loss += batchLoss / BATCH_SIZE;
        }
        
        std::cout << "\n";
        epoch_loss /= NUM_BATCHES;
        
        printf("Epoch [%d/%d], avg loss=%.6f\n", epoch + 1, EPOCHS, epoch_loss);
    }
 
    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaStreamSynchronize(stream[i]));
    }
    
    CudaCheck(cudaEventRecord(stopEvent, 0));
    CudaCheck(cudaEventSynchronize(stopEvent));
    float elapsedTime = 0.f;
    CudaCheck(cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent));

    printf("Time for training %d epochs is: %.2f s\n", EPOCHS, elapsedTime/1000);

    // -----------------
    // Testing Phase
    // -----------------

    cudaEvent_t startEvent_test, stopEvent_test;
    CudaCheck(cudaEventCreate(&startEvent_test));
    CudaCheck(cudaEventCreate(&stopEvent_test));
    CudaCheck(cudaEventRecord(startEvent_test, 0));
    {
        dim3 blockDim(16, 16);
        dim3 gridDim((NUM_CLASSES + 15) / 16, (BATCH_SIZE + 15) / 16);

        int fcSharedBytes = 2 * 16 * 16 * sizeof(float);
        int correct = 0;
        int testBatches = TEST_IMAGES / BATCH_SIZE;
        float* h_prob = (float*)malloc(BATCH_SIZE * NUM_CLASSES * sizeof(float));

        for (int batch = 0; batch < testBatches; batch++)
        {
            // -------------------------------------------------
            // Copy input images + labels
            // -------------------------------------------------
            CudaCheck(cudaMemcpy(device_TrainImages[0], &h_testImages[batch * BATCH_SIZE * IMAGE_ROWS * IMAGE_COLS], imageBytesPerBatch, cudaMemcpyHostToDevice));
            CudaCheck(cudaMemcpy(device_Labels[0], &h_testLabels[batch * BATCH_SIZE], BATCH_SIZE * sizeof(int), cudaMemcpyHostToDevice));

            // -------------------------------------------------
            // Forward pass
            // -------------------------------------------------

            // ---------- Conv1 + ReLU ----------
            dim3 blockConv1(16, 16);
            dim3 gridConv1((CONV1_OUT_COLS + 15) / 16, (CONV1_OUT_ROWS + 15) / 16, BATCH_SIZE * FIRST_OUTPUT_CHANNELS);

            ConvReluForward<<<gridConv1, blockConv1>>>(
                device_TrainImages[0], d_conv1W, d_conv1B, d_conv1Out,
                BATCH_SIZE, FIRST_INPUT_CHANNELS, IMAGE_ROWS, IMAGE_COLS, FIRST_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV1_OUT_ROWS, CONV1_OUT_COLS, 1, 0);

            // ---------- Conv2 + ReLU ----------
            dim3 blockConv2(16, 16);
            dim3 gridConv2((CONV2_OUT_COLS + 15) / 16, (CONV2_OUT_ROWS + 15) / 16, BATCH_SIZE * SECOND_OUTPUT_CHANNELS);

            ConvReluForward<<<gridConv2, blockConv2>>>(
                d_conv1Out, d_conv2W, d_conv2B, d_conv2Out,
                BATCH_SIZE, SECOND_INPUT_CHANNELS, CONV1_OUT_ROWS, CONV1_OUT_COLS, SECOND_OUTPUT_CHANNELS, FILTER_SIZE, FILTER_SIZE, CONV2_OUT_ROWS, CONV2_OUT_COLS, 1, 0);

            // ---------- MaxPool + Flatten ----------
            int totalFlat = BATCH_SIZE * SECOND_OUTPUT_CHANNELS * POOL_OUT_ROWS * POOL_OUT_COLS;
            int gridFlat = (totalFlat + BLOCK_SIZE - 1) / BLOCK_SIZE;

            MaxPoolFlattenForward<<<gridFlat, BLOCK_SIZE>>>(
                d_conv2Out,d_flat,
                BATCH_SIZE, SECOND_OUTPUT_CHANNELS, CONV2_OUT_ROWS, CONV2_OUT_COLS, POOL_SIZE, POOL_OUT_ROWS, POOL_OUT_COLS);

            // ---------- Fully Connected ----------
            FCForward<<<gridDim, blockDim, fcSharedBytes>>>(
                d_flat, d_fcW,d_fcB, d_fcOut, 
                BATCH_SIZE, FLATTEN_SIZE, NUM_CLASSES);

            // ---------- Softmax ----------
            int gridSoft = (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
            SoftmaxCrossForward<<<gridSoft, BLOCK_SIZE>>>(
                d_fcOut, device_Labels[0], d_loss, d_prob,
                BATCH_SIZE, NUM_CLASSES);

            CudaCheck(cudaDeviceSynchronize());

            // -------------------------------------------------
            // Accuracy
            // -------------------------------------------------
            CudaCheck(cudaMemcpy(h_prob, d_prob, BATCH_SIZE * NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost));

            for (int batchIdx = 0; batchIdx < BATCH_SIZE; batchIdx++){
                int pred = 0;
                float best = h_prob[batchIdx * NUM_CLASSES];

                for (int classes = 1; classes < NUM_CLASSES; classes++){
                    float prob = h_prob[batchIdx * NUM_CLASSES + classes];
                    if (prob > best){
                        best = prob;
                        pred = classes;
                    }
                }

                if (pred == h_testLabels[batch * BATCH_SIZE + batchIdx])
                    correct++;
            }
        }

        float accuracy = (float)correct / (testBatches * BATCH_SIZE);
        printf("Test accuracy = %.2f%%\n", accuracy * 100.0f);
        
        CudaCheck(cudaEventRecord(stopEvent_test, 0));
        CudaCheck(cudaEventSynchronize(stopEvent_test));
        float elapsedTime_test = 0.f;
        CudaCheck(cudaEventElapsedTime(&elapsedTime_test, startEvent_test, stopEvent_test));
        
        printf("Time for testing is: %.2f s\n", elapsedTime_test/1000);

        free(h_prob);
    }
    
    // ---------------------------------------------------------------------------------
    // Cleanup: Free device memory, pinned memory, and streams
    // ---------------------------------------------------------------------------------
    free(host_TrainImages);
    free(host_TrainLabels);
    free(h_testImages);
    free(h_testLabels);

    for (int i = 0; i < NUM_BUFFERS; i++){
        CudaCheck(cudaFree(device_TrainImages[i]));
        CudaCheck(cudaFree(device_Labels[i]));
        CudaCheck(cudaFreeHost(host_PinnedImages[i]));
        CudaCheck(cudaFreeHost(host_PinnedLabels[i]));
        CudaCheck(cudaStreamDestroy(stream[i]));
    }


    CudaCheck(cudaFree(d_conv1W));
    CudaCheck(cudaFree(d_conv1B));
    CudaCheck(cudaFree(d_conv1Out));
    CudaCheck(cudaFree(d_conv2W));
    CudaCheck(cudaFree(d_conv2B));
    CudaCheck(cudaFree(d_conv2Out));
    CudaCheck(cudaFree(d_poolOut));
    CudaCheck(cudaFree(d_poolIdx));
    CudaCheck(cudaFree(d_flat));
    CudaCheck(cudaFree(d_fcW));
    CudaCheck(cudaFree(d_fcB));
    CudaCheck(cudaFree(d_fcOut));
    CudaCheck(cudaFree(d_prob));
    CudaCheck(cudaFree(d_loss));
    CudaCheck(cudaFree(d_grad_fcOut));
    CudaCheck(cudaFree(d_grad_fcW));
    CudaCheck(cudaFree(d_grad_fcB));
    CudaCheck(cudaFree(d_grad_flat));
    CudaCheck(cudaFree(d_grad_poolOut));
    CudaCheck(cudaFree(d_grad_conv1Out));
    CudaCheck(cudaFree(d_grad_conv1W));
    CudaCheck(cudaFree(d_grad_conv1B));
    CudaCheck(cudaFree(d_grad_conv2Out));
    CudaCheck(cudaFree(d_grad_conv2W));
    CudaCheck(cudaFree(d_grad_conv2B));
    CudaCheck(cudaFree(d_grad_in));

    return 0;

}
