#pragma once
#include <cuda_runtime.h>

__global__ void SoftmaxCrossEntropyBackward(float* grad_logits, const float* prob, const int* labels, int batch_size, int num_classes);

__global__ void FullyConnectedLayerBackward(const float* grad_out, const float* in, float* gradW, float* gradB, int batch_size, int num_classes, int flatten_size);

__global__ void FullyConnectedBackward(const float* gradOut, const float* w, float* gradIn, int batch_size, int num_classes, int flatten_size);

__global__ void FlattenBackward(const float* gradFlat, float* gradPoolOut, int batchSize, int num_filters, int pool_out_rows, int pool_out_cols, int flatten_size);

__global__ void MaxPoolBackward(const float* gradOut, float* gradIn, const int* maxIdx, int batch_size, int num_filters, int pool_out_rows, int pool_out_cols);

__global__ void ReLUBackward(const float* gradOut, const float* x, float* gradIn, int n);

__global__ void ConvLayerBackward(const float* input, const float* gradOut, float* gradW, float* gradB, int batchSize, int inChannels, int outChannels, int inRows, int inCols, int outRows, int outCols, int filterSize);

__global__ void ConvBackward(const float* gradOut, const float* w, float* gradIn, int batchSize, int inChannels, int outChannels, int inRows, int inCols, int outRows, int outCols, int filterSize);

__global__ void SGDBackward(float* param, const float* grad, float lr, int n);
