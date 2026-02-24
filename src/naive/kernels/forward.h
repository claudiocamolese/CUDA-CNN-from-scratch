#pragma once
#include <cuda_runtime.h>

__global__
void ConvForward(const float* input_tensor, const float* weights, const float* bias, float* output_tensor, int batchSize, int inChannels, int outChannels, int inRows, int inCols, int filterSize);

__global__
void ReLUForward(float* input_tensor, int n);

__global__
void MaxPoolForward(const float* input, float* output, int* maxIdx, int batchSize, int channels, int inRows, int inCols, int poolSize);

__global__
void FlattenForward(const float* input_tensor, float* output_tensor, int batchSize, int inChannels, int inRows, int inCols);

__global__
void FullyConnectedForward(const float* input_tensor, const float* w, const float* b, float* output_tensor, int batch_size, int num_classes, int flattent_size);

__global__
void SoftmaxCrossEntropyForward(const float* logits, const int* labels, float* outLoss, float* outProb, int batchSize, int num_classes);