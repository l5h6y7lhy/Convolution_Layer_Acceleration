These codes are from my final project of the parallel programming course at UIUC. Specifically, I use different optimization techniques to accelerate the convolution layers of LeNet, in CUDA. The pseudocode of each convolution layer is as below.

```plaintext
for b = 0 .. Batch                     // for each image in the batch 
    for m = 0 .. Map_out               // for each output feature maps
        for h = 0 .. Height_out        // for each output element
            for w = 0 .. Width_out
            {
                output[b][m][h][w] = 0;
                for c = 0 .. Channel   // sum over all input feature maps
                    for p = 0 .. K // KxK filter
                        for q = 0 .. K
                            output[b][m][h][w] += input[b][c][h + p][w + q] * k[m][c][p][q]
            }
```

Each folder represents a single optimization technique, or a combination of several. Inside each folder, there are CUDA codes, outputs and profiling results of the corresponding optimization techniques. The report files contain detailed explanations and analysis of each optimization technique.
