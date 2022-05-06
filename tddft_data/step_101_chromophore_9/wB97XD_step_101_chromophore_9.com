%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_101_chromophore_9 TDDFT with wB97XD functional

0 1
Mg   35.275   1.477   29.052
C   32.907   2.280   31.597
C   37.580   1.208   31.424
C   37.459   1.265   26.576
C   32.960   2.654   26.775
N   35.226   1.737   31.219
C   34.180   2.049   32.102
C   34.607   1.991   33.602
C   36.178   1.881   33.427
C   36.339   1.526   31.963
C   36.934   3.137   33.909
C   34.067   0.804   34.490
C   34.197   1.039   36.009
C   35.008   2.185   36.634
O   34.667   3.352   36.728
O   36.270   1.774   37.053
N   37.260   1.152   29.014
C   38.041   0.994   30.122
C   39.402   0.777   29.734
C   39.354   0.672   28.249
C   37.968   1.086   27.874
C   40.540   0.553   30.705
C   40.591   0.208   27.315
O   40.478   0.244   26.092
C   41.900   -0.291   27.827
N   35.145   1.817   26.913
C   36.206   1.602   26.162
C   36.044   2.045   24.719
C   34.546   2.411   24.713
C   34.172   2.220   26.178
C   37.035   3.117   24.279
C   33.708   1.568   23.699
C   33.468   0.030   24.070
N   33.286   2.142   29.113
C   32.458   2.497   28.033
C   31.088   2.711   28.512
C   31.230   2.738   29.911
C   32.545   2.325   30.221
C   29.932   2.999   27.695
C   30.560   3.031   31.159
O   29.421   3.431   31.397
C   31.600   2.723   32.287
C   31.727   4.043   33.007
O   32.327   4.996   32.592
O   31.232   3.863   34.271
C   31.516   4.951   35.189
C   36.966   2.711   37.983
C   37.674   1.869   38.994
C   37.257   1.270   40.106
C   35.931   1.361   40.862
C   38.246   0.266   40.799
C   39.532   0.800   41.329
C   40.660   -0.224   41.396
C   42.025   0.395   41.130
C   42.926   -0.716   40.642
C   42.519   0.961   42.499
C   43.214   2.313   42.361
C   44.616   2.495   43.003
C   45.641   3.313   42.066
C   47.011   2.660   42.139
C   45.687   4.770   42.394
C   44.624   5.420   41.441
C   43.916   6.623   42.179
C   44.297   8.018   41.612
C   44.463   8.959   42.794
C   43.183   8.542   40.563
H   38.312   1.126   32.230
H   38.171   1.155   25.755
H   32.325   2.847   25.908
H   34.471   2.959   34.085
H   36.545   0.927   33.807
H   36.318   3.770   34.548
H   37.470   3.694   33.141
H   37.643   2.707   34.617
H   34.442   -0.171   34.178
H   33.007   0.713   34.255
H   34.354   0.128   36.587
H   33.158   1.203   36.293
H   40.222   0.204   31.688
H   41.026   1.516   30.863
H   41.251   -0.069   30.162
H   41.638   -1.126   28.477
H   42.318   0.595   28.306
H   42.614   -0.527   27.038
H   36.175   1.288   23.946
H   34.308   3.462   24.548
H   37.705   2.789   23.484
H   37.591   3.410   25.169
H   36.560   3.994   23.838
H   34.441   1.509   22.895
H   32.801   2.061   23.348
H   34.154   -0.216   24.881
H   33.760   -0.620   23.246
H   32.406   -0.098   24.279
H   30.057   3.841   27.014
H   29.010   2.982   28.277
H   30.148   2.127   27.077
H   31.127   1.971   32.919
H   30.982   5.851   34.883
H   32.561   5.122   34.933
H   31.428   4.642   36.231
H   36.300   3.407   38.494
H   37.764   3.271   37.495
H   38.728   1.821   38.722
H   35.182   2.066   40.501
H   36.101   1.582   41.916
H   35.477   0.381   40.711
H   38.491   -0.327   39.918
H   37.865   -0.386   41.585
H   39.317   1.083   42.359
H   39.842   1.641   40.709
H   40.422   -1.145   40.864
H   40.559   -0.559   42.429
H   41.937   1.150   40.348
H   43.032   -1.515   41.376
H   43.900   -0.327   40.345
H   42.480   -1.224   39.787
H   43.169   0.277   43.045
H   41.593   1.128   43.049
H   42.515   3.001   42.836
H   43.219   2.623   41.316
H   44.929   1.459   43.131
H   44.674   2.846   44.033
H   45.323   3.154   41.035
H   47.285   2.308   41.145
H   47.005   1.783   42.787
H   47.788   3.342   42.485
H   46.631   5.311   42.324
H   45.425   4.903   43.444
H   43.868   4.760   41.015
H   45.146   5.791   40.559
H   43.952   6.544   43.265
H   42.870   6.330   42.082
H   45.257   7.842   41.127
H   44.243   9.979   42.479
H   45.495   8.846   43.125
H   43.774   8.629   43.572
H   42.528   7.701   40.336
H   43.768   8.944   39.736
H   42.564   9.334   40.984
