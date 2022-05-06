%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1301_chromophore_4 TDDFT with cam-b3lyp functional

0 1
Mg   9.557   3.470   27.111
C   10.729   1.798   29.925
C   8.291   5.926   29.128
C   8.028   4.892   24.422
C   10.622   0.806   25.132
N   9.475   3.752   29.253
C   10.051   3.013   30.241
C   9.702   3.544   31.669
C   8.790   4.782   31.316
C   8.948   4.895   29.786
C   7.359   4.628   31.798
C   10.961   3.842   32.481
C   10.885   4.675   33.740
C   11.864   4.440   34.826
O   12.949   4.947   34.941
O   11.299   3.707   35.863
N   8.291   5.201   26.817
C   7.906   6.055   27.736
C   7.214   7.109   27.170
C   7.217   6.899   25.794
C   7.849   5.632   25.615
C   6.692   8.197   28.000
C   6.778   7.804   24.721
O   6.789   7.445   23.594
C   6.217   9.135   25.006
N   9.105   2.707   25.123
C   8.584   3.561   24.182
C   8.665   3.003   22.729
C   9.494   1.689   22.918
C   9.779   1.720   24.510
C   7.295   2.773   22.046
C   10.782   1.498   22.126
C   10.869   0.119   21.461
N   10.507   1.686   27.385
C   10.938   0.740   26.484
C   11.639   -0.240   27.222
C   11.636   0.082   28.622
C   10.878   1.298   28.631
C   12.306   -1.458   26.604
C   12.003   -0.338   29.923
O   12.588   -1.334   30.344
C   11.439   0.818   30.873
C   10.486   0.256   31.944
O   9.342   -0.136   31.852
O   11.259   0.067   33.107
C   10.506   -0.472   34.285
C   12.140   3.466   37.109
C   11.718   4.518   38.114
C   12.062   4.514   39.413
C   12.752   3.481   40.265
C   11.772   5.801   40.190
C   10.528   5.853   41.094
C   9.644   7.108   40.677
C   9.641   8.219   41.783
C   8.556   7.900   42.894
C   9.502   9.614   41.152
C   10.716   10.524   41.373
C   10.438   11.647   42.421
C   11.343   11.507   43.680
C   12.465   12.618   43.622
C   10.554   11.513   44.938
C   11.390   11.248   46.181
C   10.550   10.761   47.345
C   10.400   9.190   47.372
C   8.931   8.814   46.942
C   10.827   8.604   48.752
H   7.861   6.661   29.812
H   7.813   5.357   23.457
H   10.966   -0.052   24.550
H   9.245   2.744   32.253
H   9.064   5.718   31.803
H   6.631   4.692   30.989
H   7.111   5.435   32.488
H   7.156   3.661   32.258
H   11.724   4.285   31.842
H   11.248   2.825   32.748
H   9.898   4.851   34.169
H   11.168   5.651   33.344
H   7.337   9.031   27.722
H   6.746   8.056   29.079
H   5.683   8.483   27.704
H   6.914   9.793   25.525
H   5.311   8.911   25.569
H   5.909   9.577   24.058
H   9.277   3.754   22.231
H   8.951   0.785   22.639
H   7.145   3.483   21.233
H   6.461   2.959   22.722
H   7.278   1.742   21.692
H   11.593   1.705   22.824
H   10.659   2.267   21.363
H   10.489   -0.608   22.179
H   11.910   -0.184   21.352
H   10.372   0.036   20.494
H   12.960   -1.200   25.771
H   11.637   -2.225   26.215
H   12.917   -1.966   27.350
H   12.262   1.246   31.445
H   10.097   0.404   34.788
H   11.260   -0.887   34.954
H   9.772   -1.267   34.155
H   13.229   3.499   37.066
H   11.895   2.496   37.541
H   11.073   5.369   37.898
H   12.169   3.229   41.150
H   13.789   3.711   40.512
H   12.699   2.640   39.574
H   11.804   6.660   39.519
H   12.599   5.943   40.885
H   10.953   5.844   42.098
H   9.902   4.969   41.219
H   8.606   6.778   40.635
H   9.777   7.429   39.644
H   10.594   8.192   42.312
H   9.059   7.417   43.732
H   7.768   7.259   42.500
H   7.996   8.800   43.143
H   8.564   10.026   41.524
H   9.354   9.603   40.072
H   10.781   11.026   40.408
H   11.671   10.076   41.648
H   9.378   11.758   42.648
H   10.530   12.620   41.939
H   11.900   10.570   43.682
H   13.502   12.395   43.872
H   12.358   13.437   44.332
H   12.496   13.074   42.632
H   9.963   10.597   44.944
H   9.897   12.360   45.133
H   11.886   12.160   46.512
H   12.213   10.583   45.921
H   9.597   11.290   47.347
H   11.097   11.006   48.255
H   11.071   8.715   46.656
H   8.265   9.371   47.601
H   8.788   7.767   47.208
H   8.866   8.973   45.866
H   11.910   8.557   48.867
H   10.318   7.743   49.184
H   10.573   9.416   49.434

