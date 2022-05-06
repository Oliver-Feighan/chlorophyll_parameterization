%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1701_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   36.215   1.322   29.818
C   34.032   2.582   32.351
C   38.606   1.309   32.358
C   38.402   0.705   27.488
C   33.697   1.875   27.465
N   36.286   1.975   32.055
C   35.320   2.442   32.824
C   35.685   2.424   34.321
C   37.213   2.372   34.245
C   37.429   1.896   32.776
C   38.068   3.726   34.553
C   35.165   1.166   35.105
C   34.742   1.331   36.565
C   35.690   2.156   37.423
O   35.521   3.259   37.955
O   36.913   1.534   37.459
N   38.189   0.889   29.914
C   38.974   0.841   31.007
C   40.311   0.395   30.676
C   40.234   0.274   29.219
C   38.913   0.578   28.806
C   41.398   0.179   31.647
C   41.282   -0.187   28.221
O   41.157   -0.390   27.016
C   42.583   -0.689   28.874
N   36.038   1.373   27.758
C   37.187   1.212   27.018
C   36.921   1.383   25.516
C   35.365   1.531   25.435
C   34.944   1.463   26.950
C   37.789   2.544   24.801
C   34.620   0.552   24.467
C   34.148   -0.737   25.078
N   34.233   2.037   29.864
C   33.373   2.228   28.881
C   32.126   2.711   29.417
C   32.304   2.817   30.784
C   33.605   2.391   31.048
C   30.913   3.008   28.542
C   31.580   3.203   31.997
O   30.410   3.486   32.162
C   32.781   2.931   33.165
C   32.855   4.118   33.951
O   33.375   5.155   33.613
O   32.437   3.878   35.252
C   32.973   4.932   36.172
C   37.977   2.207   38.232
C   37.924   1.596   39.597
C   37.491   2.135   40.730
C   36.893   3.541   40.832
C   37.552   1.357   42.048
C   38.709   1.852   42.982
C   39.766   0.785   43.297
C   41.073   1.099   42.457
C   41.803   -0.228   42.006
C   42.121   1.807   43.428
C   42.977   2.833   42.750
C   44.010   3.468   43.557
C   44.446   4.900   43.006
C   45.945   5.084   43.263
C   43.646   6.067   43.548
C   43.437   7.099   42.529
C   42.970   8.495   43.075
C   43.893   9.691   42.847
C   43.673   10.279   41.442
C   45.370   9.475   43.223
H   39.287   1.217   33.207
H   39.051   0.442   26.650
H   32.883   1.924   26.739
H   35.273   3.367   34.679
H   37.645   1.660   34.948
H   38.813   3.933   33.784
H   38.624   3.612   35.484
H   37.387   4.572   34.641
H   35.834   0.341   34.860
H   34.179   0.892   34.729
H   34.831   0.338   37.007
H   33.738   1.732   36.697
H   41.807   -0.806   31.423
H   41.030   0.105   32.670
H   42.249   0.856   31.575
H   42.942   0.015   29.625
H   43.355   -0.938   28.145
H   42.256   -1.646   29.280
H   37.179   0.456   25.003
H   35.177   2.527   25.034
H   37.063   3.352   24.888
H   37.999   2.189   23.792
H   38.652   2.825   25.404
H   35.376   0.242   23.746
H   33.868   0.967   23.797
H   34.921   -1.503   25.131
H   33.302   -1.089   24.488
H   33.872   -0.727   26.132
H   30.535   2.013   28.308
H   31.165   3.463   27.585
H   30.169   3.584   29.093
H   32.389   2.109   33.764
H   32.397   4.960   37.097
H   32.981   5.970   35.839
H   34.000   4.696   36.450
H   37.911   3.295   38.236
H   38.970   2.050   37.811
H   38.276   0.564   39.589
H   37.224   4.147   39.989
H   37.246   4.015   41.748
H   35.806   3.596   40.903
H   37.539   0.267   42.028
H   36.645   1.583   42.609
H   38.297   2.194   43.931
H   39.208   2.713   42.536
H   39.372   -0.215   43.117
H   40.040   0.805   44.352
H   40.924   1.736   41.585
H   42.605   -0.569   42.661
H   42.257   -0.088   41.025
H   41.159   -1.107   42.032
H   42.663   1.129   44.087
H   41.536   2.373   44.153
H   42.327   3.687   42.561
H   43.499   2.416   41.888
H   44.947   2.914   43.506
H   43.800   3.552   44.623
H   44.259   4.830   41.934
H   46.112   6.135   43.497
H   46.408   4.753   42.333
H   46.362   4.470   44.061
H   44.216   6.548   44.343
H   42.743   5.718   44.049
H   42.785   6.640   41.787
H   44.390   7.165   42.004
H   42.818   8.430   44.152
H   41.965   8.531   42.654
H   43.524   10.401   43.588
H   43.644   11.367   41.492
H   42.774   9.861   40.990
H   44.568   10.110   40.842
H   45.509   8.530   43.748
H   45.651   10.203   43.984
H   46.031   9.536   42.358

