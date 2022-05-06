%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_551_chromophore_7 TDDFT with cam-b3lyp functional

0 1
Mg   25.489   -0.251   29.259
C   27.281   -0.449   32.326
C   22.580   -0.102   31.191
C   23.742   -0.497   26.411
C   28.367   -1.118   27.563
N   25.097   -0.297   31.510
C   25.909   -0.311   32.554
C   25.214   0.040   33.870
C   23.725   -0.069   33.486
C   23.793   -0.141   31.941
C   22.992   -1.250   34.121
C   25.650   1.407   34.548
C   24.895   1.946   35.778
C   25.760   2.142   37.090
O   27.027   2.324   37.153
O   25.073   1.745   38.219
N   23.440   -0.190   28.818
C   22.418   -0.135   29.740
C   21.147   -0.110   28.975
C   21.420   -0.390   27.612
C   22.924   -0.331   27.526
C   19.765   0.115   29.641
C   20.507   -0.725   26.454
O   20.980   -0.970   25.306
C   19.016   -0.783   26.654
N   25.977   -0.733   27.276
C   25.117   -0.572   26.236
C   25.808   -0.752   24.832
C   27.330   -0.747   25.303
C   27.214   -0.844   26.801
C   25.372   -2.101   24.147
C   28.142   0.572   24.897
C   29.480   0.271   24.254
N   27.467   -0.706   29.714
C   28.552   -1.118   29.002
C   29.713   -1.345   29.714
C   29.293   -1.005   31.033
C   27.927   -0.669   31.037
C   31.083   -1.775   29.305
C   29.700   -0.868   32.394
O   30.794   -0.944   32.909
C   28.467   -0.461   33.268
C   28.380   -1.259   34.525
O   27.922   -2.330   34.717
O   29.017   -0.597   35.524
C   29.180   -1.398   36.782
C   25.925   1.709   39.425
C   25.139   1.489   40.735
C   25.585   1.397   42.018
C   27.075   1.269   42.348
C   24.620   1.432   43.174
C   24.292   2.843   43.643
C   22.827   3.032   44.147
C   22.764   2.937   45.791
C   21.537   2.171   46.270
C   22.861   4.411   46.284
C   23.255   4.451   47.798
C   22.151   5.174   48.557
C   22.421   6.681   48.837
C   22.746   6.885   50.361
C   21.312   7.672   48.333
C   21.801   9.112   48.009
C   21.145   9.756   46.792
C   22.161   10.028   45.669
C   21.456   11.051   44.736
C   22.611   8.754   44.862
H   21.708   -0.178   31.844
H   23.314   -0.501   25.406
H   29.274   -1.129   26.955
H   25.548   -0.736   34.558
H   23.177   0.846   33.708
H   23.679   -1.929   34.625
H   22.579   -1.777   33.260
H   22.182   -0.875   34.747
H   25.752   2.201   33.808
H   26.624   1.232   35.006
H   24.188   1.136   35.959
H   24.398   2.866   35.471
H   19.875   0.436   30.676
H   19.205   -0.818   29.573
H   19.249   0.953   29.172
H   18.524   -1.312   25.838
H   18.651   0.233   26.809
H   18.823   -1.476   27.473
H   25.614   0.118   24.204
H   27.930   -1.574   24.923
H   24.422   -2.388   24.599
H   26.152   -2.841   24.328
H   25.329   -2.072   23.059
H   28.203   1.242   25.754
H   27.640   1.089   24.079
H   29.477   0.781   23.291
H   29.662   -0.774   24.002
H   30.375   0.641   24.755
H   31.697   -0.916   29.576
H   31.239   -2.097   28.275
H   31.378   -2.515   30.048
H   28.658   0.515   33.715
H   30.075   -1.178   37.364
H   29.367   -2.453   36.582
H   28.338   -1.347   37.472
H   26.525   2.616   39.499
H   26.537   0.814   39.314
H   24.075   1.608   40.530
H   27.386   0.345   41.862
H   27.328   1.261   43.408
H   27.568   2.089   41.826
H   24.940   0.906   44.074
H   23.697   0.902   42.938
H   24.521   3.546   42.843
H   25.007   3.056   44.438
H   22.157   2.257   43.776
H   22.406   3.970   43.785
H   23.570   2.300   46.154
H   20.781   2.194   45.486
H   21.036   2.761   47.037
H   21.750   1.174   46.658
H   21.962   4.984   46.057
H   23.658   4.932   45.753
H   24.245   4.874   47.966
H   23.266   3.466   48.265
H   22.093   4.701   49.537
H   21.112   4.978   48.294
H   23.302   6.985   48.272
H   21.915   7.403   50.841
H   23.640   7.503   50.433
H   22.740   5.974   50.960
H   20.507   7.708   49.067
H   20.826   7.281   47.439
H   22.888   9.067   47.954
H   21.476   9.739   48.839
H   20.716   10.709   47.102
H   20.346   9.154   46.360
H   23.067   10.371   46.170
H   20.366   11.041   44.738
H   21.751   10.861   43.704
H   21.856   12.065   44.774
H   22.021   7.899   45.195
H   23.665   8.582   45.078
H   22.473   8.859   43.786

