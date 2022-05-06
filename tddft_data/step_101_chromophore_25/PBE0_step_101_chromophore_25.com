%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_101_chromophore_25 TDDFT with PBE1PBE functional

0 1
Mg   -2.377   34.330   26.905
C   -3.159   32.521   29.687
C   -0.677   36.636   28.826
C   -1.782   36.115   24.062
C   -4.068   32.006   24.972
N   -2.013   34.569   29.007
C   -2.328   33.614   29.960
C   -1.878   34.050   31.357
C   -1.306   35.533   31.122
C   -1.217   35.550   29.590
C   -2.199   36.691   31.671
C   -0.980   33.101   32.172
C   -1.421   32.568   33.570
C   -0.340   32.424   34.683
O   0.531   31.555   34.692
O   -0.706   33.250   35.782
N   -1.452   36.092   26.479
C   -0.798   36.850   27.375
C   -0.216   37.968   26.667
C   -0.634   37.964   25.338
C   -1.336   36.654   25.240
C   0.653   38.907   27.363
C   -0.590   39.033   24.202
O   -1.013   38.902   23.081
C   -0.091   40.339   24.551
N   -3.043   34.197   24.837
C   -2.554   35.024   23.851
C   -2.994   34.612   22.457
C   -3.678   33.244   22.730
C   -3.626   33.106   24.283
C   -3.940   35.673   21.780
C   -3.135   32.164   21.769
C   -1.822   31.585   22.279
N   -3.576   32.753   27.249
C   -4.141   31.792   26.372
C   -4.786   30.699   27.075
C   -4.529   31.053   28.425
C   -3.633   32.139   28.452
C   -5.740   29.658   26.590
C   -4.753   30.578   29.733
O   -5.405   29.632   30.198
C   -3.845   31.537   30.667
C   -4.699   32.259   31.631
O   -5.279   33.316   31.354
O   -4.639   31.677   32.859
C   -5.295   32.434   33.993
C   -0.007   33.099   37.028
C   -0.623   34.130   37.971
C   -0.005   35.214   38.526
C   1.399   35.661   38.048
C   -0.741   36.035   39.548
C   -0.848   35.399   40.902
C   -0.200   36.215   42.061
C   -0.716   35.787   43.463
C   -2.046   36.438   43.725
C   0.264   36.115   44.655
C   1.381   35.115   44.879
C   2.736   35.777   44.977
C   3.830   34.891   45.621
C   3.875   35.157   47.094
C   5.218   35.149   44.996
C   5.566   34.190   43.832
C   5.658   34.938   42.484
C   5.329   33.985   41.327
C   5.751   34.682   39.989
C   3.807   33.653   41.134
H   -0.213   37.398   29.456
H   -1.280   36.376   23.129
H   -4.497   31.261   24.299
H   -2.784   34.189   31.947
H   -0.318   35.531   31.581
H   -2.485   37.440   30.932
H   -1.571   37.082   32.471
H   -3.051   36.329   32.245
H   0.004   33.489   32.436
H   -0.885   32.302   31.437
H   -1.735   31.574   33.250
H   -2.368   33.007   33.883
H   0.225   39.898   27.210
H   1.628   39.042   26.895
H   0.789   38.799   28.439
H   0.989   40.262   24.674
H   -0.555   40.687   25.473
H   -0.271   41.056   23.750
H   -2.050   34.441   21.938
H   -4.725   33.409   22.475
H   -4.229   36.429   22.510
H   -4.822   35.192   21.357
H   -3.336   36.133   20.998
H   -3.034   32.507   20.740
H   -3.876   31.365   21.745
H   -1.739   30.515   22.087
H   -1.516   31.746   23.313
H   -1.002   31.966   21.671
H   -5.439   28.753   27.119
H   -5.667   29.538   25.509
H   -6.704   30.045   26.920
H   -3.187   30.805   31.135
H   -4.650   33.293   34.179
H   -5.511   31.755   34.818
H   -6.242   32.877   33.685
H   1.004   33.426   36.785
H   -0.035   32.070   37.387
H   -1.647   33.993   38.319
H   1.415   36.745   37.938
H   1.518   35.362   37.006
H   2.194   35.290   38.695
H   -1.793   36.016   39.264
H   -0.357   37.054   39.497
H   -0.299   34.467   41.042
H   -1.887   35.130   41.089
H   -0.207   37.286   41.860
H   0.851   35.926   42.029
H   -0.911   34.724   43.318
H   -1.951   37.165   44.532
H   -2.750   35.660   44.019
H   -2.333   37.026   42.853
H   -0.351   36.037   45.552
H   0.533   37.170   44.603
H   1.401   34.520   43.966
H   1.160   34.467   45.727
H   2.712   36.734   45.499
H   2.961   36.212   44.003
H   3.517   33.854   45.497
H   4.693   35.850   47.291
H   4.141   34.408   47.839
H   2.891   35.572   47.310
H   6.008   35.096   45.744
H   5.315   36.211   44.771
H   4.883   33.342   43.782
H   6.563   33.834   44.091
H   6.644   35.401   42.461
H   4.973   35.786   42.480
H   5.804   33.019   41.497
H   5.777   35.766   40.102
H   5.044   34.470   39.188
H   6.644   34.149   39.662
H   3.590   32.772   40.531
H   3.413   34.584   40.726
H   3.291   33.499   42.081

