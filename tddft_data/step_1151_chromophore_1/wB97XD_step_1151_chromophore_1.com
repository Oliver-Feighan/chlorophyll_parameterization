%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1151_chromophore_1 TDDFT with wB97XD functional

0 1
Mg   -1.385   17.170   26.733
C   -1.754   14.854   29.314
C   -2.285   19.639   28.702
C   -1.392   19.183   23.965
C   -1.371   14.447   24.459
N   -1.839   17.274   28.743
C   -1.910   16.235   29.675
C   -2.425   16.682   31.032
C   -2.951   18.095   30.681
C   -2.240   18.423   29.304
C   -4.447   18.120   30.413
C   -1.187   16.766   31.963
C   -1.582   16.687   33.473
C   -0.419   16.355   34.417
O   0.776   16.370   34.059
O   -0.831   16.073   35.711
N   -1.602   19.161   26.420
C   -1.996   20.053   27.358
C   -2.197   21.370   26.764
C   -1.878   21.289   25.373
C   -1.573   19.821   25.234
C   -2.617   22.524   27.625
C   -1.980   22.338   24.233
O   -1.732   22.031   22.981
C   -2.381   23.771   24.683
N   -1.392   16.835   24.545
C   -1.324   17.847   23.666
C   -1.489   17.337   22.225
C   -1.096   15.896   22.421
C   -1.325   15.719   23.868
C   -2.822   17.670   21.511
C   0.273   15.602   21.852
C   1.492   15.559   22.855
N   -1.456   15.078   26.755
C   -1.499   14.097   25.807
C   -1.440   12.788   26.414
C   -1.544   13.078   27.754
C   -1.586   14.446   27.968
C   -1.348   11.370   25.845
C   -1.638   12.442   29.101
O   -1.628   11.252   29.450
C   -1.963   13.640   30.124
C   -1.105   13.366   31.291
O   0.118   13.302   31.311
O   -1.902   13.022   32.335
C   -1.339   12.552   33.550
C   0.273   16.006   36.656
C   -0.210   16.761   37.793
C   -0.254   16.544   39.136
C   -0.163   15.200   39.687
C   -0.688   17.586   40.068
C   0.507   18.287   40.796
C   1.247   19.317   39.865
C   1.001   20.762   40.352
C   0.265   21.526   39.204
C   2.331   21.659   40.570
C   2.450   21.766   42.063
C   3.886   21.411   42.631
C   4.242   22.000   44.023
C   5.726   22.323   44.198
C   3.696   21.143   45.172
C   3.062   21.943   46.313
C   1.600   22.413   46.185
C   0.803   21.167   46.477
C   -0.301   21.021   45.403
C   0.191   21.219   47.946
H   -2.630   20.469   29.323
H   -1.480   19.802   23.070
H   -1.200   13.613   23.774
H   -3.227   16.062   31.432
H   -2.835   18.753   31.543
H   -4.899   17.157   30.648
H   -4.749   18.323   29.385
H   -5.052   18.834   30.972
H   -0.703   17.678   31.612
H   -0.378   16.050   31.824
H   -2.361   15.936   33.603
H   -2.006   17.639   33.793
H   -1.872   23.319   27.626
H   -2.796   22.212   28.655
H   -3.573   22.957   27.332
H   -3.443   23.779   24.929
H   -2.149   24.481   23.890
H   -1.740   24.082   25.508
H   -0.757   17.950   21.698
H   -1.760   15.275   21.819
H   -3.410   16.979   20.907
H   -2.587   18.431   20.766
H   -3.460   18.154   22.250
H   0.500   16.284   21.032
H   0.258   14.677   21.274
H   2.386   16.015   22.429
H   1.794   14.545   23.115
H   1.177   16.077   23.761
H   -1.074   11.333   24.791
H   -2.377   11.032   25.969
H   -0.713   10.737   26.466
H   -2.997   13.556   30.458
H   -2.006   12.817   34.370
H   -0.354   12.997   33.693
H   -1.257   11.468   33.474
H   1.187   16.523   36.363
H   0.421   14.950   36.882
H   -0.462   17.775   37.483
H   -0.166   14.489   38.860
H   -0.941   15.070   40.439
H   0.781   14.954   40.173
H   -1.483   17.210   40.712
H   -1.085   18.388   39.446
H   1.230   17.535   41.112
H   0.095   18.765   41.685
H   0.901   19.234   38.834
H   2.311   19.083   39.847
H   0.317   20.856   41.195
H   1.047   21.713   38.468
H   -0.081   22.506   39.535
H   -0.564   20.950   38.792
H   2.211   22.631   40.092
H   3.265   21.251   40.183
H   1.751   21.092   42.557
H   2.225   22.768   42.427
H   4.594   21.784   41.891
H   3.835   20.324   42.703
H   3.695   22.930   44.177
H   5.913   23.367   43.947
H   6.299   21.735   43.482
H   6.007   22.196   45.243
H   4.557   20.617   45.583
H   2.950   20.451   44.780
H   3.621   22.798   46.692
H   3.112   21.211   47.120
H   1.244   22.739   45.207
H   1.429   23.268   46.839
H   1.310   20.203   46.488
H   -1.214   20.463   45.610
H   0.133   20.465   44.572
H   -0.591   21.997   45.014
H   0.311   20.287   48.497
H   -0.824   21.594   48.073
H   0.723   21.885   48.625

