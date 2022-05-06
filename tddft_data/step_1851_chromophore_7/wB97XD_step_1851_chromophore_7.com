%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1851_chromophore_7 TDDFT with wB97XD functional

0 1
Mg   26.620   0.519   29.138
C   28.338   -0.127   32.122
C   23.734   0.895   30.992
C   24.852   0.934   26.228
C   29.460   -0.184   27.322
N   26.130   0.393   31.257
C   26.986   0.285   32.334
C   26.281   0.438   33.663
C   24.814   0.838   33.244
C   24.874   0.707   31.708
C   23.718   0.001   33.983
C   26.949   1.385   34.668
C   26.168   1.942   35.880
C   26.967   2.342   37.158
O   27.281   3.480   37.430
O   27.267   1.238   37.926
N   24.572   0.904   28.668
C   23.563   0.954   29.616
C   22.340   1.256   28.935
C   22.587   1.268   27.461
C   24.068   0.977   27.397
C   20.996   1.481   29.682
C   21.734   1.619   26.311
O   22.000   1.353   25.108
C   20.418   2.172   26.516
N   27.036   0.228   27.113
C   26.236   0.653   26.151
C   26.964   0.587   24.755
C   28.319   0.031   25.118
C   28.303   0.080   26.617
C   26.240   -0.290   23.699
C   29.413   1.073   24.564
C   30.759   0.400   24.269
N   28.597   0.004   29.604
C   29.595   -0.275   28.718
C   30.675   -0.803   29.491
C   30.324   -0.667   30.856
C   28.991   -0.266   30.861
C   31.963   -1.262   28.961
C   30.787   -0.757   32.274
O   31.891   -1.025   32.680
C   29.488   -0.342   33.061
C   29.238   -1.395   34.156
O   28.751   -2.488   34.040
O   29.680   -0.859   35.341
C   29.505   -1.719   36.475
C   27.720   1.454   39.335
C   26.623   1.261   40.325
C   26.692   1.072   41.725
C   27.928   1.312   42.553
C   25.391   0.917   42.468
C   24.844   2.305   42.747
C   23.314   2.259   43.197
C   23.012   2.341   44.749
C   22.037   1.243   45.280
C   22.438   3.741   45.160
C   22.949   4.336   46.463
C   21.874   4.456   47.529
C   22.357   5.506   48.526
C   22.725   4.787   49.861
C   21.283   6.655   48.642
C   21.616   7.910   47.807
C   20.658   8.101   46.645
C   21.447   8.112   45.326
C   22.602   9.096   45.215
C   20.521   8.230   44.113
H   22.896   1.121   31.653
H   24.323   1.017   25.276
H   30.360   -0.483   26.780
H   26.393   -0.596   33.991
H   24.657   1.890   33.481
H   24.125   -1.010   34.009
H   22.814   0.008   33.373
H   23.424   0.387   34.958
H   27.189   2.300   34.126
H   27.799   0.917   35.164
H   25.379   1.249   36.170
H   25.632   2.808   35.491
H   20.997   2.413   30.246
H   21.020   0.664   30.403
H   20.098   1.293   29.093
H   20.607   3.018   27.178
H   19.816   1.374   26.950
H   19.929   2.550   25.618
H   27.063   1.599   24.364
H   28.394   -1.024   24.857
H   25.243   -0.581   24.031
H   26.716   -1.196   23.324
H   26.020   0.381   22.869
H   29.418   1.924   25.245
H   29.043   1.554   23.659
H   30.897   0.345   23.189
H   30.870   -0.602   24.684
H   31.494   1.129   24.610
H   31.810   -2.251   28.528
H   32.651   -1.328   29.803
H   32.565   -0.593   28.347
H   29.820   0.601   33.495
H   29.783   -2.758   36.302
H   28.464   -1.729   36.797
H   30.172   -1.355   37.257
H   28.114   2.453   39.522
H   28.446   0.660   39.507
H   25.657   1.124   39.839
H   28.843   1.375   41.963
H   27.923   0.574   43.355
H   27.724   2.212   43.131
H   25.679   0.327   43.338
H   24.664   0.345   41.892
H   24.804   2.849   41.803
H   25.522   2.900   43.360
H   22.780   1.400   42.789
H   22.847   3.159   42.799
H   23.994   2.273   45.218
H   22.664   0.835   46.072
H   21.967   0.457   44.528
H   21.109   1.542   45.767
H   21.350   3.691   45.211
H   22.595   4.453   44.349
H   23.356   5.295   46.143
H   23.702   3.677   46.894
H   21.862   3.456   47.961
H   20.881   4.620   47.111
H   23.216   6.052   48.136
H   21.952   4.050   50.074
H   22.813   5.544   50.640
H   23.643   4.248   49.624
H   21.121   7.014   49.658
H   20.294   6.255   48.418
H   22.653   7.878   47.476
H   21.507   8.857   48.335
H   20.252   9.100   46.804
H   19.818   7.407   46.667
H   22.029   7.191   45.294
H   23.482   8.643   45.672
H   22.514   10.071   45.694
H   22.812   9.153   44.147
H   20.632   7.355   43.473
H   20.554   9.113   43.474
H   19.491   8.220   44.468

