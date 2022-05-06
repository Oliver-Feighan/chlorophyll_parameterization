%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_951_chromophore_3 TDDFT with PBE1PBE functional

0 1
Mg   2.007   8.302   26.690
C   2.604   10.590   29.306
C   2.369   5.805   28.796
C   1.940   6.469   23.965
C   2.010   11.207   24.515
N   2.451   8.269   28.797
C   2.545   9.324   29.700
C   2.570   8.771   31.159
C   2.686   7.233   30.902
C   2.472   7.057   29.411
C   3.933   6.488   31.384
C   1.204   9.233   31.826
C   0.886   8.487   33.215
C   1.918   7.655   33.958
O   2.998   8.161   34.348
O   1.320   6.482   34.321
N   2.246   6.375   26.410
C   2.300   5.459   27.403
C   2.463   4.082   26.814
C   2.354   4.302   25.392
C   2.171   5.751   25.224
C   2.688   2.861   27.553
C   2.412   3.233   24.269
O   2.361   3.502   23.053
C   2.543   1.731   24.653
N   2.107   8.820   24.464
C   1.883   7.816   23.610
C   1.666   8.376   22.194
C   1.997   9.890   22.316
C   1.996   10.013   23.832
C   2.629   7.691   21.099
C   1.014   10.818   21.511
C   -0.419   11.066   22.036
N   2.401   10.487   26.841
C   2.240   11.504   25.939
C   2.487   12.813   26.473
C   2.630   12.527   27.847
C   2.557   11.110   27.998
C   2.483   14.156   25.704
C   2.721   13.133   29.211
O   2.723   14.270   29.620
C   2.747   11.838   30.181
C   3.853   11.859   31.155
O   5.041   11.746   30.918
O   3.404   12.044   32.414
C   4.445   11.939   33.466
C   2.065   5.756   35.355
C   1.247   4.738   36.071
C   1.510   4.238   37.295
C   2.622   4.630   38.218
C   0.459   3.214   37.810
C   0.858   1.812   37.732
C   1.008   1.051   39.140
C   2.045   -0.112   38.940
C   3.474   0.377   38.815
C   2.040   -1.010   40.201
C   2.411   -2.480   39.844
C   3.085   -3.174   41.060
C   4.470   -3.754   40.635
C   4.336   -5.248   40.336
C   5.762   -3.575   41.486
C   6.830   -2.553   40.887
C   7.642   -1.895   42.026
C   9.152   -1.669   41.762
C   9.871   -1.940   43.064
C   9.433   -0.330   41.240
H   2.425   4.933   29.451
H   1.968   5.836   23.075
H   1.953   12.130   23.933
H   3.433   9.128   31.719
H   1.899   6.684   31.420
H   4.737   6.670   30.671
H   3.690   5.450   31.610
H   4.272   6.925   32.323
H   0.379   9.052   31.136
H   1.225   10.293   32.077
H   -0.113   8.054   33.163
H   0.820   9.328   33.906
H   3.432   2.268   27.020
H   1.763   2.285   27.525
H   2.928   2.987   28.609
H   2.454   1.302   23.655
H   1.661   1.383   25.190
H   3.538   1.551   25.060
H   0.593   8.267   22.041
H   3.000   10.151   21.979
H   3.451   7.094   21.495
H   3.026   8.445   20.419
H   1.966   7.012   20.563
H   0.967   10.563   20.452
H   1.651   11.702   21.523
H   -1.111   10.853   21.221
H   -0.353   12.102   22.367
H   -0.718   10.375   22.824
H   3.225   14.212   24.908
H   2.635   14.999   26.378
H   1.603   14.063   25.068
H   1.824   12.033   30.727
H   5.037   12.833   33.660
H   5.230   11.193   33.338
H   3.901   11.780   34.397
H   2.292   6.561   36.054
H   2.957   5.362   34.867
H   0.293   4.499   35.602
H   2.997   3.747   38.735
H   2.168   5.387   38.857
H   3.475   5.062   37.694
H   -0.498   3.424   37.331
H   0.324   3.588   38.825
H   1.795   1.746   37.179
H   0.085   1.253   37.205
H   0.037   0.751   39.535
H   1.360   1.658   39.974
H   1.720   -0.644   38.046
H   3.459   1.437   38.560
H   3.849   -0.248   38.005
H   4.109   0.169   39.677
H   1.112   -1.010   40.774
H   2.677   -0.616   40.993
H   2.801   -2.608   38.835
H   1.417   -2.926   39.809
H   2.451   -3.855   41.628
H   3.375   -2.509   41.873
H   4.644   -3.220   39.701
H   4.354   -5.890   41.217
H   5.128   -5.521   39.638
H   3.446   -5.456   39.742
H   6.218   -4.511   41.809
H   5.557   -3.007   42.393
H   6.208   -1.871   40.306
H   7.452   -3.222   40.293
H   7.425   -2.467   42.928
H   7.025   -1.003   42.137
H   9.636   -2.237   40.967
H   9.652   -2.905   43.523
H   9.693   -1.098   43.732
H   10.941   -1.933   42.858
H   10.256   -0.401   40.528
H   9.575   0.410   42.028
H   8.630   0.035   40.600

