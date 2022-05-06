%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1551_chromophore_2 TDDFT with PBE1PBE functional

0 1
Mg   2.817   0.513   43.594
C   6.345   0.887   43.143
C   2.438   3.200   41.512
C   -0.318   -0.453   42.820
C   3.479   -2.667   44.947
N   4.245   2.034   42.626
C   5.625   1.955   42.511
C   6.220   3.142   41.740
C   4.954   4.073   41.614
C   3.763   3.031   41.886
C   4.862   5.286   42.530
C   6.811   2.876   40.347
C   6.266   1.812   39.381
C   5.774   2.162   37.953
O   5.626   3.294   37.567
O   5.682   1.058   37.172
N   1.258   1.385   42.505
C   1.293   2.511   41.731
C   -0.018   2.819   41.194
C   -0.916   1.875   41.752
C   0.004   0.866   42.437
C   -0.384   4.027   40.315
C   -2.323   1.878   41.468
O   -2.790   2.738   40.741
C   -3.315   0.868   42.075
N   1.723   -1.297   43.929
C   0.445   -1.461   43.471
C   0.012   -2.906   43.690
C   1.096   -3.566   44.579
C   2.195   -2.447   44.500
C   -1.459   -3.344   44.003
C   1.555   -5.023   44.050
C   1.420   -6.279   44.919
N   4.516   -0.603   44.105
C   4.592   -1.852   44.696
C   6.032   -2.157   44.979
C   6.729   -1.198   44.290
C   5.745   -0.200   43.842
C   6.562   -3.431   45.616
C   8.052   -0.767   43.877
O   9.236   -1.172   43.964
C   7.872   0.667   43.152
C   8.565   1.678   44.062
O   9.067   2.727   43.655
O   8.479   1.332   45.396
C   9.381   2.144   46.325
C   5.430   1.302   35.745
C   6.083   0.327   34.848
C   5.504   -0.679   34.130
C   4.035   -0.963   34.218
C   6.349   -1.575   33.249
C   6.122   -1.351   31.728
C   6.814   -0.098   31.173
C   5.898   0.647   30.185
C   6.682   1.382   29.075
C   4.790   1.585   30.984
C   3.501   1.751   30.157
C   2.339   0.825   30.615
C   1.204   1.490   31.465
C   0.864   0.746   32.821
C   -0.111   1.550   30.720
C   -0.959   2.847   31.044
C   -2.178   3.144   30.111
C   -2.085   4.513   29.368
C   -3.038   5.481   30.019
C   -2.495   4.448   27.859
H   2.322   4.012   40.791
H   -1.357   -0.739   42.648
H   3.866   -3.621   45.310
H   6.987   3.664   42.313
H   4.809   4.473   40.611
H   4.494   6.150   41.977
H   5.786   5.518   43.060
H   4.072   5.011   43.229
H   7.853   2.725   40.629
H   6.793   3.848   39.855
H   5.451   1.330   39.921
H   7.080   1.098   39.257
H   -0.704   3.753   39.310
H   0.546   4.595   40.347
H   -1.137   4.616   40.839
H   -3.250   0.734   43.154
H   -3.249   -0.179   41.778
H   -4.312   1.267   41.885
H   0.139   -3.230   42.657
H   0.699   -3.827   45.560
H   -1.943   -3.708   43.096
H   -1.992   -2.417   44.215
H   -1.379   -4.076   44.807
H   2.539   -4.866   43.609
H   0.864   -5.193   43.223
H   0.561   -6.853   44.572
H   1.175   -6.116   45.969
H   2.270   -6.946   44.770
H   7.500   -3.734   45.152
H   5.820   -4.214   45.774
H   6.873   -3.296   46.651
H   8.328   0.756   42.166
H   9.291   3.205   46.092
H   10.425   1.838   46.265
H   9.094   1.940   47.357
H   5.968   2.192   35.421
H   4.359   1.443   35.599
H   7.110   0.563   34.569
H   3.903   -1.908   34.745
H   3.585   -1.033   33.228
H   3.375   -0.244   34.704
H   6.107   -2.596   33.547
H   7.409   -1.573   33.500
H   5.069   -1.152   31.527
H   6.555   -2.221   31.236
H   7.754   -0.469   30.763
H   7.056   0.599   31.975
H   5.320   -0.065   29.597
H   6.333   2.415   29.062
H   6.504   0.839   28.147
H   7.751   1.419   29.288
H   5.106   2.564   31.346
H   4.521   1.029   31.882
H   3.764   1.698   29.101
H   3.242   2.778   30.413
H   2.774   0.009   31.192
H   2.062   0.276   29.715
H   1.492   2.498   31.765
H   -0.137   1.042   33.135
H   1.527   0.995   33.649
H   0.885   -0.339   32.721
H   -0.546   0.567   30.899
H   -0.068   1.477   29.633
H   -0.283   3.687   30.882
H   -1.298   2.879   32.080
H   -3.067   3.026   30.731
H   -2.278   2.427   29.296
H   -1.083   4.931   29.458
H   -4.104   5.558   29.805
H   -2.634   6.486   29.896
H   -2.970   5.203   31.071
H   -3.412   3.873   27.991
H   -1.800   3.927   27.200
H   -2.719   5.433   27.452

