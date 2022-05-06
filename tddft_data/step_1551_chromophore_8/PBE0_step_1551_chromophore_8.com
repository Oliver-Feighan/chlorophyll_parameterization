%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1551_chromophore_8 TDDFT with PBE1PBE functional

0 1
Mg   44.161   3.130   46.843
C   42.313   6.228   46.653
C   41.123   1.487   46.698
C   45.840   0.325   46.278
C   47.098   4.979   46.421
N   42.013   3.827   46.853
C   41.452   5.078   46.711
C   39.868   5.072   46.752
C   39.647   3.524   47.082
C   41.058   2.862   46.931
C   38.998   3.313   48.480
C   39.213   5.532   45.351
C   40.044   4.987   44.084
C   40.331   6.026   42.970
O   39.810   7.154   42.825
O   41.263   5.547   42.129
N   43.546   1.062   46.554
C   42.282   0.611   46.569
C   42.346   -0.878   46.411
C   43.715   -1.239   46.310
C   44.482   0.086   46.344
C   41.099   -1.768   46.587
C   44.306   -2.640   46.205
O   43.563   -3.610   46.287
C   45.731   -2.977   46.099
N   46.103   2.737   46.354
C   46.566   1.495   46.351
C   48.066   1.392   46.507
C   48.479   2.854   46.318
C   47.081   3.587   46.450
C   48.574   0.614   47.802
C   49.258   3.163   44.985
C   50.788   3.629   45.101
N   44.686   5.208   46.703
C   45.879   5.719   46.514
C   45.722   7.140   46.403
C   44.343   7.403   46.482
C   43.723   6.201   46.635
C   46.853   8.131   46.141
C   43.332   8.424   46.433
O   43.438   9.579   46.340
C   41.917   7.667   46.601
C   41.368   8.297   47.926
O   41.825   8.097   49.020
O   40.322   9.151   47.568
C   39.730   9.842   48.801
C   41.379   6.380   40.908
C   41.711   5.404   39.793
C   42.351   5.762   38.689
C   42.707   7.120   38.165
C   42.852   4.644   37.872
C   41.759   3.928   37.029
C   41.568   2.424   37.386
C   42.090   1.479   36.241
C   41.279   0.230   36.232
C   43.657   1.270   36.330
C   44.323   1.495   34.963
C   45.171   2.791   34.912
C   45.342   3.319   33.500
C   46.713   3.954   33.525
C   44.273   4.254   32.878
C   43.774   3.817   31.451
C   44.262   4.789   30.385
C   44.567   4.102   29.038
C   43.322   3.380   28.460
C   45.123   5.114   27.899
H   40.219   0.930   46.444
H   46.468   -0.568   46.308
H   48.064   5.401   46.136
H   39.541   5.788   47.506
H   38.930   3.082   46.392
H   38.652   4.293   48.808
H   39.725   2.823   49.129
H   38.188   2.589   48.401
H   39.092   6.611   45.254
H   38.251   5.045   45.191
H   39.352   4.353   43.530
H   40.946   4.406   44.280
H   41.465   -2.374   47.415
H   40.722   -2.364   45.756
H   40.302   -1.085   46.884
H   46.318   -2.565   46.920
H   46.000   -2.619   45.105
H   45.944   -4.037   45.965
H   48.283   0.809   45.612
H   49.090   3.223   47.142
H   47.697   0.249   48.337
H   49.131   1.254   48.486
H   49.154   -0.286   47.598
H   48.640   3.838   44.392
H   49.147   2.240   44.415
H   51.395   2.795   44.748
H   51.001   3.781   46.160
H   51.054   4.587   44.654
H   46.656   9.177   46.374
H   47.226   8.014   45.124
H   47.627   7.740   46.802
H   41.213   7.764   45.774
H   40.595   10.195   49.363
H   39.151   9.130   49.388
H   39.052   10.630   48.474
H   42.099   7.198   40.941
H   40.443   6.750   40.491
H   41.621   4.375   40.140
H   42.191   7.916   38.701
H   42.526   7.317   37.108
H   43.772   7.277   38.336
H   43.426   3.997   38.535
H   43.592   4.872   37.105
H   42.034   4.077   35.985
H   40.759   4.342   37.165
H   40.502   2.282   37.566
H   42.110   2.283   38.321
H   41.877   1.958   35.285
H   41.097   -0.307   35.301
H   40.275   0.537   36.522
H   41.713   -0.448   36.968
H   43.874   0.283   36.739
H   44.064   1.922   37.103
H   43.592   1.453   34.156
H   45.049   0.682   34.920
H   46.134   2.396   35.234
H   44.800   3.531   35.621
H   45.352   2.388   32.933
H   47.119   4.227   34.499
H   46.638   4.907   33.002
H   47.420   3.281   33.039
H   44.625   5.283   32.809
H   43.424   4.291   33.560
H   42.690   3.919   31.402
H   43.998   2.788   31.171
H   45.194   5.252   30.708
H   43.511   5.575   30.307
H   45.317   3.333   29.219
H   42.942   3.857   27.557
H   42.415   3.248   29.050
H   43.630   2.359   28.233
H   45.796   4.679   27.161
H   45.647   5.921   28.412
H   44.292   5.464   27.287

