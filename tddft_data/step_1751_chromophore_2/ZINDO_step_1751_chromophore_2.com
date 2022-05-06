%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1751_chromophore_2 ZINDO

0 1
Mg   2.077   -0.142   42.940
C   5.624   0.489   42.338
C   1.466   2.866   41.338
C   -1.132   -0.939   42.893
C   3.048   -3.503   43.827
N   3.410   1.418   41.849
C   4.747   1.543   41.954
C   5.201   2.939   41.432
C   3.880   3.791   41.158
C   2.847   2.654   41.502
C   3.624   5.158   41.943
C   6.165   2.815   40.205
C   5.841   1.866   38.946
C   5.597   2.518   37.632
O   5.929   3.697   37.362
O   4.963   1.705   36.759
N   0.399   0.833   42.224
C   0.342   2.035   41.591
C   -1.028   2.468   41.442
C   -1.859   1.358   41.852
C   -0.849   0.319   42.283
C   -1.566   3.748   40.935
C   -3.307   1.246   41.908
O   -4.010   2.175   41.473
C   -4.139   0.079   42.441
N   1.157   -1.927   43.212
C   -0.227   -1.951   43.313
C   -0.747   -3.297   43.891
C   0.635   -4.034   44.153
C   1.693   -3.082   43.715
C   -1.534   -3.093   45.174
C   0.702   -5.492   43.465
C   1.118   -6.630   44.438
N   3.931   -1.239   43.188
C   4.106   -2.632   43.442
C   5.450   -2.943   43.378
C   6.188   -1.805   43.024
C   5.183   -0.773   42.871
C   5.958   -4.368   43.618
C   7.452   -1.157   42.721
O   8.573   -1.601   42.836
C   7.150   0.343   42.376
C   7.723   1.249   43.385
O   8.221   2.337   43.070
O   7.566   0.755   44.632
C   8.465   1.519   45.560
C   4.841   1.943   35.358
C   5.886   1.100   34.664
C   5.567   -0.098   34.070
C   4.224   -0.742   34.142
C   6.579   -0.743   33.176
C   6.249   -0.675   31.609
C   6.611   0.691   30.957
C   5.447   1.351   30.159
C   6.175   2.305   29.253
C   4.377   2.174   31.101
C   2.937   1.935   30.629
C   1.791   1.727   31.638
C   0.727   2.864   31.637
C   0.623   3.443   33.111
C   -0.648   2.457   31.128
C   -1.537   3.645   30.663
C   -1.981   3.716   29.240
C   -3.498   3.996   29.165
C   -3.981   4.063   27.673
C   -3.784   5.429   29.820
H   1.339   3.828   40.838
H   -2.139   -1.287   43.133
H   3.310   -4.475   44.250
H   5.784   3.443   42.203
H   3.811   3.917   40.077
H   2.701   5.013   42.505
H   3.422   5.922   41.192
H   4.388   5.369   42.692
H   7.167   2.660   40.605
H   6.336   3.827   39.837
H   4.950   1.295   39.208
H   6.596   1.084   39.035
H   -2.125   3.633   40.007
H   -0.727   4.374   40.631
H   -2.138   4.295   41.685
H   -3.944   -0.098   43.499
H   -4.030   -0.857   41.895
H   -5.192   0.352   42.369
H   -1.369   -3.786   43.141
H   0.925   -4.266   45.177
H   -2.522   -3.533   45.037
H   -1.777   -2.060   45.421
H   -1.214   -3.540   46.115
H   1.371   -5.509   42.605
H   -0.275   -5.737   43.049
H   2.130   -6.997   44.264
H   0.389   -7.439   44.390
H   1.031   -6.173   45.424
H   5.321   -4.836   44.369
H   7.027   -4.374   43.833
H   5.752   -4.975   42.737
H   7.588   0.532   41.396
H   8.376   1.009   46.520
H   8.356   2.602   45.502
H   9.466   1.352   45.162
H   5.012   2.998   35.149
H   3.905   1.709   34.850
H   6.858   1.550   34.461
H   3.809   -0.740   33.134
H   3.568   -0.160   34.790
H   4.325   -1.773   34.481
H   6.651   -1.785   33.488
H   7.583   -0.376   33.388
H   5.254   -1.083   31.432
H   6.871   -1.478   31.213
H   7.421   0.534   30.244
H   6.997   1.324   31.756
H   4.992   0.582   29.534
H   6.162   1.929   28.230
H   7.128   2.751   29.537
H   5.507   3.146   29.067
H   4.629   3.228   30.980
H   4.535   1.910   32.146
H   2.905   1.128   29.897
H   2.831   2.886   30.107
H   2.319   1.522   32.569
H   1.260   0.803   31.412
H   1.129   3.623   30.966
H   0.659   4.530   33.176
H   1.484   3.125   33.699
H   -0.329   3.134   33.543
H   -1.232   1.993   31.923
H   -0.541   1.747   30.307
H   -1.030   4.576   30.919
H   -2.500   3.595   31.172
H   -1.764   2.827   28.647
H   -1.398   4.483   28.731
H   -4.141   3.390   29.803
H   -4.324   3.138   27.212
H   -3.153   4.375   27.037
H   -4.827   4.744   27.582
H   -4.193   5.210   30.806
H   -4.634   5.849   29.282
H   -3.009   6.160   30.048

