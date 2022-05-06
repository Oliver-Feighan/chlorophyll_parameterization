%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_851_chromophore_24 TDDFT with blyp functional

0 1
Mg   -0.791   44.659   24.214
C   0.985   43.919   27.231
C   -3.693   44.063   25.973
C   -2.362   44.905   21.370
C   2.349   44.814   22.508
N   -1.255   44.011   26.460
C   -0.376   43.833   27.528
C   -1.179   43.864   28.843
C   -2.678   43.840   28.357
C   -2.574   44.004   26.813
C   -3.562   42.646   28.738
C   -0.847   45.143   29.687
C   -0.536   45.011   31.145
C   -0.667   43.665   31.876
O   0.250   42.849   31.947
O   -1.903   43.460   32.431
N   -2.759   44.460   23.706
C   -3.838   44.347   24.553
C   -5.093   44.539   23.840
C   -4.722   44.930   22.472
C   -3.238   44.783   22.445
C   -6.499   44.279   24.505
C   -5.556   45.387   21.320
O   -5.073   45.778   20.249
C   -7.001   45.521   21.562
N   -0.109   44.990   22.256
C   -0.980   44.996   21.248
C   -0.285   45.177   19.948
C   1.268   45.453   20.357
C   1.164   45.122   21.754
C   -0.521   44.129   18.922
C   1.813   46.887   20.023
C   1.701   47.959   21.127
N   1.201   44.294   24.683
C   2.359   44.443   23.893
C   3.521   44.077   24.717
C   3.055   43.914   25.991
C   1.585   43.987   25.922
C   4.937   44.208   24.371
C   3.418   43.719   27.354
O   4.519   43.473   27.814
C   2.144   43.821   28.277
C   2.159   42.592   29.100
O   1.578   41.547   28.781
O   2.935   42.803   30.237
C   3.430   41.617   30.938
C   -2.111   42.169   33.105
C   -3.537   42.120   33.577
C   -4.144   41.224   34.431
C   -3.612   39.913   34.887
C   -5.552   41.536   34.854
C   -5.710   42.550   35.974
C   -6.037   43.939   35.396
C   -7.040   44.630   36.397
C   -6.591   46.041   36.786
C   -8.498   44.549   35.875
C   -9.188   43.264   36.159
C   -10.662   43.400   36.541
C   -11.418   42.045   36.336
C   -12.680   42.240   35.381
C   -11.874   41.395   37.697
C   -12.136   39.890   37.609
C   -13.197   39.392   38.778
C   -14.552   39.039   38.088
C   -15.686   40.100   38.522
C   -14.999   37.510   38.301
H   -4.618   43.884   26.526
H   -2.892   44.945   20.416
H   3.341   44.858   22.053
H   -1.164   42.951   29.439
H   -3.296   44.649   28.744
H   -3.837   41.929   27.964
H   -4.517   42.999   29.126
H   -3.073   42.101   29.545
H   -1.576   45.949   29.600
H   0.074   45.586   29.308
H   -1.113   45.685   31.777
H   0.511   45.217   31.368
H   -6.942   43.365   24.109
H   -7.241   45.060   24.340
H   -6.466   44.216   25.593
H   -7.427   45.754   20.585
H   -7.094   46.389   22.214
H   -7.454   44.590   21.901
H   -0.614   46.075   19.425
H   1.836   44.686   19.831
H   0.228   44.303   18.149
H   -1.516   44.162   18.478
H   -0.291   43.183   19.412
H   1.165   47.370   19.292
H   2.762   46.792   19.494
H   1.421   47.441   22.044
H   0.922   48.696   20.932
H   2.661   48.409   21.379
H   5.139   45.219   24.016
H   5.184   43.456   23.622
H   5.394   44.059   25.350
H   2.241   44.741   28.855
H   4.518   41.650   30.999
H   2.921   40.702   30.637
H   3.171   41.709   31.993
H   -1.566   42.187   34.049
H   -1.783   41.390   32.416
H   -4.146   42.988   33.327
H   -3.175   39.895   35.885
H   -2.820   39.586   34.212
H   -4.342   39.117   34.740
H   -5.946   40.584   35.207
H   -6.133   41.830   33.980
H   -4.717   42.555   36.424
H   -6.497   42.190   36.637
H   -6.473   43.884   34.399
H   -5.102   44.481   35.255
H   -7.020   44.183   37.390
H   -5.867   46.493   36.108
H   -6.101   46.054   37.759
H   -7.472   46.683   36.781
H   -8.593   44.696   34.799
H   -8.983   45.429   36.298
H   -8.719   42.811   37.032
H   -8.940   42.559   35.366
H   -11.110   44.217   35.974
H   -10.754   43.709   37.582
H   -10.722   41.313   35.926
H   -12.713   42.826   34.462
H   -13.571   42.609   35.889
H   -13.038   41.337   34.884
H   -12.705   41.979   38.091
H   -11.013   41.614   38.329
H   -11.235   39.352   37.904
H   -12.453   39.493   36.645
H   -13.250   40.193   39.515
H   -12.740   38.540   39.281
H   -14.422   39.147   37.011
H   -15.528   40.584   39.486
H   -16.625   39.574   38.692
H   -15.984   40.701   37.663
H   -14.543   36.954   37.482
H   -16.043   37.349   38.573
H   -14.525   37.197   39.231

