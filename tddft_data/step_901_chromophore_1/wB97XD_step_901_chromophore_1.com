%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_901_chromophore_1 TDDFT with wB97XD functional

0 1
Mg   -2.058   17.417   26.250
C   -2.287   15.361   29.033
C   -2.799   20.172   28.225
C   -2.077   19.437   23.494
C   -2.075   14.561   24.286
N   -2.526   17.710   28.368
C   -2.375   16.755   29.346
C   -2.518   17.420   30.793
C   -3.052   18.898   30.420
C   -2.703   18.967   28.874
C   -4.527   19.303   30.621
C   -1.218   17.427   31.683
C   -1.354   18.069   33.096
C   -1.311   17.169   34.378
O   -1.776   16.038   34.389
O   -0.823   17.845   35.447
N   -2.322   19.517   25.959
C   -2.572   20.487   26.872
C   -2.629   21.744   26.331
C   -2.261   21.557   24.978
C   -2.277   20.078   24.752
C   -2.658   23.097   27.041
C   -1.895   22.588   23.895
O   -1.807   22.285   22.752
C   -1.826   24.004   24.222
N   -2.301   17.032   24.126
C   -2.109   18.024   23.249
C   -1.998   17.381   21.883
C   -1.861   15.858   22.110
C   -2.026   15.801   23.645
C   -3.250   17.735   20.974
C   -0.517   15.144   21.599
C   0.807   15.753   22.110
N   -2.076   15.373   26.574
C   -2.128   14.333   25.650
C   -2.178   13.099   26.311
C   -2.292   13.418   27.713
C   -2.114   14.825   27.775
C   -2.255   11.699   25.725
C   -2.444   12.972   29.134
O   -2.577   11.851   29.618
C   -2.453   14.196   29.942
C   -1.426   14.055   30.933
O   -0.227   14.275   30.795
O   -2.040   13.474   32.030
C   -1.250   13.001   33.217
C   -0.809   17.171   36.713
C   -0.945   18.185   37.758
C   -0.752   18.003   39.107
C   -0.487   16.572   39.709
C   -0.894   19.190   40.097
C   0.321   19.705   40.867
C   0.086   21.141   41.361
C   1.168   22.188   41.027
C   0.992   22.710   39.599
C   2.667   21.742   41.331
C   3.501   22.707   42.230
C   3.853   22.250   43.652
C   3.384   23.176   44.832
C   4.586   23.960   45.274
C   2.847   22.386   46.087
C   1.351   22.033   45.668
C   0.430   21.727   46.840
C   -0.616   22.856   46.917
C   -0.970   23.080   48.413
C   -1.878   22.404   46.061
H   -3.200   21.052   28.732
H   -1.999   20.020   22.574
H   -2.047   13.651   23.681
H   -3.401   16.939   31.212
H   -2.560   19.648   31.039
H   -4.878   19.800   29.716
H   -4.622   20.021   31.436
H   -5.120   18.428   30.886
H   -0.366   17.850   31.149
H   -0.916   16.401   31.895
H   -2.357   18.442   33.306
H   -0.676   18.916   33.201
H   -2.705   23.077   28.130
H   -3.464   23.740   26.691
H   -1.733   23.631   26.823
H   -2.666   24.411   24.785
H   -1.875   24.505   23.256
H   -0.911   24.132   24.801
H   -1.192   17.879   21.345
H   -2.652   15.259   21.657
H   -3.894   18.314   21.636
H   -3.848   16.882   20.652
H   -3.006   18.385   20.134
H   -0.421   15.128   20.514
H   -0.476   14.106   21.928
H   1.380   15.012   22.667
H   0.576   16.559   22.806
H   1.405   16.071   21.255
H   -3.187   11.198   25.989
H   -1.421   11.159   26.174
H   -1.958   11.683   24.676
H   -3.444   14.266   30.389
H   -0.243   13.367   33.017
H   -1.205   11.912   33.222
H   -1.618   13.354   34.181
H   0.127   16.615   36.775
H   -1.702   16.546   36.751
H   -1.272   19.166   37.414
H   -0.217   15.810   38.978
H   -1.415   16.231   40.167
H   0.350   16.759   40.381
H   -1.580   18.706   40.791
H   -1.398   19.951   39.501
H   1.202   19.711   40.225
H   0.438   19.095   41.762
H   0.116   21.138   42.451
H   -0.849   21.600   41.040
H   0.928   23.088   41.594
H   0.128   22.252   39.119
H   1.916   22.523   39.052
H   0.716   23.759   39.501
H   3.262   21.579   40.433
H   2.737   20.735   41.742
H   2.804   23.519   42.439
H   4.376   23.151   41.755
H   4.942   22.244   43.703
H   3.554   21.236   43.918
H   2.646   23.841   44.383
H   5.304   24.125   44.471
H   5.102   23.329   45.998
H   4.199   24.819   45.823
H   2.818   22.985   46.998
H   3.395   21.459   46.252
H   1.435   21.188   44.985
H   0.934   22.797   45.013
H   1.026   21.622   47.747
H   -0.037   20.767   46.620
H   -0.200   23.764   46.479
H   -0.502   24.021   48.701
H   -0.596   22.291   49.065
H   -2.044   23.090   48.595
H   -1.692   22.323   44.990
H   -2.676   23.138   46.174
H   -2.290   21.447   46.380

