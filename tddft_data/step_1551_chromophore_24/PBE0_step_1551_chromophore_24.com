%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1551_chromophore_24 TDDFT with PBE1PBE functional

0 1
Mg   -0.192   43.809   24.678
C   2.107   43.195   27.186
C   -2.647   42.769   26.867
C   -2.396   43.538   22.028
C   2.387   44.290   22.541
N   -0.249   42.901   26.827
C   0.829   42.916   27.701
C   0.413   42.532   29.207
C   -1.196   42.327   28.956
C   -1.420   42.717   27.462
C   -1.758   40.996   29.370
C   0.716   43.586   30.259
C   -0.024   43.439   31.626
C   -0.261   41.993   32.177
O   0.630   41.111   32.203
O   -1.653   41.768   32.385
N   -2.250   43.394   24.512
C   -3.057   43.066   25.508
C   -4.433   42.919   25.058
C   -4.344   43.043   23.610
C   -2.981   43.328   23.309
C   -5.636   42.581   25.874
C   -5.449   42.834   22.602
O   -5.317   43.179   21.447
C   -6.863   42.264   22.940
N   -0.001   43.878   22.641
C   -1.079   43.856   21.785
C   -0.640   44.119   20.299
C   0.863   44.515   20.515
C   1.088   44.265   21.997
C   -0.851   42.886   19.387
C   1.281   45.932   20.042
C   0.628   47.162   20.832
N   1.864   43.824   24.769
C   2.812   44.034   23.845
C   4.138   43.951   24.422
C   3.923   43.712   25.779
C   2.496   43.580   25.896
C   5.415   44.188   23.573
C   4.561   43.501   27.095
O   5.730   43.456   27.427
C   3.396   43.200   28.099
C   3.666   41.951   28.815
O   3.804   40.887   28.212
O   3.875   42.199   30.175
C   4.238   41.055   31.023
C   -2.069   40.393   32.804
C   -3.100   40.594   33.893
C   -2.907   41.055   35.119
C   -1.560   41.405   35.708
C   -4.030   41.400   36.089
C   -4.616   42.835   35.927
C   -6.087   42.796   35.582
C   -7.136   42.771   36.771
C   -7.426   44.270   37.153
C   -8.524   42.055   36.459
C   -9.077   41.153   37.560
C   -10.236   41.798   38.366
C   -11.619   41.391   37.898
C   -12.329   42.531   37.052
C   -12.570   41.077   39.159
C   -13.273   39.697   39.219
C   -14.201   39.521   37.943
C   -13.812   38.056   37.435
C   -12.871   38.137   36.240
C   -15.018   37.201   37.142
H   -3.478   42.596   27.554
H   -2.958   43.350   21.110
H   3.169   44.510   21.811
H   0.960   41.634   29.494
H   -1.723   43.086   29.535
H   -2.647   41.140   29.984
H   -0.952   40.538   29.943
H   -2.055   40.198   28.690
H   0.423   44.550   29.842
H   1.758   43.764   30.525
H   -1.010   43.891   31.512
H   0.569   43.996   32.351
H   -6.362   41.877   25.467
H   -6.186   43.520   25.932
H   -5.411   42.223   26.878
H   -7.291   42.902   23.713
H   -6.761   41.275   23.386
H   -7.394   42.158   21.994
H   -1.317   44.865   19.883
H   1.527   43.775   20.068
H   0.064   42.299   19.303
H   -1.292   43.094   18.412
H   -1.589   42.163   19.734
H   0.945   46.077   19.015
H   2.350   46.016   19.843
H   0.026   47.719   20.114
H   1.471   47.759   21.179
H   -0.063   46.881   21.628
H   5.495   45.199   23.176
H   5.423   43.451   22.770
H   6.328   44.023   24.145
H   3.342   43.980   28.858
H   4.975   40.431   30.517
H   3.430   40.367   31.273
H   4.784   41.312   31.930
H   -1.263   39.728   33.115
H   -2.636   39.926   31.999
H   -4.112   40.479   33.504
H   -1.322   42.455   35.538
H   -0.777   40.905   35.139
H   -1.449   41.137   36.759
H   -3.770   41.304   37.143
H   -4.835   40.677   35.957
H   -4.038   43.231   35.093
H   -4.402   43.322   36.878
H   -6.359   41.897   35.030
H   -6.238   43.564   34.824
H   -6.671   42.185   37.564
H   -8.028   44.732   36.370
H   -6.534   44.839   37.418
H   -8.110   44.251   38.002
H   -8.487   41.524   35.508
H   -9.253   42.818   36.184
H   -8.251   40.885   38.219
H   -9.467   40.258   37.076
H   -10.198   42.881   38.250
H   -10.099   41.575   39.424
H   -11.556   40.603   37.148
H   -12.158   42.237   36.016
H   -11.812   43.490   37.089
H   -13.401   42.668   37.194
H   -13.340   41.843   39.252
H   -12.028   41.103   40.104
H   -13.771   39.587   40.182
H   -12.512   38.930   39.368
H   -13.862   40.252   37.209
H   -15.248   39.562   38.244
H   -13.294   37.482   38.203
H   -11.828   38.032   36.542
H   -13.001   39.036   35.638
H   -13.021   37.320   35.536
H   -14.838   36.164   37.428
H   -15.248   37.188   36.077
H   -15.896   37.587   37.660

