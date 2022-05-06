%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1001_chromophore_10 TDDFT with wB97XD functional

0 1
Mg   40.976   7.934   29.615
C   42.930   9.328   32.195
C   38.743   6.702   31.912
C   39.262   6.578   27.099
C   43.487   8.889   27.436
N   40.868   8.125   31.713
C   41.734   8.782   32.519
C   41.197   8.686   34.047
C   39.808   7.981   33.927
C   39.799   7.519   32.484
C   39.635   6.883   34.944
C   41.067   10.125   34.728
C   40.331   10.157   36.092
C   41.181   10.366   37.400
O   42.171   9.737   37.708
O   40.808   11.576   37.988
N   39.221   6.854   29.537
C   38.486   6.413   30.567
C   37.342   5.644   30.093
C   37.398   5.679   28.673
C   38.680   6.396   28.334
C   36.401   4.862   30.949
C   36.427   5.302   27.604
O   36.686   5.326   26.442
C   34.985   5.082   28.070
N   41.413   7.668   27.541
C   40.402   7.255   26.724
C   40.915   7.346   25.260
C   42.117   8.320   25.349
C   42.309   8.373   26.849
C   41.361   5.965   24.658
C   41.846   9.825   24.896
C   40.550   10.524   25.285
N   42.855   8.726   29.780
C   43.770   9.061   28.825
C   44.933   9.660   29.317
C   44.628   9.849   30.642
C   43.397   9.229   30.866
C   46.059   10.374   28.553
C   45.211   10.329   31.901
O   46.276   10.881   32.090
C   44.144   9.922   32.936
C   44.749   9.076   33.996
O   45.063   7.879   33.980
O   44.922   9.969   35.036
C   45.314   9.368   36.348
C   41.693   12.058   39.099
C   40.775   12.172   40.305
C   40.071   11.199   41.017
C   40.046   9.687   40.788
C   39.266   11.652   42.177
C   37.849   11.255   42.333
C   37.443   10.164   43.386
C   36.789   10.766   44.757
C   37.389   10.010   45.973
C   35.248   10.792   44.966
C   34.639   12.216   44.747
C   33.444   12.230   43.738
C   32.185   12.874   44.315
C   31.390   11.919   45.232
C   31.319   13.457   43.208
C   30.544   14.737   43.656
C   31.395   16.015   43.589
C   31.007   16.895   42.346
C   32.092   17.992   42.150
C   29.598   17.558   42.325
H   37.985   6.279   32.575
H   38.708   6.235   26.223
H   44.228   9.333   26.768
H   41.940   8.030   34.500
H   39.042   8.748   34.041
H   40.042   7.206   35.902
H   40.226   6.023   34.632
H   38.605   6.576   35.129
H   40.566   10.718   33.962
H   42.071   10.512   34.902
H   39.785   9.217   36.172
H   39.532   10.882   35.938
H   37.040   4.522   31.764
H   36.103   3.909   30.512
H   35.606   5.446   31.410
H   34.902   4.125   28.586
H   34.368   4.972   27.178
H   34.648   5.913   28.689
H   40.152   7.732   24.584
H   43.003   8.066   24.768
H   41.041   5.178   25.342
H   42.434   6.026   24.482
H   40.836   5.711   23.737
H   41.933   9.814   23.810
H   42.780   10.273   25.234
H   40.173   11.010   24.385
H   40.749   11.138   26.163
H   39.745   9.845   25.564
H   45.717   11.308   28.107
H   46.503   9.768   27.763
H   46.878   10.467   29.266
H   43.737   10.831   33.378
H   46.385   9.557   36.417
H   45.132   8.294   36.329
H   44.752   9.856   37.145
H   42.038   13.055   38.826
H   42.566   11.423   39.251
H   40.726   13.213   40.626
H   39.055   9.255   40.645
H   40.431   9.147   41.653
H   40.710   9.416   39.968
H   39.317   12.726   42.354
H   39.811   11.188   42.999
H   37.546   10.909   41.345
H   37.256   12.113   42.648
H   38.215   9.438   43.642
H   36.683   9.577   42.869
H   37.152   11.791   44.830
H   38.439   9.752   45.834
H   36.900   9.060   46.192
H   37.200   10.667   46.822
H   35.028   10.552   46.006
H   34.845   10.042   44.286
H   35.382   12.973   44.496
H   34.275   12.517   45.729
H   33.104   11.276   43.338
H   33.886   12.876   42.979
H   32.496   13.721   44.926
H   30.321   11.975   45.024
H   31.633   12.172   46.263
H   31.683   10.895   45.001
H   30.670   12.660   42.847
H   31.920   13.866   42.396
H   30.189   14.478   44.654
H   29.629   14.911   43.090
H   32.435   15.688   43.599
H   31.272   16.609   44.495
H   30.991   16.152   41.548
H   32.911   17.815   42.848
H   31.606   18.949   42.339
H   32.460   17.956   41.125
H   28.891   16.964   41.746
H   29.587   18.570   41.920
H   29.188   17.536   43.335

