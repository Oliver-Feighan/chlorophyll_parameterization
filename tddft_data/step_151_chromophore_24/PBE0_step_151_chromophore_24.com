%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_151_chromophore_24 TDDFT with PBE1PBE functional

0 1
Mg   -0.067   43.736   25.372
C   2.089   43.468   28.085
C   -2.665   42.525   27.542
C   -2.075   43.359   22.776
C   2.668   44.232   23.289
N   -0.256   43.209   27.609
C   0.738   43.196   28.513
C   0.197   42.928   29.926
C   -1.274   42.592   29.722
C   -1.418   42.804   28.205
C   -1.646   41.196   30.283
C   0.410   44.207   30.844
C   0.706   43.963   32.334
C   0.514   42.671   32.953
O   1.390   41.807   33.057
O   -0.753   42.572   33.403
N   -2.184   43.323   25.193
C   -2.965   42.761   26.179
C   -4.266   42.463   25.602
C   -4.143   42.730   24.200
C   -2.753   43.198   23.975
C   -5.489   42.117   26.433
C   -5.210   42.635   23.113
O   -5.138   43.133   21.995
C   -6.555   42.076   23.445
N   0.226   43.830   23.341
C   -0.744   43.633   22.447
C   -0.138   43.800   21.019
C   1.258   44.430   21.276
C   1.438   44.128   22.689
C   -0.209   42.435   20.331
C   1.428   46.036   20.905
C   0.187   46.926   21.222
N   1.989   43.924   25.577
C   2.974   44.089   24.629
C   4.245   44.058   25.283
C   3.974   43.864   26.607
C   2.561   43.780   26.798
C   5.565   44.134   24.722
C   4.560   43.763   27.933
O   5.745   43.844   28.286
C   3.451   43.476   28.941
C   3.744   42.148   29.412
O   3.436   41.096   28.854
O   4.484   42.165   30.597
C   5.068   40.893   31.138
C   -0.849   41.372   34.258
C   -2.264   41.326   34.795
C   -2.768   41.207   36.082
C   -2.032   41.382   37.403
C   -4.258   41.032   36.330
C   -4.993   42.378   36.664
C   -6.376   42.542   35.951
C   -7.589   42.375   36.922
C   -8.374   43.666   37.218
C   -8.470   41.158   36.569
C   -9.102   40.422   37.820
C   -10.668   40.415   37.688
C   -11.403   40.161   39.060
C   -11.665   38.640   39.545
C   -12.633   40.961   38.992
C   -12.407   42.343   39.618
C   -12.959   43.492   38.824
C   -14.376   44.038   39.304
C   -15.499   43.598   38.295
C   -14.306   45.542   39.468
H   -3.345   41.944   28.170
H   -2.647   43.060   21.895
H   3.556   44.481   22.705
H   0.727   42.058   30.314
H   -1.947   43.307   30.195
H   -1.956   40.510   29.495
H   -2.382   41.339   31.074
H   -0.798   40.701   30.758
H   -0.398   44.933   30.756
H   1.253   44.698   30.358
H   0.211   44.770   32.874
H   1.727   44.344   32.360
H   -5.224   42.480   27.426
H   -5.593   41.045   26.265
H   -6.339   42.613   25.965
H   -7.301   42.775   23.821
H   -6.416   41.266   24.162
H   -6.978   41.412   22.690
H   -0.756   44.480   20.433
H   2.056   43.874   20.785
H   -0.656   41.734   21.036
H   0.750   42.155   19.895
H   -0.954   42.591   19.551
H   1.480   46.106   19.818
H   2.332   46.454   21.346
H   -0.359   47.116   20.298
H   0.568   47.836   21.686
H   -0.501   46.343   21.835
H   5.489   43.655   23.746
H   6.368   43.663   25.289
H   5.712   45.207   24.596
H   3.501   44.136   29.807
H   5.541   40.336   30.329
H   4.347   40.182   31.539
H   5.848   41.117   31.866
H   -0.126   41.356   35.074
H   -0.773   40.472   33.648
H   -3.072   41.090   34.103
H   -1.100   41.927   37.253
H   -1.720   40.346   37.535
H   -2.578   41.845   38.225
H   -4.426   40.362   37.173
H   -4.762   40.463   35.550
H   -4.443   43.294   36.447
H   -5.084   42.438   37.749
H   -6.495   41.806   35.155
H   -6.361   43.435   35.326
H   -7.164   42.211   37.912
H   -9.428   43.532   36.977
H   -8.087   44.402   36.466
H   -8.321   44.004   38.253
H   -7.966   40.413   35.954
H   -9.258   41.534   35.917
H   -8.718   40.804   38.766
H   -8.881   39.367   37.660
H   -10.923   39.581   37.034
H   -11.050   41.265   37.123
H   -10.718   40.520   39.829
H   -10.737   38.137   39.813
H   -12.052   37.975   38.773
H   -12.405   38.665   40.345
H   -13.540   40.490   39.371
H   -12.924   40.927   37.942
H   -11.384   42.658   39.822
H   -12.837   42.191   40.608
H   -12.931   43.160   37.786
H   -12.256   44.319   38.725
H   -14.602   43.717   40.321
H   -15.082   43.177   37.381
H   -16.143   44.393   37.919
H   -16.182   42.954   38.850
H   -14.122   46.085   38.541
H   -13.579   45.778   40.245
H   -15.228   45.895   39.931

