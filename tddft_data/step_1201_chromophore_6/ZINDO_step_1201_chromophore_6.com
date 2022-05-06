%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1201_chromophore_6 ZINDO

0 1
Mg   16.454   -2.838   27.046
C   15.661   -0.437   29.499
C   18.368   -4.482   29.421
C   17.629   -4.754   24.573
C   14.909   -0.664   24.546
N   17.222   -2.207   29.146
C   16.578   -1.374   30.006
C   17.081   -1.482   31.540
C   18.073   -2.777   31.379
C   17.938   -3.172   29.836
C   19.524   -2.540   31.744
C   15.970   -1.629   32.575
C   15.811   -0.665   33.651
C   17.122   -0.078   34.250
O   17.341   1.108   34.436
O   18.012   -1.077   34.568
N   17.640   -4.555   27.089
C   18.303   -5.141   28.139
C   18.889   -6.352   27.664
C   18.730   -6.383   26.260
C   17.946   -5.243   25.882
C   19.823   -7.068   28.501
C   19.324   -7.385   25.301
O   19.077   -7.373   24.068
C   20.141   -8.701   25.860
N   16.217   -2.761   24.917
C   16.951   -3.654   24.146
C   16.901   -3.303   22.622
C   15.787   -2.174   22.648
C   15.636   -1.833   24.104
C   18.263   -2.787   22.026
C   14.533   -2.414   21.837
C   13.261   -2.545   22.595
N   15.516   -0.978   26.950
C   14.838   -0.270   25.956
C   14.090   0.898   26.609
C   14.404   0.848   27.971
C   15.306   -0.306   28.134
C   13.160   1.835   25.958
C   14.117   1.581   29.252
O   13.437   2.552   29.496
C   14.876   0.652   30.226
C   15.905   1.611   30.915
O   16.995   1.855   30.409
O   15.500   2.234   32.069
C   16.436   3.341   32.518
C   19.232   -0.680   35.339
C   19.097   -1.422   36.651
C   20.123   -1.647   37.504
C   21.419   -0.907   37.383
C   20.001   -2.469   38.771
C   19.348   -3.886   38.542
C   20.222   -5.085   38.767
C   19.770   -6.035   39.779
C   20.094   -5.453   41.169
C   20.452   -7.493   39.597
C   22.011   -7.537   39.596
C   22.549   -8.735   40.410
C   24.014   -9.107   40.115
C   24.106   -10.264   39.072
C   24.821   -9.487   41.390
C   26.327   -9.239   41.317
C   26.914   -9.021   42.713
C   28.062   -10.038   43.125
C   29.482   -9.622   42.748
C   27.919   -10.223   44.621
H   18.916   -5.060   30.168
H   18.011   -5.193   23.649
H   14.391   -0.020   23.833
H   17.583   -0.545   31.782
H   17.862   -3.607   32.052
H   19.857   -3.203   32.543
H   19.633   -1.496   32.040
H   20.171   -2.717   30.885
H   15.929   -2.599   33.072
H   15.081   -1.587   31.945
H   15.511   -1.290   34.492
H   15.103   0.140   33.454
H   20.839   -7.007   28.112
H   19.641   -8.139   28.590
H   19.785   -6.745   29.541
H   19.533   -9.365   26.475
H   21.045   -8.301   26.318
H   20.636   -9.281   25.081
H   16.567   -4.143   22.012
H   16.177   -1.255   22.209
H   19.108   -3.213   22.567
H   18.416   -1.710   22.101
H   18.355   -3.078   20.980
H   14.749   -3.250   21.172
H   14.429   -1.480   21.284
H   13.407   -3.040   23.555
H   12.601   -3.198   22.023
H   12.778   -1.583   22.765
H   12.746   1.195   25.179
H   13.634   2.720   25.532
H   12.522   2.119   26.795
H   14.155   0.330   30.978
H   16.293   4.290   32.002
H   17.500   3.104   32.516
H   16.241   3.547   33.571
H   19.354   0.378   35.571
H   20.137   -1.080   34.881
H   18.120   -1.807   36.943
H   21.178   -0.030   36.781
H   22.189   -1.471   36.858
H   21.831   -0.599   38.344
H   19.227   -1.951   39.337
H   20.898   -2.599   39.376
H   19.101   -3.938   37.481
H   18.371   -3.925   39.022
H   21.232   -4.835   39.093
H   20.390   -5.564   37.803
H   18.689   -6.123   39.673
H   20.857   -6.094   41.611
H   19.221   -5.456   41.822
H   20.521   -4.452   41.102
H   20.201   -7.828   38.590
H   19.999   -8.063   40.408
H   22.444   -6.658   40.073
H   22.343   -7.706   38.571
H   21.807   -9.520   40.264
H   22.454   -8.471   41.464
H   24.519   -8.329   39.542
H   24.925   -10.940   39.314
H   24.369   -9.856   38.096
H   23.152   -10.789   39.017
H   24.546   -10.510   41.650
H   24.463   -8.875   42.218
H   26.588   -8.437   40.626
H   26.604   -10.234   40.966
H   26.199   -8.894   43.526
H   27.371   -8.055   42.497
H   27.961   -11.007   42.638
H   29.343   -8.735   42.131
H   30.049   -10.326   42.138
H   30.085   -9.289   43.593
H   28.289   -11.210   44.895
H   26.935   -10.031   45.048
H   28.556   -9.490   45.117

