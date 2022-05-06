%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_751_chromophore_7 TDDFT with PBE1PBE functional

0 1
Mg   25.788   -0.129   29.202
C   27.702   -0.339   32.025
C   22.935   -0.053   31.264
C   23.769   -0.273   26.453
C   28.543   -0.740   27.201
N   25.314   -0.174   31.366
C   26.351   -0.180   32.305
C   25.795   -0.029   33.740
C   24.264   -0.147   33.482
C   24.129   -0.086   31.988
C   23.532   -1.319   34.219
C   26.499   1.160   34.473
C   25.652   2.047   35.467
C   26.387   2.367   36.838
O   27.591   2.638   36.864
O   25.579   2.288   37.934
N   23.652   0.048   28.871
C   22.688   0.032   29.840
C   21.422   0.245   29.149
C   21.609   0.097   27.758
C   23.063   -0.018   27.644
C   20.095   0.400   30.006
C   20.612   0.175   26.675
O   20.884   0.192   25.495
C   19.193   0.336   27.059
N   26.106   -0.458   27.085
C   25.074   -0.541   26.203
C   25.622   -0.618   24.778
C   27.117   -0.869   25.048
C   27.305   -0.690   26.545
C   24.834   -1.675   23.933
C   28.135   -0.005   24.226
C   29.435   -0.582   23.656
N   27.709   -0.605   29.489
C   28.761   -0.832   28.601
C   29.974   -1.139   29.364
C   29.613   -0.971   30.731
C   28.221   -0.607   30.727
C   31.263   -1.495   28.753
C   30.195   -0.849   32.019
O   31.317   -1.071   32.478
C   28.908   -0.461   32.897
C   28.689   -1.568   33.887
O   27.882   -2.519   33.922
O   29.454   -1.302   35.024
C   29.123   -2.045   36.233
C   26.325   2.498   39.196
C   25.382   2.224   40.356
C   25.660   1.977   41.692
C   27.063   1.753   42.309
C   24.483   1.822   42.698
C   24.340   2.953   43.753
C   23.097   2.962   44.568
C   23.345   2.767   46.128
C   23.344   1.261   46.612
C   22.348   3.643   46.973
C   22.910   4.320   48.162
C   22.072   5.496   48.640
C   22.769   6.841   48.480
C   23.750   7.085   49.688
C   21.831   7.978   48.315
C   21.586   8.439   46.867
C   21.180   9.888   46.571
C   22.350   10.930   46.463
C   22.181   12.046   47.524
C   22.391   11.595   45.041
H   22.029   -0.143   31.867
H   23.063   -0.463   25.642
H   29.460   -0.753   26.609
H   26.071   -0.887   34.352
H   23.785   0.750   33.876
H   22.476   -1.355   33.954
H   23.648   -1.225   35.299
H   23.836   -2.272   33.786
H   26.821   1.818   33.666
H   27.368   0.770   35.003
H   24.682   1.561   35.575
H   25.412   2.969   34.939
H   19.303   0.997   29.554
H   20.350   1.064   30.832
H   19.720   -0.568   30.337
H   18.536   0.229   26.197
H   18.934   1.320   27.450
H   18.928   -0.451   27.766
H   25.369   0.368   24.388
H   27.312   -1.917   24.817
H   24.251   -2.439   24.446
H   25.472   -2.186   23.212
H   24.117   -1.160   23.294
H   28.259   0.874   24.859
H   27.592   0.319   23.338
H   29.758   -1.422   24.270
H   30.307   0.069   23.719
H   29.237   -0.897   22.632
H   31.318   -2.582   28.815
H   32.150   -1.075   29.227
H   31.247   -1.256   27.689
H   29.004   0.436   33.508
H   29.132   -3.122   36.067
H   28.120   -1.718   36.509
H   29.791   -1.779   37.052
H   26.658   3.534   39.259
H   27.108   1.752   39.327
H   24.352   2.313   40.011
H   26.966   1.006   43.097
H   27.366   2.617   42.900
H   27.766   1.605   41.489
H   24.669   0.892   43.235
H   23.527   1.741   42.181
H   24.472   3.888   43.208
H   25.230   2.831   44.370
H   22.454   2.187   44.150
H   22.561   3.909   44.501
H   24.383   3.076   46.246
H   23.876   1.194   47.561
H   23.968   0.619   45.990
H   22.354   0.807   46.613
H   21.578   3.008   47.411
H   21.868   4.423   46.381
H   23.894   4.513   47.735
H   23.045   3.552   48.924
H   21.810   5.404   49.694
H   21.085   5.571   48.185
H   23.447   6.740   47.632
H   23.746   8.093   50.103
H   24.775   6.771   49.490
H   23.450   6.460   50.529
H   22.099   8.808   48.969
H   20.794   7.738   48.550
H   20.775   7.816   46.489
H   22.457   8.094   46.309
H   20.595   10.097   47.467
H   20.617   9.898   45.638
H   23.307   10.509   46.771
H   22.074   11.499   48.461
H   21.257   12.575   47.292
H   23.131   12.565   47.655
H   21.429   11.460   44.546
H   23.259   11.202   44.512
H   22.381   12.682   45.119

