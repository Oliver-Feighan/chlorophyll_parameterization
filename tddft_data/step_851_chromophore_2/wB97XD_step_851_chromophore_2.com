%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_851_chromophore_2 TDDFT with wB97XD functional

0 1
Mg   3.219   0.954   44.372
C   6.236   2.748   44.098
C   1.619   3.835   43.001
C   0.488   -0.746   44.196
C   5.092   -1.773   45.291
N   3.882   2.980   43.534
C   5.171   3.470   43.455
C   5.248   4.848   42.852
C   3.718   5.220   42.576
C   2.981   3.968   43.090
C   3.199   6.503   43.416
C   6.143   5.106   41.651
C   6.696   3.903   40.885
C   7.098   4.177   39.458
O   7.686   5.192   39.085
O   6.746   3.084   38.634
N   1.317   1.454   43.623
C   0.826   2.666   43.146
C   -0.606   2.532   42.985
C   -0.959   1.267   43.355
C   0.300   0.540   43.684
C   -1.367   3.783   42.580
C   -2.385   0.670   43.296
O   -3.374   1.341   42.929
C   -2.664   -0.839   43.542
N   2.941   -1.069   44.501
C   1.667   -1.452   44.564
C   1.570   -3.002   44.883
C   3.107   -3.372   45.227
C   3.770   -2.021   45.055
C   0.527   -3.309   46.020
C   3.754   -4.557   44.491
C   4.454   -5.613   45.260
N   5.276   0.512   44.612
C   5.854   -0.610   45.065
C   7.265   -0.409   45.277
C   7.460   0.900   44.922
C   6.195   1.421   44.440
C   8.358   -1.352   45.609
C   8.490   1.942   44.721
O   9.707   1.935   44.891
C   7.720   3.177   44.239
C   8.017   4.277   45.132
O   8.926   5.127   44.866
O   6.995   4.549   46.028
C   6.836   5.903   46.477
C   6.993   3.228   37.227
C   5.736   2.814   36.614
C   5.563   2.149   35.424
C   6.630   1.605   34.484
C   4.089   1.745   35.249
C   3.272   2.849   34.523
C   2.905   2.740   33.033
C   1.434   2.330   32.667
C   1.245   0.805   32.834
C   1.010   2.823   31.212
C   0.050   4.038   31.309
C   -1.025   3.856   30.171
C   -0.700   4.869   29.041
C   -1.139   6.260   29.512
C   -1.361   4.427   27.755
C   -0.387   4.280   26.602
C   -0.713   3.182   25.546
C   -1.272   3.827   24.211
C   -2.579   3.147   23.775
C   -0.229   4.114   23.092
H   1.007   4.695   42.720
H   -0.445   -1.295   44.339
H   5.622   -2.544   45.855
H   5.564   5.529   43.644
H   3.496   5.310   41.513
H   2.558   7.182   42.853
H   4.079   7.006   43.819
H   2.664   6.170   44.305
H   6.979   5.760   41.896
H   5.542   5.655   40.926
H   6.096   2.993   40.908
H   7.649   3.552   41.281
H   -2.371   3.480   42.283
H   -0.903   4.217   41.694
H   -1.386   4.449   43.442
H   -2.447   -1.036   44.592
H   -2.220   -1.488   42.787
H   -3.701   -1.133   43.383
H   1.295   -3.484   43.945
H   3.144   -3.601   46.292
H   0.333   -4.381   46.056
H   -0.406   -2.749   45.959
H   0.860   -2.972   47.002
H   4.522   -4.042   43.915
H   3.103   -5.004   43.740
H   4.046   -6.591   45.003
H   4.412   -5.486   46.342
H   5.524   -5.540   45.066
H   7.971   -2.310   45.264
H   8.619   -1.476   46.660
H   9.201   -1.075   44.976
H   8.271   3.445   43.338
H   5.884   6.064   46.982
H   6.899   6.551   45.603
H   7.638   6.138   47.177
H   7.752   2.462   37.066
H   7.312   4.211   36.881
H   4.859   3.068   37.208
H   7.642   1.965   34.669
H   6.390   2.205   33.607
H   6.615   0.525   34.336
H   3.599   1.466   36.182
H   4.150   0.802   34.705
H   3.735   3.824   34.674
H   2.397   2.805   35.172
H   3.601   2.063   32.539
H   3.196   3.695   32.595
H   0.786   2.739   33.443
H   2.048   0.291   33.362
H   1.091   0.337   31.861
H   0.219   0.677   33.181
H   0.707   1.989   30.580
H   1.943   3.253   30.848
H   0.660   4.937   31.220
H   -0.354   4.125   32.317
H   -1.970   4.153   30.627
H   -1.143   2.863   29.736
H   0.383   4.869   28.913
H   -0.307   6.960   29.442
H   -1.704   6.438   30.427
H   -1.832   6.785   28.854
H   -2.089   5.170   27.430
H   -1.922   3.505   27.906
H   0.584   4.077   27.053
H   -0.282   5.236   26.089
H   -1.499   2.576   25.997
H   0.177   2.556   25.480
H   -1.619   4.826   24.474
H   -2.618   2.081   23.999
H   -2.761   3.288   22.710
H   -3.376   3.630   24.343
H   -0.143   5.180   22.879
H   -0.301   3.493   22.200
H   0.782   3.898   23.436

