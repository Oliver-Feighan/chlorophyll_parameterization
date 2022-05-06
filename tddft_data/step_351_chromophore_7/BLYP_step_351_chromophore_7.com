%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_351_chromophore_7 TDDFT with blyp functional

0 1
Mg   25.360   0.518   29.725
C   27.385   0.153   32.534
C   22.634   0.841   31.785
C   23.335   0.288   26.855
C   28.066   -0.547   27.687
N   25.162   0.626   31.981
C   26.048   0.418   32.896
C   25.549   0.718   34.249
C   23.952   0.718   34.020
C   23.856   0.661   32.436
C   23.154   -0.481   34.603
C   26.180   2.073   34.797
C   27.117   1.820   36.001
C   27.159   2.721   37.307
O   27.432   3.905   37.325
O   26.733   2.040   38.441
N   23.346   0.610   29.362
C   22.352   0.803   30.341
C   21.090   0.999   29.742
C   21.263   0.699   28.343
C   22.678   0.467   28.145
C   19.834   1.183   30.579
C   20.203   0.561   27.302
O   20.471   0.130   26.211
C   18.817   1.036   27.561
N   25.638   -0.042   27.591
C   24.707   0.033   26.610
C   25.311   -0.295   25.217
C   26.809   -0.327   25.494
C   26.830   -0.301   27.027
C   24.769   -1.543   24.557
C   27.548   0.960   24.937
C   29.090   0.802   24.893
N   27.334   0.042   29.979
C   28.331   -0.398   29.046
C   29.542   -0.634   29.775
C   29.234   -0.427   31.139
C   27.902   -0.042   31.209
C   30.743   -1.148   29.104
C   29.757   -0.470   32.514
O   30.916   -0.598   32.945
C   28.558   -0.025   33.407
C   28.314   -1.205   34.339
O   27.618   -2.193   34.059
O   28.973   -0.935   35.496
C   28.774   -1.823   36.701
C   26.759   2.892   39.606
C   26.389   2.190   40.942
C   25.221   1.787   41.412
C   23.936   1.916   40.689
C   25.076   1.152   42.797
C   24.869   2.167   43.938
C   23.831   1.620   44.979
C   23.935   2.504   46.276
C   23.461   1.668   47.503
C   23.223   3.827   46.022
C   23.776   4.919   46.990
C   22.649   5.491   47.830
C   23.040   6.970   48.130
C   22.519   7.510   49.405
C   22.584   7.850   46.940
C   23.372   9.156   46.840
C   22.416   10.357   47.232
C   23.054   11.285   48.338
C   21.935   11.501   49.483
C   23.604   12.572   47.703
H   21.831   0.851   32.525
H   22.761   0.518   25.954
H   28.933   -0.838   27.090
H   26.006   -0.069   34.850
H   23.554   1.642   34.439
H   23.868   -1.231   34.943
H   22.346   -0.856   33.974
H   22.609   -0.085   35.460
H   25.370   2.773   35.003
H   26.789   2.493   33.997
H   28.114   2.101   35.661
H   27.109   0.783   36.336
H   19.491   2.203   30.401
H   19.926   1.169   31.665
H   19.055   0.476   30.295
H   18.349   0.369   28.285
H   18.223   0.999   26.648
H   18.736   2.003   28.058
H   25.115   0.545   24.551
H   27.310   -1.262   25.244
H   23.950   -1.934   25.161
H   25.491   -2.345   24.402
H   24.316   -1.292   23.598
H   27.352   1.839   25.552
H   27.182   1.105   23.921
H   29.575   1.556   25.513
H   29.483   0.954   23.888
H   29.477   -0.192   25.114
H   31.594   -0.499   29.309
H   30.834   -1.272   28.025
H   30.918   -2.108   29.589
H   28.818   0.903   33.915
H   27.834   -1.490   37.141
H   29.603   -1.922   37.401
H   28.588   -2.861   36.425
H   26.249   3.854   39.550
H   27.824   3.078   39.743
H   27.259   2.110   41.593
H   23.796   2.918   40.284
H   23.034   1.639   41.235
H   23.923   1.235   39.837
H   26.008   0.632   43.018
H   24.289   0.398   42.818
H   24.524   3.108   43.510
H   25.847   2.259   44.412
H   23.982   0.573   45.242
H   22.792   1.655   44.651
H   25.014   2.635   46.353
H   22.682   0.933   47.297
H   23.127   2.328   48.303
H   24.391   1.188   47.808
H   22.188   3.581   46.260
H   23.364   4.181   45.001
H   24.154   5.749   46.393
H   24.521   4.484   47.656
H   22.586   4.851   48.710
H   21.639   5.569   47.427
H   24.110   7.079   48.306
H   21.923   6.726   49.873
H   21.778   8.275   49.171
H   23.356   7.837   50.022
H   21.506   7.993   47.023
H   22.709   7.330   45.990
H   23.692   9.344   45.815
H   24.310   9.061   47.386
H   21.447   10.124   47.674
H   22.151   10.949   46.357
H   23.878   10.718   48.770
H   21.909   10.744   50.267
H   20.929   11.657   49.093
H   22.265   12.391   50.019
H   23.400   13.402   48.379
H   23.240   12.874   46.721
H   24.687   12.625   47.591

