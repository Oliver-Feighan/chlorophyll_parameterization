%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_451_chromophore_7 TDDFT with wB97XD functional

0 1
Mg   25.468   0.974   29.314
C   27.465   0.557   32.196
C   22.768   1.434   31.158
C   23.703   1.056   26.357
C   28.195   -0.080   27.402
N   25.173   1.001   31.527
C   26.170   0.844   32.544
C   25.482   1.042   33.901
C   23.928   1.129   33.491
C   23.926   1.210   31.944
C   23.043   -0.089   33.912
C   26.107   2.327   34.532
C   26.766   2.183   35.902
C   26.157   3.182   37.023
O   25.644   4.292   36.860
O   26.111   2.477   38.226
N   23.537   1.310   28.903
C   22.515   1.449   29.783
C   21.249   1.511   29.144
C   21.466   1.367   27.697
C   22.934   1.269   27.586
C   19.920   1.710   29.897
C   20.486   1.221   26.582
O   20.815   1.084   25.380
C   18.963   1.315   26.820
N   25.949   0.711   27.180
C   25.057   0.786   26.149
C   25.749   0.572   24.723
C   27.217   0.352   25.135
C   27.122   0.329   26.673
C   25.153   -0.568   23.903
C   28.169   1.511   24.644
C   29.676   1.162   24.620
N   27.413   0.433   29.627
C   28.398   -0.063   28.771
C   29.600   -0.262   29.560
C   29.311   -0.053   30.861
C   27.967   0.407   30.866
C   30.897   -0.663   29.017
C   29.794   -0.151   32.198
O   30.936   -0.417   32.588
C   28.675   0.322   33.150
C   28.316   -0.689   34.186
O   27.595   -1.618   33.911
O   29.027   -0.570   35.356
C   28.744   -1.653   36.273
C   25.471   3.297   39.271
C   25.687   2.883   40.637
C   24.886   2.883   41.729
C   23.380   3.323   41.855
C   25.337   2.415   43.098
C   25.256   3.204   44.427
C   24.003   2.808   45.234
C   23.938   3.842   46.460
C   23.987   3.067   47.703
C   22.677   4.815   46.364
C   23.062   6.391   46.276
C   22.443   7.220   47.420
C   22.744   8.764   47.346
C   23.419   9.309   48.658
C   21.524   9.716   46.980
C   21.709   10.498   45.664
C   21.168   9.774   44.420
C   22.151   9.843   43.197
C   21.745   11.032   42.312
C   22.285   8.467   42.463
H   21.869   1.421   31.778
H   23.086   1.024   25.456
H   29.083   -0.458   26.892
H   25.685   0.156   34.504
H   23.615   2.087   33.907
H   23.643   -0.933   34.253
H   22.431   -0.380   33.059
H   22.289   0.143   34.665
H   25.344   3.105   34.576
H   26.814   2.769   33.829
H   27.856   2.187   35.872
H   26.582   1.156   36.217
H   19.972   1.658   30.985
H   19.269   0.911   29.541
H   19.409   2.594   29.515
H   18.618   2.287   27.173
H   18.630   0.562   27.534
H   18.435   1.128   25.885
H   25.769   1.495   24.143
H   27.583   -0.621   24.807
H   24.655   -1.324   24.511
H   25.990   -0.892   23.284
H   24.442   -0.182   23.174
H   28.155   2.272   25.424
H   27.890   1.871   23.654
H   29.928   1.251   23.563
H   29.706   0.096   24.844
H   30.276   1.609   25.413
H   30.892   -1.022   27.988
H   31.257   -1.480   29.643
H   31.589   0.178   29.037
H   28.942   1.252   33.653
H   29.582   -1.734   36.965
H   28.601   -2.647   35.850
H   27.869   -1.354   36.851
H   24.435   3.151   38.963
H   25.608   4.378   39.255
H   26.719   2.603   40.852
H   23.218   4.049   42.652
H   22.739   2.485   42.128
H   23.119   3.638   40.845
H   26.392   2.165   42.984
H   24.875   1.446   43.283
H   25.277   4.292   44.371
H   26.205   3.126   44.959
H   24.170   1.803   45.622
H   23.094   2.663   44.650
H   24.792   4.507   46.328
H   24.456   3.595   48.534
H   24.741   2.280   47.683
H   23.041   2.675   48.076
H   21.989   4.619   47.187
H   22.170   4.604   45.423
H   22.571   6.782   45.385
H   24.120   6.636   46.371
H   22.858   6.753   48.313
H   21.374   7.012   47.453
H   23.461   9.094   46.594
H   23.831   8.546   49.318
H   22.644   9.741   49.292
H   24.166   10.044   48.356
H   21.324   10.364   47.833
H   20.651   9.081   46.827
H   22.734   10.856   45.570
H   21.252   11.487   45.696
H   20.221   10.200   44.088
H   21.010   8.737   44.716
H   23.152   10.107   43.539
H   20.751   11.461   42.439
H   21.691   10.834   41.241
H   22.515   11.802   42.359
H   23.304   8.083   42.514
H   22.071   8.607   41.403
H   21.736   7.664   42.953

