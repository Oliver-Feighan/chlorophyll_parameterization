%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_951_chromophore_7 ZINDO

0 1
Mg   26.014   -0.317   29.267
C   27.908   -0.341   32.286
C   23.169   0.382   31.344
C   24.017   -0.179   26.507
C   28.719   -0.867   27.441
N   25.562   -0.106   31.556
C   26.548   -0.046   32.536
C   25.979   0.185   33.943
C   24.397   0.392   33.581
C   24.343   0.185   32.075
C   23.428   -0.537   34.377
C   26.778   1.259   34.743
C   26.035   1.968   35.881
C   26.732   1.881   37.170
O   27.877   2.180   37.450
O   25.886   1.289   38.129
N   23.868   -0.156   28.992
C   22.996   0.195   29.918
C   21.649   0.137   29.290
C   21.881   -0.084   27.920
C   23.328   -0.133   27.770
C   20.339   0.207   30.103
C   20.858   -0.281   26.756
O   21.197   -0.413   25.528
C   19.363   -0.122   27.124
N   26.288   -0.549   27.302
C   25.384   -0.351   26.315
C   25.912   -0.393   24.891
C   27.444   -0.595   25.225
C   27.512   -0.682   26.747
C   25.162   -1.473   23.928
C   28.291   0.557   24.636
C   29.634   0.006   24.104
N   27.889   -0.636   29.754
C   28.931   -0.832   28.888
C   30.152   -1.090   29.561
C   29.821   -0.868   30.945
C   28.404   -0.636   31.009
C   31.387   -1.503   29.055
C   30.404   -0.802   32.317
O   31.536   -0.861   32.740
C   29.155   -0.397   33.208
C   29.083   -1.475   34.156
O   28.687   -2.580   33.957
O   29.664   -1.031   35.338
C   29.279   -1.767   36.509
C   26.420   1.292   39.479
C   25.204   1.152   40.415
C   25.227   1.072   41.781
C   26.484   0.825   42.513
C   23.909   1.172   42.505
C   23.729   0.063   43.570
C   22.277   0.151   44.085
C   22.273   0.054   45.632
C   21.034   -0.775   46.206
C   22.205   1.566   46.224
C   23.189   1.962   47.367
C   22.515   2.241   48.679
C   22.842   3.719   49.194
C   22.495   3.861   50.678
C   22.024   4.848   48.464
C   22.919   6.011   47.877
C   22.311   7.442   47.964
C   22.485   8.347   46.741
C   22.594   9.883   47.154
C   21.517   8.007   45.623
H   22.313   0.646   31.969
H   23.347   -0.138   25.646
H   29.550   -1.056   26.758
H   26.180   -0.754   34.459
H   24.211   1.457   33.719
H   22.868   -1.149   33.670
H   22.642   0.058   34.842
H   24.083   -1.116   35.029
H   27.037   1.960   33.950
H   27.629   0.790   35.238
H   25.057   1.511   36.032
H   25.771   2.994   35.626
H   19.815   -0.748   30.108
H   19.790   0.997   29.590
H   20.351   0.524   31.146
H   18.797   -0.612   26.332
H   19.161   0.948   27.158
H   19.139   -0.702   28.020
H   25.737   0.597   24.471
H   27.747   -1.594   24.913
H   25.879   -2.195   23.538
H   24.814   -0.858   23.098
H   24.369   -2.048   24.407
H   28.514   1.353   25.346
H   27.709   1.007   23.831
H   30.296   -0.227   24.939
H   30.121   0.759   23.485
H   29.574   -0.874   23.464
H   31.984   -1.856   29.895
H   32.011   -0.638   28.830
H   31.351   -2.268   28.279
H   29.263   0.540   33.755
H   28.282   -1.598   36.913
H   29.939   -1.503   37.335
H   29.297   -2.837   36.297
H   26.726   2.268   39.857
H   26.992   0.377   39.631
H   24.237   1.351   39.955
H   26.527   -0.126   43.043
H   26.698   1.623   43.223
H   27.354   0.977   41.873
H   23.024   1.310   41.884
H   23.974   2.087   43.094
H   24.453   0.297   44.350
H   23.970   -0.926   43.181
H   21.681   -0.645   43.640
H   21.790   1.089   43.820
H   23.188   -0.389   46.024
H   20.190   -0.131   46.455
H   21.246   -1.203   47.186
H   20.537   -1.491   45.551
H   21.162   1.649   46.529
H   22.343   2.330   45.459
H   23.689   2.854   46.988
H   23.927   1.181   47.544
H   22.930   1.554   49.417
H   21.434   2.216   48.543
H   23.901   3.897   49.007
H   21.969   4.764   50.985
H   23.355   3.648   51.312
H   21.869   3.039   51.026
H   21.240   5.291   49.079
H   21.488   4.389   47.633
H   23.104   5.682   46.854
H   23.876   6.172   48.373
H   22.900   7.881   48.769
H   21.294   7.383   48.351
H   23.444   8.055   46.312
H   22.254   10.077   48.172
H   22.137   10.551   46.424
H   23.612   10.252   47.029
H   21.001   7.073   45.843
H   22.064   7.715   44.726
H   20.790   8.806   45.479

