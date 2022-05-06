%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1751_chromophore_3 ZINDO

0 1
Mg   0.711   7.963   25.875
C   1.455   9.946   28.626
C   1.152   5.227   27.839
C   0.192   6.099   23.151
C   0.819   10.858   23.869
N   1.459   7.619   28.016
C   1.479   8.584   28.944
C   1.549   7.967   30.322
C   1.771   6.476   30.104
C   1.527   6.371   28.536
C   3.180   5.965   30.488
C   0.273   8.254   31.134
C   0.457   8.898   32.469
C   1.649   8.323   33.237
O   2.709   8.854   33.394
O   1.301   7.166   33.864
N   0.596   5.983   25.612
C   0.650   5.007   26.531
C   0.320   3.733   25.960
C   0.101   3.968   24.550
C   0.377   5.395   24.387
C   0.285   2.487   26.719
C   -0.435   2.957   23.471
O   -0.462   3.230   22.279
C   -0.844   1.615   23.899
N   0.642   8.394   23.808
C   0.252   7.477   22.894
C   0.214   8.027   21.460
C   0.163   9.557   21.765
C   0.530   9.619   23.251
C   1.406   7.609   20.630
C   -1.294   10.202   21.689
C   -1.369   11.491   20.907
N   1.131   10.013   26.103
C   1.031   11.092   25.250
C   1.427   12.297   25.863
C   1.582   11.924   27.229
C   1.430   10.511   27.289
C   1.352   13.677   25.344
C   1.745   12.407   28.583
O   1.850   13.545   29.066
C   1.634   11.148   29.548
C   2.965   11.048   30.286
O   4.043   10.918   29.711
O   2.866   11.363   31.611
C   4.104   11.562   32.356
C   2.368   6.575   34.669
C   1.635   5.639   35.543
C   1.891   5.231   36.841
C   3.064   5.713   37.651
C   0.994   4.191   37.412
C   1.729   2.821   37.548
C   0.860   1.630   37.103
C   0.114   0.871   38.308
C   -1.375   0.556   38.128
C   0.899   -0.421   38.360
C   2.342   -0.370   38.956
C   3.280   -1.304   38.173
C   3.850   -2.383   39.135
C   5.291   -2.302   39.384
C   3.303   -3.776   38.693
C   2.848   -4.772   39.858
C   3.766   -5.960   40.080
C   3.017   -7.288   40.298
C   3.211   -7.917   41.649
C   3.308   -8.360   39.252
H   1.170   4.368   28.513
H   -0.036   5.516   22.257
H   1.016   11.715   23.221
H   2.388   8.490   30.781
H   0.940   5.929   30.548
H   3.654   6.713   31.124
H   3.830   5.969   29.613
H   3.143   5.020   31.029
H   -0.374   7.387   31.269
H   -0.381   8.889   30.537
H   -0.430   8.744   33.083
H   0.584   9.980   32.446
H   -0.508   1.774   26.493
H   0.418   2.559   27.799
H   1.191   1.993   26.366
H   -1.543   1.748   24.724
H   -0.125   0.862   24.221
H   -1.358   1.114   23.078
H   -0.590   7.513   20.932
H   1.018   9.997   21.252
H   2.357   7.600   21.162
H   1.572   8.338   19.836
H   1.224   6.620   20.209
H   -1.665   10.613   22.628
H   -1.988   9.393   21.460
H   -0.411   11.821   20.504
H   -1.707   12.317   21.532
H   -2.078   11.254   20.114
H   0.843   14.347   26.037
H   0.857   13.878   24.394
H   2.398   13.983   25.342
H   0.836   11.396   30.248
H   4.724   10.732   32.017
H   3.887   11.506   33.423
H   4.608   12.458   31.994
H   2.959   7.293   35.237
H   3.080   6.082   34.006
H   0.707   5.266   35.108
H   2.739   5.916   38.672
H   3.581   6.544   37.170
H   3.770   4.883   37.664
H   0.008   4.140   36.950
H   0.814   4.401   38.466
H   2.044   2.587   38.565
H   2.625   2.894   36.932
H   1.512   0.947   36.557
H   0.133   1.956   36.359
H   0.399   1.518   39.137
H   -1.916   1.056   38.932
H   -1.604   -0.510   38.137
H   -1.819   0.886   37.189
H   0.802   -0.820   37.351
H   0.437   -1.101   39.076
H   2.247   -0.485   40.035
H   2.683   0.645   38.753
H   4.139   -0.717   37.845
H   2.919   -1.809   37.277
H   3.538   -2.251   40.171
H   5.832   -2.393   38.442
H   5.697   -3.024   40.093
H   5.642   -1.324   39.714
H   3.995   -4.308   38.039
H   2.557   -3.626   37.912
H   1.792   -4.971   39.678
H   2.957   -4.141   40.740
H   4.456   -5.672   40.872
H   4.343   -6.226   39.194
H   1.936   -7.158   40.246
H   3.869   -7.258   42.216
H   3.564   -8.948   41.661
H   2.255   -7.906   42.172
H   2.964   -9.377   39.437
H   4.355   -8.573   39.038
H   2.852   -8.084   38.301

