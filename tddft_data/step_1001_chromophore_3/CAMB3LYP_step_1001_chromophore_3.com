%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1001_chromophore_3 TDDFT with cam-b3lyp functional

0 1
Mg   1.861   8.220   25.957
C   2.504   10.180   28.922
C   2.403   5.513   27.960
C   1.361   6.543   23.387
C   2.049   11.280   24.239
N   2.484   7.936   28.252
C   2.545   8.837   29.282
C   2.395   8.142   30.652
C   2.662   6.668   30.264
C   2.477   6.681   28.726
C   4.028   6.083   30.545
C   0.979   8.394   31.191
C   0.913   8.716   32.644
C   2.139   8.353   33.512
O   3.056   9.121   33.833
O   2.005   7.094   34.026
N   2.017   6.252   25.671
C   2.221   5.288   26.560
C   2.144   4.012   25.961
C   1.757   4.266   24.693
C   1.719   5.771   24.518
C   2.418   2.741   26.642
C   1.399   3.332   23.576
O   1.025   3.806   22.555
C   1.399   1.863   23.599
N   1.877   8.830   24.030
C   1.554   7.885   23.116
C   1.276   8.504   21.745
C   1.267   9.990   22.072
C   1.805   10.101   23.510
C   2.433   8.164   20.737
C   -0.168   10.619   21.953
C   -0.390   11.661   20.819
N   2.042   10.368   26.469
C   2.135   11.442   25.654
C   2.242   12.694   26.428
C   2.334   12.226   27.737
C   2.319   10.802   27.674
C   2.331   14.141   25.947
C   2.558   12.683   29.132
O   2.786   13.790   29.537
C   2.482   11.314   30.003
C   3.715   11.305   30.874
O   4.741   10.666   30.755
O   3.341   12.159   31.976
C   4.193   12.000   33.124
C   2.860   6.615   35.059
C   2.044   5.863   36.094
C   2.442   5.468   37.340
C   3.869   5.776   37.913
C   1.554   4.523   38.187
C   1.496   3.077   37.633
C   1.261   1.942   38.679
C   2.499   1.089   38.816
C   3.715   1.767   39.460
C   2.144   -0.279   39.512
C   2.576   -1.600   38.732
C   3.013   -2.828   39.660
C   4.544   -2.887   39.960
C   4.953   -4.410   40.247
C   5.257   -1.808   40.916
C   6.863   -1.918   40.891
C   7.472   -0.552   41.176
C   9.025   -0.394   40.817
C   9.882   0.093   42.018
C   9.230   0.475   39.480
H   2.427   4.636   28.610
H   0.922   6.126   22.478
H   1.633   12.182   23.786
H   3.163   8.565   31.299
H   1.864   6.015   30.617
H   4.282   5.267   29.868
H   4.172   5.722   31.563
H   4.834   6.815   30.506
H   0.335   7.568   30.889
H   0.483   9.259   30.750
H   0.019   8.267   33.079
H   0.885   9.800   32.757
H   2.848   2.894   27.632
H   3.080   2.015   26.168
H   1.447   2.267   26.783
H   1.601   1.434   22.617
H   0.450   1.584   24.056
H   2.175   1.426   24.228
H   0.274   8.260   21.392
H   1.994   10.470   21.416
H   3.371   7.766   21.124
H   2.677   9.020   20.108
H   2.152   7.379   20.036
H   -0.392   11.183   22.859
H   -0.948   9.869   21.820
H   0.427   11.444   20.131
H   -0.504   12.633   21.298
H   -1.284   11.447   20.232
H   3.205   14.199   25.299
H   2.374   14.629   26.921
H   1.411   14.474   25.465
H   1.689   11.337   30.750
H   4.771   11.080   33.215
H   3.590   12.129   34.022
H   4.761   12.930   33.106
H   3.400   7.443   35.517
H   3.691   6.034   34.659
H   0.973   5.739   35.932
H   4.534   6.208   37.165
H   4.237   4.827   38.302
H   3.802   6.490   38.734
H   0.513   4.832   38.283
H   1.993   4.458   39.182
H   2.283   2.954   36.889
H   0.577   2.939   37.063
H   0.343   1.435   38.385
H   1.001   2.479   39.591
H   2.801   0.747   37.826
H   4.705   1.394   39.198
H   3.676   1.634   40.541
H   3.570   2.827   39.251
H   1.081   -0.345   39.742
H   2.654   -0.291   40.475
H   3.401   -1.273   38.099
H   1.801   -1.977   38.065
H   2.724   -3.680   39.045
H   2.508   -2.752   40.623
H   4.956   -2.788   38.955
H   4.093   -4.987   40.585
H   5.662   -4.340   41.072
H   5.434   -4.966   39.442
H   4.799   -1.944   41.896
H   5.014   -0.791   40.608
H   7.315   -2.391   40.020
H   7.083   -2.511   41.779
H   7.420   -0.442   42.259
H   6.882   0.290   40.814
H   9.368   -1.382   40.509
H   10.285   1.087   41.820
H   10.812   -0.458   42.156
H   9.331   0.046   42.957
H   9.743   -0.193   38.787
H   9.795   1.384   39.685
H   8.266   0.728   39.038

