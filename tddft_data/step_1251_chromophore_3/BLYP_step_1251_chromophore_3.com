%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1251_chromophore_3 TDDFT with blyp functional

0 1
Mg   2.134   8.433   26.869
C   2.571   10.546   29.554
C   2.751   5.834   29.092
C   2.416   6.231   24.176
C   2.368   11.136   24.741
N   2.595   8.270   29.087
C   2.543   9.270   30.006
C   2.541   8.727   31.479
C   2.896   7.260   31.279
C   2.629   7.064   29.742
C   4.361   6.911   31.577
C   1.362   9.160   32.404
C   0.857   8.209   33.503
C   1.818   7.560   34.511
O   2.961   7.967   34.730
O   1.282   6.497   35.111
N   2.537   6.335   26.675
C   2.769   5.455   27.731
C   2.957   4.111   27.100
C   2.732   4.206   25.707
C   2.573   5.624   25.475
C   3.214   2.853   27.970
C   2.863   3.134   24.642
O   2.858   3.358   23.420
C   3.329   1.650   24.991
N   2.144   8.645   24.783
C   2.246   7.606   23.864
C   2.195   8.123   22.375
C   2.114   9.680   22.533
C   2.263   9.849   24.126
C   3.465   7.706   21.613
C   0.919   10.386   21.806
C   1.109   11.488   20.765
N   2.469   10.410   27.044
C   2.535   11.434   26.111
C   2.596   12.749   26.714
C   2.629   12.489   28.078
C   2.519   11.020   28.219
C   2.613   14.126   26.079
C   2.680   12.994   29.372
O   2.772   14.152   29.764
C   2.518   11.792   30.381
C   3.650   12.009   31.325
O   4.842   11.920   31.162
O   3.187   12.383   32.577
C   4.149   12.432   33.675
C   2.054   5.972   36.252
C   1.271   4.949   36.974
C   1.568   4.420   38.229
C   2.650   4.864   39.149
C   0.560   3.451   38.764
C   0.910   2.015   38.434
C   0.999   1.100   39.763
C   1.892   -0.194   39.531
C   3.320   0.248   39.903
C   1.368   -1.333   40.377
C   2.045   -2.683   39.925
C   2.583   -3.557   41.089
C   4.068   -3.852   40.940
C   4.264   -5.283   41.344
C   4.996   -2.918   41.792
C   6.132   -2.232   41.083
C   7.488   -2.935   41.321
C   8.440   -2.023   42.207
C   9.136   -0.923   41.307
C   9.437   -2.961   42.857
H   2.921   5.035   29.817
H   2.431   5.665   23.242
H   2.274   12.062   24.170
H   3.375   9.330   31.838
H   2.268   6.587   31.864
H   5.005   7.786   31.664
H   4.778   6.297   30.779
H   4.468   6.334   32.495
H   0.536   9.459   31.758
H   1.684   10.046   32.950
H   0.493   7.339   32.957
H   0.100   8.668   34.139
H   4.087   2.266   27.688
H   2.355   2.192   28.085
H   3.567   3.202   28.940
H   3.202   0.971   24.148
H   2.927   1.197   25.898
H   4.412   1.700   25.106
H   1.262   7.852   21.882
H   3.017   10.153   22.148
H   4.097   6.995   22.147
H   4.131   8.569   21.597
H   3.193   7.293   20.642
H   0.252   10.762   22.582
H   0.292   9.709   21.226
H   0.411   11.235   19.968
H   2.124   11.445   20.372
H   0.910   12.439   21.259
H   2.537   14.035   24.996
H   3.624   14.457   26.318
H   1.790   14.722   26.473
H   1.553   11.822   30.886
H   4.256   11.528   34.274
H   3.729   13.128   34.401
H   5.082   12.934   33.418
H   2.402   6.725   36.959
H   2.964   5.448   35.961
H   0.301   4.672   36.562
H   2.851   5.926   39.007
H   3.580   4.321   38.982
H   2.341   4.683   40.178
H   -0.394   3.821   38.390
H   0.539   3.466   39.853
H   1.827   2.002   37.846
H   0.080   1.729   37.788
H   -0.012   0.850   40.085
H   1.404   1.731   40.553
H   1.922   -0.410   38.463
H   3.591   0.232   40.959
H   3.463   1.272   39.558
H   3.958   -0.472   39.390
H   0.300   -1.513   40.250
H   1.591   -1.033   41.400
H   2.723   -2.507   39.090
H   1.227   -3.197   39.421
H   1.992   -4.468   40.999
H   2.364   -3.103   42.055
H   4.346   -3.765   39.889
H   3.402   -5.573   41.944
H   5.242   -5.378   41.817
H   4.212   -5.848   40.414
H   5.430   -3.377   42.680
H   4.352   -2.198   42.297
H   6.334   -1.178   41.273
H   6.071   -2.403   40.008
H   8.048   -3.283   40.453
H   7.412   -3.861   41.892
H   7.926   -1.524   43.029
H   10.195   -0.847   41.552
H   8.755   0.095   41.381
H   9.113   -1.189   40.250
H   10.477   -2.649   42.770
H   9.358   -4.013   42.583
H   9.311   -2.910   43.939

