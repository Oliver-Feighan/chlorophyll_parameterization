%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1051_chromophore_1 ZINDO

0 1
Mg   -2.122   17.921   26.572
C   -2.533   15.930   29.433
C   -3.130   20.557   28.322
C   -2.260   19.595   23.679
C   -1.536   14.998   24.698
N   -2.824   18.208   28.654
C   -2.881   17.286   29.697
C   -3.181   17.997   31.014
C   -3.635   19.444   30.526
C   -3.085   19.434   29.118
C   -5.255   19.604   30.497
C   -1.894   18.140   32.037
C   -2.067   17.633   33.516
C   -0.728   17.556   34.317
O   0.374   17.793   33.912
O   -1.024   17.408   35.654
N   -2.470   19.847   26.074
C   -2.929   20.792   26.974
C   -3.022   22.078   26.293
C   -2.502   21.824   24.978
C   -2.428   20.333   24.876
C   -3.508   23.341   26.918
C   -2.172   22.794   23.838
O   -1.847   22.434   22.662
C   -2.184   24.272   24.084
N   -1.869   17.399   24.486
C   -2.064   18.249   23.455
C   -2.138   17.562   22.113
C   -1.655   16.152   22.469
C   -1.668   16.204   24.029
C   -3.534   17.738   21.401
C   -0.396   15.866   21.698
C   0.895   16.116   22.490
N   -2.005   15.889   26.911
C   -1.720   14.764   26.059
C   -1.694   13.551   26.816
C   -2.017   14.000   28.137
C   -2.179   15.395   28.141
C   -1.476   12.163   26.325
C   -2.163   13.546   29.470
O   -2.042   12.396   29.963
C   -2.668   14.766   30.436
C   -1.681   14.839   31.550
O   -0.518   15.143   31.543
O   -2.267   14.409   32.730
C   -1.351   13.918   33.798
C   0.040   17.388   36.590
C   -0.297   18.279   37.827
C   -0.137   18.010   39.153
C   0.408   16.762   39.751
C   -0.561   19.085   40.153
C   0.514   19.985   40.739
C   0.637   21.399   40.033
C   1.225   22.567   40.907
C   0.546   23.864   40.375
C   2.755   22.524   40.830
C   3.284   23.384   41.965
C   3.757   22.476   43.074
C   3.925   23.332   44.296
C   5.404   23.052   44.838
C   2.820   22.995   45.380
C   1.458   23.678   45.108
C   0.284   22.699   44.880
C   -0.970   23.245   45.581
C   -1.070   22.781   47.054
C   -2.266   22.817   44.832
H   -3.116   21.498   28.876
H   -2.261   20.162   22.746
H   -1.301   14.203   23.988
H   -3.916   17.501   31.647
H   -3.132   20.218   31.106
H   -5.483   20.162   29.589
H   -5.566   20.230   31.333
H   -5.813   18.684   30.324
H   -1.577   19.176   32.157
H   -1.049   17.645   31.558
H   -2.561   16.663   33.568
H   -2.689   18.361   34.036
H   -4.097   22.981   27.762
H   -4.218   23.892   26.301
H   -2.749   24.107   27.076
H   -3.211   24.603   24.235
H   -1.829   24.918   23.281
H   -1.571   24.428   24.972
H   -1.427   18.092   21.479
H   -2.447   15.484   22.131
H   -4.054   16.795   21.236
H   -3.295   18.080   20.394
H   -4.227   18.392   21.930
H   -0.271   16.302   20.706
H   -0.579   14.817   21.468
H   0.642   16.544   23.460
H   1.572   16.759   21.927
H   1.422   15.177   22.659
H   -2.445   11.709   26.116
H   -0.874   11.685   27.098
H   -0.892   12.272   25.411
H   -3.689   14.575   30.766
H   -1.381   14.691   34.566
H   -0.337   13.659   33.492
H   -1.718   13.037   34.323
H   1.032   17.631   36.209
H   0.007   16.352   36.928
H   -0.830   19.220   37.692
H   -0.452   16.344   40.275
H   1.191   16.969   40.480
H   0.568   16.116   38.888
H   -0.917   18.452   40.966
H   -1.392   19.671   39.762
H   1.421   19.382   40.697
H   0.361   20.091   41.813
H   -0.306   21.590   39.520
H   1.345   21.319   39.209
H   0.841   22.418   41.916
H   -0.440   23.891   40.840
H   0.176   23.706   39.362
H   1.295   24.656   40.358
H   2.967   22.904   39.830
H   3.041   21.475   40.897
H   2.541   24.099   42.319
H   4.141   23.924   41.562
H   4.708   22.032   42.781
H   2.939   21.757   43.125
H   3.904   24.388   44.025
H   5.589   22.078   45.291
H   5.557   23.823   45.594
H   6.077   23.255   44.005
H   3.188   23.452   46.299
H   2.726   21.931   45.596
H   1.571   24.345   44.254
H   1.240   24.391   45.903
H   0.470   21.655   45.133
H   0.186   22.680   43.795
H   -0.816   24.323   45.529
H   -2.073   22.858   47.474
H   -0.507   23.571   47.553
H   -0.679   21.786   47.263
H   -3.004   22.248   45.397
H   -2.034   22.154   43.999
H   -2.592   23.813   44.532

