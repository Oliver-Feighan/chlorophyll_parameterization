%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_301_chromophore_1 ZINDO

0 1
Mg   -1.222   17.169   26.589
C   -1.711   14.985   29.217
C   -2.527   19.796   28.449
C   -1.509   19.034   23.854
C   -1.394   14.333   24.549
N   -1.865   17.347   28.568
C   -1.849   16.402   29.557
C   -2.154   16.960   30.952
C   -2.720   18.338   30.593
C   -2.383   18.477   29.105
C   -4.144   18.607   31.099
C   -0.925   17.020   31.896
C   -1.070   17.524   33.338
C   -0.362   16.787   34.490
O   0.427   15.892   34.387
O   -0.836   17.297   35.705
N   -1.540   19.202   26.248
C   -2.172   20.100   27.131
C   -2.260   21.419   26.526
C   -1.844   21.219   25.177
C   -1.638   19.736   25.051
C   -2.703   22.776   27.252
C   -1.720   22.267   24.180
O   -1.363   21.953   23.069
C   -1.835   23.761   24.422
N   -1.306   16.698   24.494
C   -1.462   17.687   23.539
C   -1.481   17.183   22.091
C   -1.233   15.600   22.345
C   -1.388   15.534   23.883
C   -2.821   17.555   21.341
C   0.116   15.094   21.759
C   1.485   15.521   22.464
N   -1.535   15.091   26.826
C   -1.577   14.039   25.884
C   -1.586   12.815   26.488
C   -1.718   13.120   27.870
C   -1.651   14.523   27.962
C   -1.700   11.545   25.922
C   -1.750   12.559   29.237
O   -1.741   11.393   29.626
C   -1.803   13.761   30.141
C   -0.651   13.693   31.096
O   0.557   13.852   30.853
O   -1.104   13.118   32.284
C   -0.046   12.696   33.112
C   -0.330   16.611   36.939
C   -0.403   17.584   38.094
C   -0.650   17.197   39.347
C   -0.550   15.706   39.751
C   -0.890   18.328   40.392
C   0.431   18.821   41.057
C   0.864   20.190   40.569
C   0.791   21.304   41.650
C   -0.618   21.630   42.068
C   1.667   22.540   41.359
C   2.545   22.878   42.526
C   4.032   22.458   42.488
C   4.813   22.936   43.731
C   6.225   23.581   43.406
C   4.958   21.883   44.891
C   4.425   22.250   46.270
C   3.074   21.479   46.689
C   1.883   22.483   46.616
C   2.030   23.502   47.755
C   0.516   21.733   46.570
H   -2.823   20.510   29.220
H   -1.707   19.673   22.991
H   -1.376   13.453   23.904
H   -2.895   16.320   31.431
H   -2.159   19.019   31.233
H   -4.561   17.684   31.501
H   -4.683   18.822   30.177
H   -4.131   19.340   31.905
H   -0.243   17.666   31.342
H   -0.478   16.026   31.923
H   -2.121   17.661   33.593
H   -0.605   18.510   33.324
H   -3.392   23.296   26.586
H   -1.851   23.420   27.471
H   -3.126   22.559   28.233
H   -1.387   24.088   25.361
H   -2.880   24.040   24.283
H   -1.197   24.271   23.701
H   -0.617   17.668   21.637
H   -2.084   15.047   21.947
H   -3.445   18.082   22.063
H   -3.313   16.681   20.915
H   -2.560   18.295   20.585
H   0.121   15.600   20.793
H   0.103   14.023   21.557
H   1.865   16.282   21.783
H   2.052   14.622   22.709
H   1.186   16.055   23.366
H   -1.299   11.580   24.909
H   -2.757   11.310   25.797
H   -1.247   10.736   26.494
H   -2.684   13.731   30.783
H   0.711   13.410   33.435
H   0.474   11.791   32.797
H   -0.413   12.490   34.118
H   0.692   16.307   36.711
H   -0.984   15.761   37.132
H   -0.610   18.622   37.836
H   -0.326   15.046   38.913
H   -1.470   15.393   40.244
H   0.143   15.612   40.587
H   -1.419   17.850   41.218
H   -1.501   19.052   39.854
H   1.239   18.122   40.841
H   0.153   18.837   42.110
H   0.418   20.344   39.586
H   1.934   20.082   40.390
H   1.265   20.820   42.504
H   -1.271   20.913   41.571
H   -0.914   22.632   41.757
H   -0.669   21.608   43.157
H   1.018   23.412   41.273
H   2.203   22.526   40.409
H   2.002   22.543   43.409
H   2.757   23.943   42.624
H   4.553   22.791   41.591
H   3.867   21.386   42.595
H   4.278   23.782   44.161
H   7.024   23.302   44.094
H   6.057   24.655   43.491
H   6.623   23.313   42.427
H   5.973   21.551   45.108
H   4.473   20.925   44.707
H   4.237   23.323   46.218
H   5.129   22.073   47.083
H   3.187   21.067   47.692
H   2.758   20.653   46.051
H   1.805   23.026   45.675
H   1.860   24.429   47.207
H   3.063   23.575   48.098
H   1.321   23.258   48.546
H   0.253   21.506   47.603
H   0.460   20.895   45.875
H   -0.197   22.534   46.375
