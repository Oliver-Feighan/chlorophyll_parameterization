%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1651_chromophore_6 TDDFT with wB97XD functional

0 1
Mg   17.493   -2.364   27.883
C   16.879   -0.212   30.550
C   19.569   -4.251   29.830
C   18.408   -4.002   25.118
C   15.681   0.010   25.847
N   18.276   -2.163   29.879
C   17.792   -1.229   30.858
C   18.287   -1.700   32.271
C   19.308   -2.806   32.006
C   19.001   -3.134   30.480
C   20.840   -2.436   32.182
C   17.037   -2.238   33.022
C   16.947   -2.153   34.543
C   18.086   -1.417   35.261
O   18.202   -0.216   35.526
O   19.086   -2.349   35.538
N   18.648   -3.991   27.524
C   19.458   -4.579   28.457
C   19.963   -5.824   27.855
C   19.658   -5.801   26.505
C   18.917   -4.541   26.308
C   20.850   -6.790   28.710
C   19.951   -6.856   25.437
O   19.552   -6.726   24.293
C   20.939   -8.043   25.664
N   17.119   -1.911   25.782
C   17.663   -2.820   24.817
C   17.343   -2.362   23.399
C   16.376   -1.153   23.643
C   16.366   -0.994   25.182
C   18.508   -1.950   22.427
C   14.977   -1.409   23.050
C   14.352   -0.287   22.134
N   16.464   -0.543   28.075
C   15.675   0.277   27.205
C   15.054   1.357   27.912
C   15.544   1.224   29.212
C   16.359   0.027   29.247
C   14.131   2.409   27.325
C   15.460   1.915   30.508
O   14.928   2.936   30.906
C   16.279   0.918   31.456
C   17.276   1.765   32.265
O   18.364   2.179   31.859
O   16.767   2.069   33.504
C   17.419   3.158   34.319
C   20.296   -1.687   36.054
C   21.090   -2.785   36.770
C   20.879   -3.443   37.953
C   19.581   -3.250   38.745
C   22.041   -4.260   38.533
C   21.761   -5.716   38.800
C   21.866   -6.132   40.349
C   22.930   -7.250   40.631
C   22.409   -8.673   40.331
C   23.553   -7.113   42.038
C   24.885   -7.779   42.341
C   24.802   -8.816   43.495
C   25.735   -10.038   43.352
C   24.958   -11.412   43.557
C   26.884   -10.087   44.379
C   28.076   -9.077   44.099
C   29.427   -9.730   44.282
C   29.908   -10.519   43.033
C   30.662   -9.617   42.085
C   30.621   -11.828   43.439
H   20.194   -4.909   30.438
H   18.588   -4.547   24.189
H   15.130   0.651   25.156
H   18.687   -0.844   32.813
H   19.096   -3.755   32.499
H   21.430   -3.128   32.784
H   20.950   -1.536   32.788
H   21.306   -2.358   31.200
H   16.960   -3.281   32.715
H   16.115   -1.790   32.651
H   16.993   -3.089   35.099
H   16.028   -1.682   34.891
H   20.889   -7.773   28.242
H   20.263   -6.894   29.623
H   21.852   -6.495   29.023
H   21.181   -8.370   24.652
H   20.397   -8.764   26.277
H   21.911   -7.892   26.133
H   16.932   -3.322   23.089
H   16.902   -0.304   23.207
H   19.389   -1.627   22.982
H   18.154   -1.154   21.771
H   18.851   -2.799   21.836
H   14.265   -1.491   23.872
H   14.987   -2.285   22.401
H   14.762   0.688   22.399
H   13.272   -0.141   22.142
H   14.665   -0.672   21.164
H   13.388   2.571   28.106
H   13.653   2.220   26.364
H   14.444   3.439   27.156
H   15.591   0.377   32.105
H   16.974   3.201   35.313
H   17.190   4.064   33.758
H   18.499   3.027   34.385
H   20.137   -0.871   36.759
H   20.864   -1.303   35.207
H   22.051   -2.900   36.269
H   19.861   -3.230   39.798
H   18.935   -4.094   38.507
H   19.150   -2.271   38.537
H   22.462   -3.706   39.372
H   22.878   -4.229   37.836
H   22.542   -6.280   38.289
H   20.774   -5.897   38.376
H   20.839   -6.406   40.588
H   22.087   -5.225   40.913
H   23.769   -7.135   39.945
H   21.926   -9.284   41.094
H   23.175   -9.323   39.908
H   21.634   -8.575   39.572
H   22.835   -7.319   42.831
H   23.581   -6.033   42.188
H   25.582   -6.965   42.545
H   25.396   -8.223   41.487
H   23.760   -9.100   43.646
H   25.122   -8.277   44.387
H   26.151   -10.008   42.345
H   24.688   -11.831   42.588
H   24.050   -11.349   44.157
H   25.614   -12.097   44.093
H   27.298   -11.094   44.417
H   26.390   -9.889   45.330
H   28.051   -8.212   44.762
H   28.157   -8.758   43.060
H   29.423   -10.418   45.128
H   30.164   -8.949   44.467
H   29.003   -10.955   42.610
H   31.450   -10.080   41.492
H   30.962   -8.662   42.516
H   29.879   -9.453   41.345
H   31.095   -12.175   42.521
H   29.959   -12.675   43.619
H   31.353   -11.602   44.215

