%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1751_chromophore_6 TDDFT with wB97XD functional

0 1
Mg   17.348   -1.664   27.040
C   16.683   0.509   29.698
C   19.165   -3.503   29.179
C   18.186   -3.566   24.447
C   15.881   0.719   24.815
N   18.040   -1.423   29.141
C   17.454   -0.568   30.051
C   17.736   -1.030   31.488
C   18.736   -2.209   31.338
C   18.645   -2.394   29.806
C   20.180   -1.970   31.946
C   16.470   -1.450   32.438
C   16.570   -1.529   33.981
C   17.944   -1.176   34.527
O   18.386   0.018   34.582
O   18.742   -2.244   34.968
N   18.457   -3.382   26.853
C   19.106   -4.031   27.853
C   19.779   -5.185   27.373
C   19.484   -5.236   25.933
C   18.675   -4.011   25.680
C   20.649   -6.019   28.247
C   19.936   -6.247   24.926
O   19.623   -6.215   23.707
C   20.755   -7.563   25.294
N   17.255   -1.339   24.933
C   17.559   -2.338   24.105
C   17.243   -2.013   22.647
C   16.356   -0.755   22.768
C   16.498   -0.407   24.242
C   18.625   -1.811   22.011
C   14.900   -1.082   22.376
C   14.421   -0.210   21.237
N   16.526   0.284   27.137
C   15.932   1.055   26.176
C   15.335   2.206   26.858
C   15.630   2.035   28.228
C   16.300   0.841   28.359
C   14.473   3.262   26.216
C   15.367   2.631   29.557
O   14.868   3.642   29.980
C   15.992   1.612   30.545
C   16.886   2.322   31.510
O   17.906   2.857   31.236
O   16.302   2.313   32.812
C   17.004   3.151   33.846
C   19.949   -2.053   35.692
C   19.554   -2.512   37.048
C   20.278   -2.363   38.131
C   21.596   -1.611   38.255
C   19.789   -2.901   39.455
C   19.681   -4.404   39.587
C   20.506   -4.947   40.820
C   21.082   -6.298   40.400
C   19.980   -7.324   40.430
C   22.274   -6.773   41.321
C   23.484   -7.408   40.547
C   23.533   -8.910   40.654
C   24.897   -9.474   41.227
C   25.075   -10.989   40.758
C   25.125   -9.387   42.751
C   26.303   -8.464   43.112
C   27.120   -8.841   44.373
C   28.534   -9.349   44.063
C   29.226   -9.790   45.368
C   28.564   -10.496   43.128
H   19.727   -4.141   29.864
H   18.377   -4.279   23.643
H   15.305   1.338   24.125
H   18.192   -0.262   32.112
H   18.285   -3.086   31.802
H   20.227   -1.084   32.579
H   20.945   -2.095   31.180
H   20.488   -2.783   32.605
H   16.157   -2.468   32.205
H   15.673   -0.779   32.118
H   16.289   -2.518   34.344
H   15.867   -0.768   34.320
H   20.096   -6.959   28.254
H   20.905   -5.751   29.272
H   21.656   -6.179   27.863
H   21.711   -7.251   25.714
H   20.877   -8.103   24.355
H   20.306   -8.048   26.161
H   16.691   -2.797   22.127
H   16.719   0.056   22.137
H   18.856   -0.767   21.799
H   18.659   -2.404   21.097
H   19.394   -2.091   22.732
H   14.206   -0.917   23.201
H   14.805   -2.145   22.153
H   14.484   0.864   21.409
H   13.372   -0.499   21.171
H   14.943   -0.433   20.306
H   14.477   4.199   26.772
H   13.506   2.779   26.078
H   14.874   3.566   25.249
H   15.167   1.132   31.072
H   16.894   2.704   34.834
H   16.551   4.140   33.925
H   18.075   3.273   33.682
H   20.385   -1.054   35.667
H   20.785   -2.646   35.321
H   18.636   -3.095   37.126
H   21.893   -1.226   37.280
H   22.357   -2.339   38.538
H   21.591   -0.818   39.002
H   18.800   -2.455   39.559
H   20.366   -2.501   40.289
H   19.994   -4.880   38.657
H   18.614   -4.599   39.687
H   19.905   -4.942   41.730
H   21.343   -4.262   40.950
H   21.406   -6.251   39.361
H   20.267   -8.122   41.114
H   19.831   -7.784   39.453
H   19.026   -6.860   40.683
H   21.918   -7.387   42.148
H   22.692   -5.914   41.846
H   24.445   -7.018   40.884
H   23.357   -7.216   39.482
H   23.330   -9.294   39.655
H   22.698   -9.203   41.291
H   25.675   -8.836   40.808
H   25.432   -11.072   39.732
H   24.081   -11.423   40.653
H   25.792   -11.576   41.331
H   25.103   -10.373   43.217
H   24.338   -8.825   43.253
H   25.863   -7.477   43.253
H   26.902   -8.334   42.211
H   26.516   -9.581   44.898
H   27.244   -8.088   45.151
H   29.019   -8.517   43.553
H   29.878   -10.657   45.264
H   28.445   -10.117   46.055
H   29.794   -8.937   45.738
H   28.216   -10.087   42.180
H   27.847   -11.214   43.525
H   29.543   -10.971   43.065

