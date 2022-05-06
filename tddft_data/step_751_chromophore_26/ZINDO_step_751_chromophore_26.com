%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_751_chromophore_26 ZINDO

0 1
Mg   -9.515   18.275   42.767
C   -6.050   18.130   43.078
C   -9.173   21.699   42.090
C   -12.906   18.394   42.277
C   -9.617   14.807   42.742
N   -7.751   19.766   42.602
C   -6.459   19.373   42.599
C   -5.545   20.641   42.231
C   -6.604   21.789   42.092
C   -7.936   21.085   42.250
C   -6.392   22.971   43.150
C   -4.777   20.352   40.858
C   -5.315   19.280   39.930
C   -5.128   19.570   38.391
O   -4.169   20.152   37.886
O   -6.198   19.116   37.688
N   -10.832   19.810   42.273
C   -10.490   21.122   42.047
C   -11.737   21.882   41.894
C   -12.841   21.029   41.923
C   -12.219   19.669   42.126
C   -11.690   23.419   41.518
C   -14.297   21.412   41.745
O   -14.561   22.590   41.509
C   -15.392   20.442   41.864
N   -10.945   16.849   42.518
C   -12.355   17.108   42.400
C   -13.201   15.841   42.473
C   -12.127   14.704   42.443
C   -10.820   15.527   42.489
C   -14.133   15.842   43.685
C   -12.077   13.660   41.259
C   -12.610   12.280   41.546
N   -8.167   16.741   42.885
C   -8.358   15.360   42.930
C   -7.150   14.743   43.279
C   -6.168   15.820   43.337
C   -6.882   16.964   43.072
C   -6.952   13.326   43.653
C   -4.769   16.120   43.639
O   -3.835   15.346   43.859
C   -4.639   17.639   43.394
C   -4.031   18.214   44.558
O   -2.984   18.883   44.590
O   -4.842   17.991   45.583
C   -4.382   18.382   46.959
C   -5.845   19.149   36.253
C   -6.844   18.273   35.515
C   -7.623   18.625   34.425
C   -7.342   19.755   33.395
C   -8.873   17.778   34.092
C   -10.202   18.564   34.112
C   -10.829   18.966   32.719
C   -11.338   20.393   32.582
C   -12.915   20.552   32.255
C   -10.516   21.068   31.413
C   -10.406   22.592   31.640
C   -10.179   23.326   30.261
C   -11.174   24.409   29.822
C   -10.721   25.755   30.490
C   -11.317   24.557   28.270
C   -12.254   23.496   27.592
C   -11.771   22.853   26.223
C   -12.878   23.039   25.151
C   -12.620   21.950   24.027
C   -12.859   24.481   24.546
H   -9.153   22.774   41.899
H   -13.977   18.229   42.146
H   -9.729   13.724   42.837
H   -4.839   20.787   43.048
H   -6.432   22.392   41.200
H   -7.350   23.100   43.654
H   -6.098   23.931   42.726
H   -5.667   22.680   43.910
H   -3.841   19.993   41.287
H   -4.561   21.253   40.284
H   -6.386   19.126   40.059
H   -4.822   18.351   40.216
H   -12.420   23.964   42.117
H   -12.036   23.466   40.486
H   -10.745   23.961   41.537
H   -15.299   20.093   42.892
H   -15.369   19.632   41.136
H   -16.339   20.978   41.812
H   -13.796   15.754   41.564
H   -12.195   14.084   43.337
H   -14.572   16.833   43.795
H   -13.733   15.386   44.591
H   -14.945   15.155   43.450
H   -11.112   13.499   40.779
H   -12.719   14.048   40.468
H   -13.164   11.834   40.720
H   -13.250   12.279   42.428
H   -11.796   11.618   41.843
H   -7.386   12.693   42.879
H   -7.609   13.030   44.471
H   -5.893   13.226   43.888
H   -3.934   17.666   42.564
H   -3.512   19.039   46.976
H   -4.098   17.485   47.510
H   -5.199   18.910   47.451
H   -4.855   18.716   36.108
H   -5.889   20.180   35.902
H   -7.368   17.514   36.094
H   -6.403   20.258   33.626
H   -8.189   20.381   33.673
H   -7.430   19.293   32.412
H   -8.986   16.899   34.727
H   -8.697   17.442   33.070
H   -10.036   19.397   34.796
H   -10.856   17.866   34.634
H   -11.746   18.377   32.676
H   -10.189   18.618   31.908
H   -11.116   20.889   33.527
H   -13.447   19.635   32.510
H   -13.073   20.797   31.205
H   -13.439   21.260   32.897
H   -11.014   20.780   30.488
H   -9.498   20.689   31.318
H   -9.568   22.887   32.273
H   -11.318   23.022   32.054
H   -10.096   22.508   29.546
H   -9.201   23.800   30.182
H   -12.170   24.112   30.150
H   -9.878   25.553   31.152
H   -11.481   26.156   31.161
H   -10.460   26.586   29.835
H   -10.323   24.442   27.838
H   -11.776   25.504   27.985
H   -13.174   24.063   27.450
H   -12.410   22.686   28.306
H   -11.585   21.797   26.416
H   -10.852   23.358   25.923
H   -13.925   22.892   25.418
H   -12.501   20.912   24.339
H   -11.636   22.173   23.615
H   -13.312   22.008   23.187
H   -13.863   24.825   24.300
H   -12.200   24.499   23.677
H   -12.506   25.203   25.283

