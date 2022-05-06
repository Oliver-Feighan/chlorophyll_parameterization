%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1001_chromophore_5 TDDFT with PBE1PBE functional

0 1
Mg   24.929   -8.041   46.769
C   27.266   -5.537   45.809
C   22.507   -6.356   44.896
C   22.915   -10.657   47.249
C   27.670   -9.851   48.207
N   24.944   -6.213   45.387
C   25.939   -5.291   45.261
C   25.516   -4.097   44.405
C   23.902   -4.101   44.590
C   23.787   -5.665   44.912
C   23.291   -3.169   45.632
C   25.996   -4.317   42.964
C   25.801   -3.160   42.000
C   26.293   -3.383   40.596
O   27.427   -3.217   40.246
O   25.245   -3.772   39.748
N   23.018   -8.516   46.031
C   22.179   -7.669   45.364
C   20.859   -8.324   45.275
C   20.972   -9.548   46.000
C   22.322   -9.631   46.463
C   19.692   -7.749   44.518
C   19.808   -10.622   46.195
O   18.682   -10.457   45.714
C   20.000   -11.894   46.983
N   25.305   -9.944   47.514
C   24.235   -10.848   47.651
C   24.687   -11.999   48.488
C   26.319   -11.804   48.471
C   26.446   -10.407   48.113
C   24.115   -11.856   49.872
C   27.198   -12.743   47.547
C   27.865   -13.987   48.261
N   26.965   -7.659   47.207
C   27.912   -8.482   47.758
C   29.161   -7.772   47.824
C   28.931   -6.600   47.053
C   27.608   -6.650   46.633
C   30.398   -8.163   48.493
C   29.593   -5.363   46.649
O   30.709   -4.864   46.893
C   28.534   -4.678   45.780
C   28.371   -3.203   46.223
O   28.896   -2.220   45.693
O   27.404   -3.065   47.276
C   27.099   -1.676   47.682
C   25.501   -3.634   38.285
C   24.941   -4.803   37.537
C   25.246   -5.013   36.234
C   25.969   -4.036   35.349
C   24.769   -6.274   35.508
C   23.338   -6.061   34.933
C   23.077   -6.575   33.521
C   23.217   -5.530   32.410
C   21.871   -5.245   31.691
C   24.236   -6.057   31.322
C   25.471   -5.142   31.124
C   26.432   -5.625   30.025
C   26.353   -4.815   28.699
C   27.714   -4.093   28.510
C   25.990   -5.733   27.473
C   25.531   -4.932   26.224
C   23.993   -4.972   25.977
C   23.285   -3.614   26.108
C   22.645   -3.486   27.523
C   22.170   -3.347   25.074
H   21.729   -5.666   44.564
H   22.319   -11.504   47.595
H   28.518   -10.413   48.603
H   25.930   -3.219   44.900
H   23.421   -3.968   43.621
H   23.778   -2.199   45.532
H   23.537   -3.526   46.632
H   22.217   -2.983   45.652
H   25.563   -5.185   42.467
H   27.072   -4.396   43.116
H   26.399   -2.370   42.454
H   24.776   -2.810   41.877
H   19.722   -6.660   44.482
H   18.733   -7.946   44.997
H   19.640   -8.256   43.554
H   20.320   -11.625   47.990
H   20.770   -12.485   46.487
H   19.044   -12.411   47.071
H   24.280   -12.949   48.142
H   26.722   -11.974   49.469
H   23.063   -12.139   49.914
H   24.036   -10.789   50.077
H   24.768   -12.246   50.654
H   28.002   -12.152   47.107
H   26.603   -13.095   46.704
H   27.768   -13.889   49.343
H   28.910   -14.151   47.999
H   27.232   -14.790   47.883
H   31.120   -8.649   47.837
H   30.160   -8.762   49.371
H   30.877   -7.240   48.820
H   28.861   -4.627   44.742
H   26.729   -1.076   46.850
H   27.997   -1.229   48.108
H   26.328   -1.936   48.408
H   26.559   -3.690   38.027
H   25.137   -2.673   37.922
H   24.212   -5.401   38.084
H   26.964   -4.454   35.195
H   26.158   -3.076   35.829
H   25.321   -3.876   34.488
H   24.847   -7.122   36.188
H   25.484   -6.521   34.724
H   23.186   -4.982   34.900
H   22.629   -6.502   35.634
H   22.125   -7.100   33.596
H   23.811   -7.379   33.466
H   23.566   -4.595   32.848
H   21.316   -6.138   31.404
H   22.117   -4.617   30.835
H   21.249   -4.691   32.395
H   23.795   -6.251   30.344
H   24.444   -7.096   31.581
H   25.980   -5.027   32.080
H   25.237   -4.103   30.892
H   26.096   -6.633   29.779
H   27.398   -5.641   30.529
H   25.551   -4.077   28.692
H   28.278   -4.623   27.743
H   28.457   -4.193   29.301
H   27.627   -3.028   28.295
H   25.199   -6.423   27.765
H   26.882   -6.284   27.173
H   26.027   -5.316   25.333
H   25.798   -3.889   26.398
H   23.624   -5.608   26.782
H   23.780   -5.489   25.041
H   24.024   -2.818   26.022
H   23.505   -3.609   28.181
H   22.036   -4.390   27.510
H   22.084   -2.552   27.565
H   21.191   -3.042   25.445
H   21.978   -4.198   24.421
H   22.521   -2.557   24.411

