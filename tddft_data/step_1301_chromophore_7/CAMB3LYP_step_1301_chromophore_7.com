%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1301_chromophore_7 TDDFT with cam-b3lyp functional

0 1
Mg   26.168   -0.767   29.431
C   28.145   -1.147   32.434
C   23.536   -0.372   31.447
C   24.448   -0.244   26.579
C   29.021   -1.665   27.508
N   25.898   -0.920   31.718
C   26.864   -0.821   32.728
C   26.220   -0.649   34.163
C   24.643   -0.729   33.826
C   24.680   -0.663   32.254
C   23.911   -1.966   34.360
C   26.801   0.576   34.951
C   27.372   0.382   36.301
C   26.489   -0.304   37.350
O   25.405   -0.813   37.208
O   27.035   -0.222   38.609
N   24.279   -0.309   29.056
C   23.263   -0.259   29.999
C   22.029   -0.062   29.352
C   22.250   0.064   27.997
C   23.711   -0.126   27.822
C   20.643   0.005   30.074
C   21.278   0.092   26.796
O   21.596   0.230   25.611
C   19.786   0.106   26.989
N   26.638   -1.078   27.270
C   25.801   -0.653   26.312
C   26.444   -0.605   24.905
C   27.872   -1.172   25.202
C   27.853   -1.427   26.738
C   25.636   -1.302   23.823
C   28.988   -0.142   24.719
C   30.316   -0.896   24.499
N   28.164   -1.244   29.833
C   29.200   -1.646   28.930
C   30.430   -1.690   29.733
C   30.056   -1.572   31.059
C   28.664   -1.355   31.091
C   31.812   -1.992   29.189
C   30.552   -1.588   32.436
O   31.713   -1.853   32.830
C   29.397   -1.322   33.324
C   29.281   -2.509   34.188
O   28.549   -3.454   33.874
O   30.041   -2.393   35.337
C   29.764   -3.438   36.361
C   26.253   -0.777   39.763
C   25.654   0.316   40.678
C   25.386   0.428   42.015
C   25.524   -0.769   42.936
C   24.618   1.611   42.518
C   24.248   1.694   44.015
C   22.758   1.348   44.301
C   22.540   0.832   45.712
C   21.369   -0.206   45.845
C   22.465   1.987   46.843
C   23.855   2.394   47.369
C   23.767   2.898   48.825
C   24.313   4.343   48.978
C   24.567   4.750   50.462
C   23.484   5.409   48.305
C   24.200   6.191   47.203
C   23.676   7.580   46.882
C   23.167   7.790   45.466
C   22.343   9.065   45.434
C   22.371   6.654   44.803
H   22.591   -0.272   31.984
H   23.965   0.051   25.645
H   29.943   -1.809   26.942
H   26.563   -1.539   34.690
H   24.186   0.218   34.112
H   23.460   -2.593   33.591
H   23.167   -1.556   35.042
H   24.612   -2.570   34.937
H   25.907   1.199   34.975
H   27.489   1.118   34.301
H   27.429   1.344   36.809
H   28.293   -0.200   36.331
H   20.222   0.993   29.887
H   20.743   -0.184   31.143
H   20.024   -0.782   29.642
H   19.639   1.041   27.529
H   19.411   -0.772   27.514
H   19.291   0.205   26.024
H   26.570   0.418   24.550
H   28.048   -2.101   24.661
H   26.131   -2.220   23.505
H   25.502   -0.610   22.991
H   24.635   -1.540   24.182
H   29.198   0.605   25.484
H   28.629   0.357   23.819
H   30.915   -0.675   25.382
H   30.708   -0.719   23.498
H   30.281   -1.984   24.439
H   31.988   -1.406   28.286
H   31.871   -3.017   28.825
H   32.521   -1.823   29.999
H   29.574   -0.436   33.934
H   29.603   -4.451   35.990
H   28.945   -3.216   37.045
H   30.582   -3.368   37.078
H   27.037   -1.252   40.353
H   25.580   -1.633   39.716
H   25.280   1.128   40.054
H   26.256   -0.647   43.734
H   25.829   -1.679   42.420
H   24.588   -0.930   43.470
H   23.716   1.514   41.914
H   25.228   2.486   42.295
H   24.356   2.682   44.462
H   24.955   1.074   44.566
H   22.452   0.539   43.637
H   22.161   2.209   44.002
H   23.384   0.199   45.985
H   20.836   -0.236   44.895
H   20.684   0.167   46.607
H   21.528   -1.242   46.144
H   21.769   1.765   47.651
H   22.077   2.880   46.352
H   24.108   3.259   46.755
H   24.637   1.651   47.215
H   24.364   2.214   49.429
H   22.751   2.825   49.213
H   25.353   4.308   48.651
H   25.559   5.184   50.590
H   24.529   3.854   51.081
H   23.895   5.401   51.022
H   23.099   6.055   49.094
H   22.587   5.068   47.787
H   24.136   5.555   46.320
H   25.231   6.373   47.508
H   24.381   8.324   47.252
H   22.799   7.718   47.516
H   24.041   7.936   44.830
H   21.466   9.083   46.081
H   22.043   9.356   44.427
H   23.004   9.892   45.694
H   21.374   7.034   44.584
H   22.198   5.794   45.450
H   22.822   6.359   43.856

