%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_16 TDDFT with cam-b3lyp functional

0 1
Mg   40.405   41.300   26.790
C   39.532   43.623   29.433
C   41.017   38.991   29.205
C   41.776   39.473   24.428
C   40.204   44.097   24.587
N   40.275   41.355   29.142
C   39.817   42.342   29.943
C   39.995   41.890   31.424
C   40.351   40.348   31.273
C   40.589   40.172   29.770
C   41.437   40.006   32.257
C   38.652   42.119   32.235
C   38.367   41.250   33.508
C   37.635   41.846   34.711
O   36.415   41.712   34.898
O   38.433   42.726   35.436
N   41.137   39.379   26.764
C   41.342   38.660   27.890
C   42.022   37.430   27.476
C   42.302   37.483   26.041
C   41.750   38.840   25.682
C   42.363   36.404   28.503
C   43.074   36.500   25.131
O   43.179   36.621   23.918
C   43.712   35.304   25.808
N   40.949   41.780   24.866
C   41.344   40.774   24.028
C   41.430   41.221   22.582
C   40.498   42.427   22.580
C   40.568   42.781   24.122
C   42.900   41.478   22.059
C   39.002   42.281   22.107
C   38.170   41.073   22.521
N   39.994   43.394   26.898
C   39.909   44.373   25.970
C   39.585   45.621   26.520
C   39.414   45.369   27.853
C   39.676   44.067   28.077
C   39.594   46.957   25.785
C   39.088   45.991   29.154
O   38.764   47.133   29.402
C   39.159   44.886   30.205
C   40.125   45.308   31.303
O   41.312   45.299   31.146
O   39.510   45.550   32.477
C   40.344   45.949   33.529
C   37.910   43.446   36.505
C   38.523   42.849   37.710
C   37.865   42.919   38.901
C   36.524   43.620   39.127
C   38.398   42.065   40.082
C   37.689   40.701   40.282
C   38.774   39.680   40.646
C   38.307   38.542   41.521
C   36.877   38.066   41.193
C   38.524   38.811   43.032
C   39.989   38.695   43.433
C   40.173   37.713   44.709
C   39.831   38.459   46.023
C   41.072   38.843   46.777
C   38.871   37.624   46.978
C   37.401   37.783   46.599
C   36.547   36.496   46.785
C   35.968   36.125   45.393
C   36.077   34.612   45.106
C   34.620   36.759   45.126
H   41.236   38.221   29.947
H   42.263   38.848   23.677
H   40.099   44.870   23.822
H   40.840   42.308   31.971
H   39.541   39.700   31.608
H   41.430   40.508   33.224
H   42.319   40.205   31.648
H   41.359   38.929   32.407
H   37.765   41.978   31.616
H   38.592   43.187   32.445
H   39.294   40.851   33.920
H   37.823   40.361   33.188
H   43.137   36.847   29.130
H   42.675   35.437   28.109
H   41.629   36.089   29.246
H   44.300   35.639   26.663
H   44.439   34.803   25.169
H   42.933   34.605   26.115
H   41.039   40.409   21.970
H   40.821   43.319   22.043
H   43.213   42.518   21.980
H   43.250   40.947   21.174
H   43.637   41.189   22.808
H   39.018   42.170   21.023
H   38.374   43.152   22.293
H   37.400   41.467   23.183
H   38.787   40.389   23.104
H   37.697   40.623   21.648
H   40.274   47.097   24.944
H   39.895   47.576   26.631
H   38.584   47.201   25.457
H   38.114   44.835   30.511
H   41.167   46.582   33.198
H   40.763   45.138   34.126
H   39.665   46.575   34.108
H   36.820   43.411   36.488
H   38.181   44.497   36.404
H   39.357   42.156   37.596
H   35.768   42.900   38.813
H   36.428   44.527   38.531
H   36.266   43.736   40.180
H   38.376   42.643   41.005
H   39.403   41.862   39.711
H   37.260   40.362   39.339
H   37.003   40.855   41.115
H   39.694   40.128   41.022
H   38.986   39.219   39.682
H   38.960   37.692   41.321
H   36.246   38.522   41.955
H   36.697   36.991   41.212
H   36.637   38.355   40.170
H   37.911   38.117   43.606
H   38.293   39.840   43.310
H   40.506   39.622   43.679
H   40.518   38.190   42.625
H   41.152   37.232   44.697
H   39.367   37.003   44.526
H   39.286   39.391   45.870
H   40.975   39.887   47.074
H   41.991   38.558   46.265
H   41.118   38.250   47.691
H   38.878   37.954   48.017
H   39.033   36.550   46.883
H   37.327   38.222   45.605
H   37.000   38.581   47.224
H   35.685   36.746   47.403
H   37.002   35.714   47.393
H   36.515   36.492   44.524
H   36.837   34.354   44.368
H   35.079   34.336   44.766
H   36.175   34.023   46.017
H   33.982   36.137   44.498
H   34.826   37.553   44.408
H   34.065   36.968   46.040
