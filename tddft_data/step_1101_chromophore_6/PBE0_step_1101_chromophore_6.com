%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1101_chromophore_6 TDDFT with PBE1PBE functional

0 1
Mg   16.866   -2.477   27.317
C   16.208   -0.320   30.004
C   18.511   -4.579   29.372
C   17.586   -4.276   24.646
C   15.067   -0.087   25.213
N   17.472   -2.276   29.384
C   17.006   -1.387   30.321
C   17.445   -1.804   31.789
C   18.307   -3.145   31.562
C   18.084   -3.384   30.042
C   19.783   -3.131   31.969
C   16.317   -2.029   32.884
C   16.555   -1.797   34.407
C   18.047   -1.390   34.905
O   18.436   -0.281   35.163
O   18.792   -2.569   35.075
N   17.957   -4.166   27.032
C   18.507   -4.936   27.954
C   18.944   -6.216   27.366
C   18.794   -6.103   26.007
C   18.109   -4.807   25.832
C   19.434   -7.329   28.207
C   19.195   -7.119   24.893
O   19.077   -6.829   23.657
C   19.759   -8.506   25.169
N   16.373   -2.153   25.251
C   16.940   -3.027   24.372
C   16.636   -2.534   22.849
C   15.452   -1.557   23.098
C   15.623   -1.215   24.530
C   17.928   -1.877   22.390
C   14.041   -2.213   22.951
C   13.160   -1.606   21.904
N   15.945   -0.548   27.573
C   15.221   0.176   26.653
C   14.649   1.278   27.364
C   15.040   1.190   28.709
C   15.674   -0.035   28.800
C   13.980   2.330   26.716
C   14.972   1.764   30.055
O   14.447   2.820   30.417
C   15.731   0.811   30.947
C   16.756   1.585   31.740
O   17.870   1.873   31.339
O   16.111   2.149   32.875
C   16.872   3.261   33.523
C   19.907   -2.439   36.031
C   19.518   -3.386   37.152
C   19.293   -3.191   38.423
C   19.260   -1.776   39.025
C   18.889   -4.363   39.342
C   19.930   -5.310   40.080
C   19.610   -6.799   39.820
C   20.729   -7.592   39.110
C   20.299   -8.981   38.600
C   22.043   -7.588   39.827
C   21.935   -8.173   41.263
C   23.083   -9.195   41.336
C   24.579   -8.742   41.475
C   25.223   -9.166   40.151
C   25.218   -9.534   42.698
C   26.693   -9.056   42.970
C   27.675   -10.256   42.792
C   29.195   -9.749   43.052
C   29.882   -10.556   44.183
C   29.962   -9.771   41.684
H   19.000   -5.255   30.076
H   17.918   -4.835   23.769
H   14.360   0.596   24.737
H   18.141   -1.073   32.198
H   17.781   -3.980   32.026
H   19.943   -3.811   32.806
H   20.081   -2.132   32.286
H   20.384   -3.490   31.133
H   16.072   -3.082   32.747
H   15.370   -1.566   32.606
H   16.342   -2.680   35.009
H   15.866   -0.988   34.650
H   20.286   -7.806   27.723
H   18.667   -8.078   28.407
H   19.906   -6.937   29.108
H   19.147   -9.038   25.896
H   20.801   -8.404   25.472
H   19.718   -9.045   24.222
H   16.401   -3.323   22.135
H   15.580   -0.618   22.559
H   17.798   -0.796   22.331
H   18.147   -2.207   21.375
H   18.776   -2.074   23.045
H   13.541   -2.391   23.903
H   14.341   -3.189   22.569
H   12.089   -1.586   22.104
H   13.319   -2.175   20.988
H   13.515   -0.585   21.768
H   13.678   3.073   27.454
H   13.167   1.959   26.091
H   14.723   2.763   26.046
H   15.093   0.380   31.719
H   17.552   2.758   34.211
H   16.259   3.910   34.148
H   17.594   3.856   32.965
H   20.098   -1.446   36.437
H   20.724   -2.801   35.407
H   19.521   -4.418   36.799
H   18.966   -1.669   40.070
H   18.573   -1.149   38.458
H   20.215   -1.252   38.968
H   18.123   -4.908   38.790
H   18.251   -3.911   40.101
H   19.744   -5.076   41.128
H   20.944   -4.984   39.850
H   18.681   -6.937   39.267
H   19.366   -7.262   40.776
H   20.809   -6.942   38.239
H   21.084   -9.683   38.882
H   20.368   -8.968   37.513
H   19.292   -9.246   38.923
H   22.347   -6.558   40.016
H   22.811   -8.101   39.248
H   20.949   -8.549   41.533
H   22.286   -7.395   41.942
H   22.768   -9.838   40.514
H   22.714   -9.801   42.163
H   24.478   -7.659   41.555
H   25.923   -9.990   40.290
H   25.821   -8.327   39.796
H   24.531   -9.398   39.341
H   25.176   -10.589   42.427
H   24.555   -9.531   43.564
H   26.795   -8.649   43.976
H   27.025   -8.286   42.274
H   27.549   -10.823   41.870
H   27.493   -11.071   43.493
H   29.227   -8.716   43.397
H   30.330   -9.824   44.854
H   30.650   -11.197   43.750
H   29.152   -11.162   44.720
H   30.805   -10.461   41.655
H   30.333   -8.781   41.419
H   29.323   -10.068   40.851

