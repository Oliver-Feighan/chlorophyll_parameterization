%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1351_chromophore_27 TDDFT with cam-b3lyp functional

0 1
Mg   -5.763   25.149   26.677
C   -3.906   26.797   29.013
C   -6.880   22.944   29.211
C   -7.248   23.329   24.293
C   -4.408   27.174   24.181
N   -5.316   24.816   28.876
C   -4.571   25.704   29.659
C   -4.934   25.495   31.140
C   -5.735   24.149   31.117
C   -5.983   23.941   29.638
C   -4.973   23.002   31.824
C   -5.846   26.687   31.691
C   -5.521   27.152   33.149
C   -6.330   26.525   34.247
O   -7.451   26.865   34.696
O   -5.498   25.655   34.977
N   -6.911   23.379   26.716
C   -7.236   22.641   27.862
C   -8.184   21.535   27.445
C   -8.204   21.675   26.054
C   -7.492   22.851   25.624
C   -8.826   20.487   28.345
C   -8.910   20.712   25.101
O   -8.747   20.867   23.929
C   -9.935   19.747   25.578
N   -5.744   25.214   24.448
C   -6.537   24.397   23.728
C   -6.618   24.849   22.234
C   -5.803   26.147   22.249
C   -5.209   26.178   23.675
C   -6.076   23.857   21.258
C   -6.704   27.283   21.880
C   -7.293   27.325   20.444
N   -4.428   26.572   26.558
C   -3.907   27.356   25.485
C   -3.033   28.452   25.994
C   -2.949   28.246   27.386
C   -3.828   27.113   27.672
C   -2.431   29.518   25.165
C   -2.381   28.670   28.674
O   -1.644   29.665   28.961
C   -2.978   27.798   29.763
C   -1.879   27.004   30.404
O   -1.075   26.331   29.847
O   -1.930   27.290   31.767
C   -0.967   26.386   32.447
C   -5.836   25.178   36.351
C   -6.971   24.103   36.342
C   -7.271   23.092   37.238
C   -6.534   22.976   38.586
C   -8.354   22.117   36.798
C   -9.738   22.623   37.133
C   -10.500   21.672   38.129
C   -11.755   21.102   37.525
C   -12.910   22.145   37.528
C   -12.067   19.708   38.156
C   -11.146   18.629   37.628
C   -10.524   17.719   38.720
C   -8.969   17.417   38.535
C   -8.191   18.440   39.412
C   -8.527   15.990   38.866
C   -7.518   15.349   37.886
C   -6.257   14.696   38.629
C   -4.848   15.355   38.333
C   -3.802   14.363   37.709
C   -4.288   16.071   39.562
H   -7.309   22.250   29.937
H   -7.868   22.892   23.508
H   -4.117   27.982   23.507
H   -4.080   25.445   31.815
H   -6.727   24.312   31.537
H   -4.624   22.221   31.149
H   -5.479   22.434   32.605
H   -4.061   23.464   32.204
H   -6.899   26.403   31.712
H   -5.666   27.590   31.108
H   -5.774   28.204   33.279
H   -4.465   27.023   33.383
H   -8.372   20.580   29.331
H   -8.461   19.524   27.987
H   -9.907   20.615   28.284
H   -10.797   20.281   25.976
H   -9.709   18.940   26.276
H   -10.355   19.261   24.697
H   -7.670   25.039   22.019
H   -5.020   26.059   21.496
H   -5.602   22.978   21.693
H   -5.371   24.269   20.537
H   -6.962   23.460   20.764
H   -6.200   28.216   22.134
H   -7.487   27.157   22.627
H   -7.018   26.401   19.935
H   -6.847   28.277   20.156
H   -8.369   27.499   20.447
H   -3.097   29.769   24.339
H   -1.503   29.167   24.713
H   -2.286   30.461   25.693
H   -3.587   28.427   30.412
H   -1.337   25.371   32.299
H   -0.907   26.623   33.509
H   0.031   26.405   32.009
H   -6.148   26.052   36.922
H   -4.888   24.848   36.776
H   -7.594   24.074   35.448
H   -6.760   22.014   39.046
H   -7.053   23.704   39.209
H   -5.475   23.235   38.613
H   -8.231   21.102   37.176
H   -8.328   21.980   35.717
H   -10.390   22.819   36.282
H   -9.553   23.551   37.674
H   -10.742   22.185   39.059
H   -9.954   20.801   38.492
H   -11.484   21.022   36.472
H   -13.810   21.681   37.930
H   -13.116   22.429   36.496
H   -12.671   22.989   38.175
H   -13.079   19.428   37.862
H   -11.960   19.934   39.216
H   -10.380   19.101   37.013
H   -11.754   17.986   36.991
H   -11.167   16.839   38.728
H   -10.709   18.192   39.684
H   -8.749   17.650   37.493
H   -7.448   17.851   39.950
H   -8.872   19.002   40.050
H   -7.790   19.265   38.824
H   -9.462   15.444   38.737
H   -8.258   15.788   39.902
H   -7.217   16.103   37.158
H   -7.992   14.580   37.278
H   -6.219   13.628   38.412
H   -6.415   14.762   39.706
H   -4.877   16.211   37.660
H   -4.378   13.807   36.970
H   -3.471   13.586   38.398
H   -2.904   14.896   37.395
H   -3.499   16.677   39.116
H   -3.891   15.344   40.270
H   -5.013   16.718   40.056
