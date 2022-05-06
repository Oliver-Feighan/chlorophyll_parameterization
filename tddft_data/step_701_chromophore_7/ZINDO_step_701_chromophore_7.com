%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_701_chromophore_7 ZINDO

0 1
Mg   25.484   -0.730   29.493
C   27.760   -0.812   32.261
C   22.964   -0.486   31.651
C   23.494   -0.654   26.819
C   28.243   -1.433   27.482
N   25.517   -0.735   31.744
C   26.509   -0.496   32.618
C   25.907   -0.307   34.026
C   24.346   -0.298   33.752
C   24.247   -0.573   32.277
C   23.536   -1.395   34.582
C   26.386   1.019   34.711
C   26.149   1.078   36.239
C   27.331   1.026   37.198
O   28.418   1.528   36.978
O   26.876   0.751   38.438
N   23.484   -0.554   29.289
C   22.599   -0.500   30.280
C   21.265   -0.315   29.720
C   21.453   -0.356   28.277
C   22.902   -0.615   28.067
C   20.079   -0.213   30.653
C   20.417   -0.355   27.147
O   20.675   -0.437   25.978
C   18.967   -0.283   27.624
N   25.796   -1.233   27.387
C   24.834   -0.980   26.466
C   25.422   -0.975   25.066
C   26.896   -1.156   25.260
C   27.042   -1.248   26.813
C   24.728   -2.036   24.093
C   27.821   -0.037   24.617
C   29.068   -0.431   23.899
N   27.571   -1.204   29.806
C   28.560   -1.499   28.829
C   29.837   -1.651   29.485
C   29.563   -1.437   30.861
C   28.173   -1.112   30.980
C   31.111   -1.906   28.847
C   30.109   -1.374   32.198
O   31.229   -1.605   32.613
C   28.974   -0.944   33.154
C   28.799   -1.942   34.204
O   28.126   -3.017   34.030
O   29.287   -1.460   35.447
C   28.825   -2.365   36.579
C   27.818   0.678   39.512
C   27.228   1.328   40.707
C   27.716   1.324   41.915
C   29.008   0.547   42.371
C   27.190   2.175   43.057
C   25.804   1.774   43.628
C   24.711   2.876   43.343
C   23.909   3.218   44.584
C   22.744   2.189   44.605
C   23.370   4.701   44.443
C   23.565   5.487   45.818
C   22.237   5.876   46.509
C   22.315   5.902   48.104
C   21.872   4.607   48.900
C   21.512   7.083   48.579
C   22.209   8.457   48.662
C   21.614   9.566   47.717
C   22.375   9.682   46.362
C   23.209   10.983   46.295
C   21.513   9.435   45.118
H   22.123   -0.357   32.336
H   22.915   -0.423   25.922
H   28.994   -1.552   26.698
H   26.132   -1.116   34.720
H   23.912   0.681   33.958
H   23.999   -1.719   35.514
H   23.428   -2.352   34.071
H   22.498   -1.071   34.660
H   25.838   1.898   34.369
H   27.432   1.089   34.415
H   25.470   0.272   36.518
H   25.449   1.896   36.405
H   19.226   0.340   30.260
H   20.335   0.322   31.568
H   19.806   -1.226   30.949
H   18.285   -0.470   26.794
H   18.689   0.616   28.174
H   18.937   -1.084   28.363
H   25.246   -0.024   24.562
H   27.175   -2.160   24.942
H   25.494   -2.771   23.844
H   24.247   -1.660   23.190
H   23.939   -2.691   24.463
H   28.182   0.581   25.440
H   27.257   0.557   23.898
H   29.004   -1.410   23.424
H   29.889   -0.352   24.611
H   29.267   0.277   23.094
H   30.962   -1.807   27.772
H   31.467   -2.918   29.045
H   31.897   -1.249   29.219
H   29.247   -0.011   33.647
H   29.224   -1.941   37.501
H   29.255   -3.363   36.493
H   27.758   -2.586   36.623
H   28.731   1.214   39.255
H   28.083   -0.349   39.760
H   26.218   1.730   40.632
H   29.351   -0.171   41.627
H   28.720   -0.106   43.195
H   29.730   1.286   42.719
H   27.201   3.229   42.778
H   27.907   2.293   43.869
H   25.733   1.463   44.670
H   25.489   0.870   43.106
H   24.104   2.561   42.495
H   25.284   3.737   43.000
H   24.633   3.171   45.398
H   21.958   2.544   45.272
H   22.996   1.257   45.110
H   22.280   1.901   43.662
H   22.303   4.655   44.221
H   23.769   5.125   43.522
H   23.933   6.447   45.453
H   24.297   5.044   46.493
H   21.520   5.065   46.389
H   21.686   6.703   46.061
H   23.357   6.015   48.401
H   22.537   4.608   49.763
H   21.980   3.745   48.242
H   20.839   4.717   49.232
H   21.159   6.942   49.600
H   20.592   7.019   47.998
H   23.240   8.260   48.370
H   22.262   8.798   49.696
H   21.559   10.498   48.279
H   20.572   9.272   47.593
H   23.137   8.926   46.174
H   22.779   11.787   46.893
H   23.204   11.330   45.262
H   24.267   10.834   46.512
H   22.130   8.913   44.387
H   21.322   10.390   44.628
H   20.622   8.881   45.413
