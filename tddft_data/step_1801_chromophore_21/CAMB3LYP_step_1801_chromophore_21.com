%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1801_chromophore_21 TDDFT with cam-b3lyp functional

0 1
Mg   15.638   51.459   25.913
C   17.229   50.353   28.676
C   13.067   52.665   27.879
C   13.968   52.121   23.062
C   17.996   49.857   23.881
N   15.241   51.589   28.069
C   16.087   51.136   29.030
C   15.695   51.563   30.395
C   14.225   52.090   30.187
C   14.128   52.169   28.649
C   13.136   51.084   30.748
C   16.621   52.659   30.952
C   17.021   52.706   32.432
C   16.088   51.839   33.239
O   16.341   50.710   33.611
O   14.925   52.489   33.498
N   13.845   52.352   25.524
C   12.931   52.758   26.469
C   11.717   53.121   25.784
C   11.955   53.039   24.365
C   13.307   52.493   24.238
C   10.497   53.461   26.460
C   11.029   53.309   23.332
O   11.317   52.995   22.157
C   9.660   53.991   23.561
N   16.067   51.206   23.817
C   15.222   51.538   22.846
C   15.806   51.459   21.374
C   17.101   50.702   21.678
C   17.022   50.526   23.242
C   14.895   50.602   20.375
C   18.354   51.412   21.121
C   19.260   50.691   20.055
N   17.405   50.429   26.216
C   18.222   49.843   25.303
C   19.164   49.071   26.087
C   18.803   49.254   27.374
C   17.736   50.100   27.425
C   20.196   48.165   25.613
C   19.166   48.889   28.705
O   19.952   48.020   29.055
C   18.226   49.669   29.604
C   17.671   48.773   30.534
O   16.606   48.199   30.550
O   18.546   48.753   31.628
C   18.023   48.088   32.820
C   13.958   51.537   34.050
C   12.636   52.214   34.320
C   12.271   52.901   35.485
C   13.187   53.321   36.657
C   10.805   53.354   35.643
C   10.233   54.536   34.902
C   9.972   55.709   35.804
C   8.466   56.183   35.774
C   8.313   57.675   35.396
C   7.843   56.028   37.154
C   6.285   55.987   37.177
C   5.623   57.325   37.830
C   4.079   57.605   37.346
C   3.951   59.125   37.116
C   3.048   57.187   38.434
C   1.566   57.633   37.967
C   0.594   56.516   38.401
C   -0.851   57.093   38.562
C   -1.418   56.923   39.988
C   -1.786   56.358   37.518
H   12.249   52.945   28.545
H   13.522   52.457   22.123
H   18.706   49.318   23.251
H   15.760   50.675   31.022
H   14.099   53.116   30.532
H   12.472   51.600   31.441
H   13.603   50.185   31.151
H   12.506   50.720   29.936
H   16.281   53.683   30.794
H   17.560   52.608   30.400
H   17.116   53.761   32.687
H   17.953   52.162   32.581
H   9.735   52.704   26.279
H   10.132   54.398   26.038
H   10.478   53.639   27.535
H   9.053   53.435   24.276
H   9.225   54.138   22.572
H   9.991   54.979   23.879
H   16.025   52.445   20.962
H   17.197   49.710   21.237
H   14.087   50.090   20.898
H   15.454   49.894   19.764
H   14.533   51.322   19.640
H   19.049   51.637   21.931
H   18.171   52.374   20.642
H   19.309   49.614   20.216
H   20.288   50.984   20.267
H   19.002   51.020   19.049
H   20.317   47.267   26.218
H   21.175   48.607   25.429
H   19.812   47.946   24.616
H   18.843   50.386   30.144
H   17.162   47.450   32.623
H   17.818   48.811   33.610
H   18.790   47.433   33.232
H   14.281   51.085   34.988
H   13.730   50.760   33.320
H   11.904   52.174   33.515
H   12.916   52.710   37.518
H   12.956   54.351   36.932
H   14.255   53.335   36.439
H   10.713   53.505   36.719
H   10.134   52.502   35.540
H   9.331   54.184   34.402
H   10.869   54.865   34.080
H   10.497   56.559   35.367
H   10.348   55.668   36.826
H   7.859   55.600   35.081
H   7.292   57.715   35.016
H   8.828   57.863   34.454
H   8.421   58.463   36.141
H   8.227   56.832   37.782
H   8.146   55.132   37.696
H   6.011   55.225   37.907
H   5.804   55.819   36.213
H   6.320   58.155   37.713
H   5.637   57.100   38.896
H   3.903   57.044   36.429
H   3.402   59.301   36.191
H   4.891   59.611   36.854
H   3.418   59.625   37.924
H   3.351   57.807   39.278
H   3.055   56.119   38.649
H   1.551   57.578   36.879
H   1.262   58.603   38.360
H   0.925   56.004   39.304
H   0.637   55.825   37.558
H   -0.749   58.157   38.350
H   -2.080   57.715   40.340
H   -0.618   56.845   40.724
H   -1.979   56.008   40.177
H   -2.672   56.081   38.091
H   -1.331   55.475   37.070
H   -2.109   57.069   36.757

