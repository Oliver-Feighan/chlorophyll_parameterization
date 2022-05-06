%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_851_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.405   24.681   27.176
C   -4.185   27.063   29.463
C   -6.043   22.591   29.910
C   -6.412   22.478   25.071
C   -4.509   26.853   24.594
N   -5.164   24.782   29.485
C   -4.673   25.873   30.130
C   -4.807   25.767   31.585
C   -5.059   24.244   31.708
C   -5.555   23.820   30.327
C   -3.794   23.412   32.104
C   -5.875   26.632   32.186
C   -5.506   27.201   33.527
C   -6.513   26.921   34.661
O   -7.690   27.166   34.691
O   -5.920   26.256   35.672
N   -6.239   22.860   27.401
C   -6.402   22.093   28.594
C   -6.946   20.790   28.262
C   -7.198   20.837   26.857
C   -6.649   22.129   26.375
C   -7.160   19.581   29.220
C   -7.866   19.870   25.940
O   -7.996   20.012   24.733
C   -8.529   18.651   26.523
N   -5.385   24.635   25.128
C   -5.750   23.510   24.471
C   -5.617   23.719   22.973
C   -5.159   25.167   22.796
C   -5.028   25.630   24.256
C   -4.844   22.638   22.209
C   -6.258   26.040   22.134
C   -5.847   27.347   21.663
N   -4.413   26.521   27.007
C   -4.175   27.297   25.908
C   -3.402   28.486   26.354
C   -3.427   28.461   27.764
C   -3.997   27.222   28.079
C   -2.707   29.489   25.491
C   -3.066   29.161   28.952
O   -2.469   30.181   29.144
C   -3.560   28.315   30.099
C   -2.312   27.944   30.861
O   -1.348   27.294   30.405
O   -2.458   28.449   32.171
C   -1.556   27.822   33.157
C   -6.852   25.971   36.768
C   -7.557   24.640   36.539
C   -7.633   23.548   37.355
C   -6.803   23.444   38.688
C   -8.564   22.391   37.142
C   -10.090   22.764   37.335
C   -10.866   21.879   38.380
C   -11.559   20.741   37.538
C   -13.083   21.129   37.312
C   -11.309   19.413   38.155
C   -10.120   18.613   37.423
C   -9.555   17.361   38.119
C   -8.019   17.351   38.203
C   -7.542   18.475   39.221
C   -7.441   15.917   38.545
C   -6.001   15.681   38.092
C   -5.046   15.100   39.165
C   -3.784   15.883   39.276
C   -2.952   15.670   37.985
C   -2.991   15.494   40.562
H   -6.215   21.910   30.747
H   -6.435   21.748   24.260
H   -4.099   27.584   23.894
H   -3.836   25.946   32.047
H   -5.708   24.006   32.551
H   -3.953   22.849   33.023
H   -2.894   24.020   32.204
H   -3.552   22.552   31.480
H   -6.774   26.025   32.294
H   -6.180   27.445   31.526
H   -5.467   28.283   33.402
H   -4.533   26.869   33.889
H   -6.913   19.957   30.213
H   -6.419   18.831   28.945
H   -8.178   19.225   29.381
H   -7.727   17.963   26.792
H   -9.133   18.153   25.764
H   -9.264   18.928   27.278
H   -6.656   23.776   22.647
H   -4.144   25.198   22.400
H   -4.467   21.863   22.876
H   -3.994   22.924   21.590
H   -5.516   22.144   21.507
H   -7.176   26.114   22.718
H   -6.505   25.492   21.224
H   -4.830   27.701   21.832
H   -6.527   28.003   22.208
H   -6.049   27.384   20.593
H   -2.524   30.340   26.147
H   -3.257   29.803   24.604
H   -1.743   29.026   25.282
H   -4.232   28.983   30.639
H   -1.599   28.399   34.081
H   -0.500   27.781   32.891
H   -1.957   26.823   33.323
H   -7.463   26.797   37.133
H   -6.044   25.790   37.477
H   -8.067   24.464   35.592
H   -6.416   22.431   38.576
H   -7.576   23.389   39.454
H   -6.036   24.201   38.850
H   -8.471   21.646   37.933
H   -8.455   22.033   36.118
H   -10.658   22.779   36.405
H   -10.093   23.801   37.673
H   -11.703   22.445   38.790
H   -10.344   21.456   39.238
H   -11.174   20.767   36.519
H   -13.744   20.749   38.091
H   -13.460   20.772   36.354
H   -13.228   22.204   37.200
H   -12.201   18.800   38.023
H   -11.087   19.485   39.220
H   -9.336   19.348   37.242
H   -10.275   18.229   36.414
H   -9.878   16.533   37.488
H   -10.015   17.014   39.044
H   -7.604   17.661   37.244
H   -7.045   19.322   38.748
H   -6.875   17.991   39.934
H   -8.417   18.910   39.703
H   -8.199   15.261   38.118
H   -7.536   15.871   39.630
H   -5.636   16.527   37.509
H   -6.210   14.853   37.415
H   -4.731   14.073   38.979
H   -5.567   14.950   40.110
H   -4.135   16.910   39.379
H   -2.877   16.572   37.379
H   -3.257   14.777   37.440
H   -1.920   15.490   38.286
H   -2.811   16.502   40.933
H   -2.059   14.959   40.382
H   -3.712   15.079   41.267

