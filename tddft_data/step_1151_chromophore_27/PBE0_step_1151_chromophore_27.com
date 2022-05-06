%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1151_chromophore_27 TDDFT with PBE1PBE functional

0 1
Mg   -5.989   25.172   26.457
C   -4.141   26.906   28.917
C   -7.207   23.093   28.851
C   -7.658   23.640   24.016
C   -4.216   26.963   23.975
N   -5.772   25.129   28.717
C   -4.941   25.901   29.466
C   -5.066   25.545   30.928
C   -5.840   24.104   30.883
C   -6.295   24.094   29.386
C   -4.936   22.802   31.185
C   -5.850   26.661   31.822
C   -5.425   26.838   33.289
C   -6.550   26.514   34.251
O   -7.397   27.396   34.635
O   -6.350   25.345   34.903
N   -7.140   23.515   26.408
C   -7.571   22.852   27.537
C   -8.369   21.737   27.105
C   -8.476   21.751   25.694
C   -7.752   22.965   25.275
C   -9.028   20.858   28.154
C   -9.172   20.853   24.680
O   -9.214   21.027   23.462
C   -9.885   19.608   25.189
N   -5.890   25.215   24.272
C   -6.898   24.703   23.515
C   -7.039   25.346   22.075
C   -5.931   26.424   22.186
C   -5.267   26.187   23.511
C   -6.722   24.269   20.972
C   -6.374   27.882   21.898
C   -6.596   28.226   20.450
N   -4.438   26.596   26.380
C   -3.817   27.259   25.297
C   -2.785   28.208   25.771
C   -2.882   28.046   27.156
C   -3.924   27.130   27.549
C   -1.958   29.088   24.992
C   -2.363   28.590   28.454
O   -1.534   29.436   28.598
C   -3.197   27.919   29.604
C   -2.241   27.270   30.461
O   -1.452   26.375   30.158
O   -2.311   27.798   31.658
C   -1.462   27.172   32.660
C   -7.114   25.131   36.203
C   -8.122   23.916   36.040
C   -8.240   22.777   36.768
C   -7.385   22.344   37.996
C   -9.360   21.837   36.399
C   -10.777   22.302   36.815
C   -11.526   21.308   37.763
C   -12.476   20.462   36.835
C   -13.828   21.176   36.549
C   -12.638   18.988   37.387
C   -11.571   17.936   36.978
C   -11.068   17.086   38.213
C   -9.532   16.935   38.120
C   -8.868   18.212   38.870
C   -9.060   15.561   38.635
C   -7.612   15.261   38.116
C   -6.471   15.173   39.180
C   -5.298   14.170   38.884
C   -4.807   13.454   40.160
C   -4.105   14.791   38.106
H   -7.727   22.401   29.515
H   -8.237   23.259   23.172
H   -3.670   27.414   23.144
H   -4.078   25.385   31.359
H   -6.817   24.137   31.363
H   -5.034   21.979   30.476
H   -5.289   22.267   32.067
H   -3.880   23.065   31.240
H   -6.858   26.256   31.909
H   -5.986   27.608   31.300
H   -5.072   27.851   33.478
H   -4.542   26.242   33.520
H   -9.940   21.327   28.524
H   -8.333   20.696   28.978
H   -9.472   19.925   27.805
H   -10.553   19.146   24.462
H   -10.587   19.981   25.935
H   -9.162   18.864   25.523
H   -8.006   25.801   21.857
H   -5.164   26.208   21.443
H   -6.816   23.274   21.407
H   -5.697   24.314   20.605
H   -7.480   24.364   20.194
H   -5.715   28.578   22.417
H   -7.376   28.013   22.307
H   -6.614   27.299   19.877
H   -5.874   28.950   20.073
H   -7.561   28.674   20.212
H   -1.318   29.630   25.687
H   -2.625   29.665   24.352
H   -1.340   28.414   24.398
H   -3.737   28.764   30.032
H   -0.452   27.359   32.294
H   -1.697   26.127   32.859
H   -1.637   27.751   33.567
H   -7.607   26.021   36.594
H   -6.391   24.857   36.971
H   -8.636   24.037   35.087
H   -6.651   23.126   38.191
H   -6.976   21.334   38.000
H   -8.043   22.365   38.864
H   -9.217   20.838   36.809
H   -9.402   21.750   35.313
H   -11.404   22.478   35.941
H   -10.654   23.248   37.342
H   -12.118   21.930   38.435
H   -10.860   20.562   38.195
H   -11.985   20.343   35.869
H   -14.607   20.668   37.117
H   -14.064   21.219   35.486
H   -14.010   22.184   36.921
H   -13.640   18.706   37.061
H   -12.758   19.108   38.463
H   -10.831   18.530   36.441
H   -12.029   17.246   36.268
H   -11.553   16.112   38.138
H   -11.457   17.540   39.124
H   -9.378   17.023   37.045
H   -9.672   18.944   38.945
H   -8.016   18.525   38.266
H   -8.600   17.953   39.895
H   -9.587   14.759   38.118
H   -8.998   15.362   39.705
H   -7.480   15.993   37.319
H   -7.634   14.298   37.606
H   -6.942   14.956   40.139
H   -5.945   16.128   39.208
H   -5.709   13.334   38.318
H   -3.817   13.847   40.392
H   -4.696   12.376   40.039
H   -5.462   13.572   41.023
H   -4.408   15.343   37.216
H   -3.517   13.934   37.777
H   -3.555   15.464   38.763

