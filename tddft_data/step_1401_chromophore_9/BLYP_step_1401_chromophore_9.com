%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1401_chromophore_9 TDDFT with blyp functional

0 1
Mg   35.994   1.008   30.241
C   33.400   1.866   32.436
C   38.213   1.029   32.987
C   38.376   0.023   28.087
C   33.771   1.402   27.608
N   35.774   1.444   32.422
C   34.668   1.641   33.100
C   34.953   1.661   34.578
C   36.452   1.770   34.719
C   36.861   1.343   33.309
C   36.976   3.190   35.171
C   34.214   0.456   35.240
C   33.636   0.668   36.698
C   34.044   1.893   37.485
O   33.502   3.007   37.526
O   35.178   1.655   38.225
N   37.963   0.611   30.479
C   38.732   0.737   31.642
C   40.108   0.240   31.318
C   40.176   -0.006   29.936
C   38.779   0.089   29.469
C   41.232   0.203   32.390
C   41.396   -0.385   29.103
O   41.328   -0.620   27.893
C   42.792   -0.456   29.678
N   36.094   0.803   28.178
C   37.174   0.401   27.497
C   36.989   0.459   25.957
C   35.524   0.972   25.903
C   35.059   1.060   27.321
C   38.053   1.419   25.226
C   34.627   -0.025   25.048
C   34.164   -1.454   25.489
N   34.001   1.607   30.002
C   33.165   1.657   28.885
C   31.879   2.270   29.255
C   31.958   2.398   30.696
C   33.229   1.886   31.089
C   30.662   2.567   28.395
C   31.238   2.756   31.830
O   30.054   3.171   31.880
C   32.089   2.342   33.033
C   32.190   3.408   34.087
O   33.059   4.252   33.993
O   31.172   3.364   34.952
C   31.020   4.502   35.831
C   35.772   2.858   38.802
C   37.085   2.588   39.389
C   37.464   1.885   40.465
C   36.389   1.177   41.342
C   38.938   1.504   40.620
C   39.831   1.868   41.823
C   40.187   0.673   42.718
C   41.680   0.255   42.510
C   41.764   -1.117   41.797
C   42.560   0.362   43.781
C   44.066   0.647   43.590
C   44.487   2.018   44.038
C   44.490   3.143   42.933
C   45.984   3.148   42.371
C   44.011   4.449   43.471
C   43.229   5.336   42.438
C   44.167   6.221   41.640
C   43.638   7.707   41.495
C   43.656   8.161   39.977
C   44.527   8.731   42.325
H   38.887   1.083   33.844
H   39.131   -0.169   27.322
H   33.057   1.517   26.790
H   34.538   2.630   34.853
H   36.778   1.103   35.518
H   36.177   3.579   35.803
H   37.114   3.893   34.350
H   37.843   3.077   35.822
H   34.896   -0.393   35.294
H   33.399   0.156   34.581
H   34.012   -0.188   37.258
H   32.551   0.659   36.591
H   41.873   -0.676   32.447
H   40.929   0.481   33.399
H   41.866   1.026   32.060
H   42.773   -1.338   30.318
H   43.156   0.455   30.153
H   43.487   -0.631   28.857
H   37.015   -0.527   25.492
H   35.374   1.944   25.432
H   37.538   2.304   24.850
H   38.501   0.969   24.340
H   38.814   1.757   25.928
H   35.297   -0.186   24.203
H   33.817   0.536   24.582
H   34.401   -1.702   26.523
H   34.575   -2.223   24.834
H   33.078   -1.543   25.519
H   30.951   2.710   27.354
H   30.114   3.451   28.722
H   30.084   1.645   28.470
H   31.661   1.532   33.623
H   30.810   5.359   35.191
H   31.931   4.656   36.411
H   30.266   4.280   36.586
H   35.130   3.336   39.542
H   35.968   3.555   37.987
H   37.955   2.876   38.798
H   36.396   0.093   41.223
H   35.377   1.551   41.183
H   36.560   1.299   42.411
H   39.541   1.868   39.787
H   38.945   0.432   40.424
H   39.154   2.517   42.378
H   40.722   2.469   41.645
H   39.523   -0.180   42.584
H   39.959   1.027   43.723
H   42.149   0.979   41.844
H   42.566   -1.149   41.059
H   40.899   -1.395   41.194
H   41.868   -1.972   42.466
H   42.559   -0.660   44.160
H   42.031   1.116   44.363
H   44.251   0.516   42.524
H   44.639   -0.091   44.151
H   45.512   1.840   44.364
H   43.842   2.329   44.860
H   43.839   2.806   42.126
H   46.368   4.150   42.560
H   45.988   3.035   41.287
H   46.627   2.325   42.683
H   44.849   5.061   43.804
H   43.346   4.120   44.270
H   42.551   5.955   43.025
H   42.734   4.698   41.706
H   44.110   5.801   40.636
H   45.170   6.290   42.060
H   42.609   7.772   41.849
H   44.152   7.453   39.313
H   44.094   9.139   39.779
H   42.609   8.014   39.711
H   45.523   8.354   42.552
H   44.078   9.066   43.260
H   44.524   9.696   41.818

