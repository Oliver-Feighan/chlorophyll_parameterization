%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_251_chromophore_11 TDDFT with blyp functional

0 1
Mg   53.430   23.680   44.039
C   50.605   25.722   43.592
C   51.432   20.929   43.527
C   56.214   21.922   43.716
C   55.421   26.611   44.290
N   51.346   23.393   43.365
C   50.336   24.382   43.243
C   48.978   23.812   43.212
C   49.267   22.251   43.293
C   50.778   22.167   43.441
C   48.414   21.544   44.374
C   48.223   24.061   41.942
C   49.034   24.029   40.638
C   48.418   23.462   39.300
O   47.922   24.100   38.357
O   48.482   22.045   39.281
N   53.816   21.666   43.704
C   52.860   20.719   43.650
C   53.500   19.428   43.609
C   54.890   19.670   43.737
C   55.049   21.133   43.757
C   52.807   18.104   43.430
C   55.977   18.599   43.823
O   55.698   17.417   43.773
C   57.462   18.808   43.858
N   55.501   24.202   43.750
C   56.495   23.262   43.689
C   57.905   23.905   43.901
C   57.592   25.426   43.766
C   56.092   25.426   43.964
C   58.577   23.602   45.262
C   57.806   26.062   42.312
C   58.535   27.355   42.246
N   53.144   25.682   44.068
C   53.994   26.676   44.359
C   53.226   27.914   44.762
C   51.874   27.519   44.508
C   51.892   26.209   43.951
C   53.718   29.265   45.261
C   50.538   28.037   44.441
O   50.105   29.153   44.681
C   49.653   26.859   43.865
C   49.163   27.302   42.502
O   49.805   27.462   41.517
O   47.788   27.388   42.554
C   47.156   27.526   41.238
C   47.706   21.408   38.241
C   48.584   21.192   37.044
C   48.340   20.565   35.892
C   46.952   20.112   35.632
C   49.384   20.405   34.726
C   50.294   19.202   34.866
C   51.643   19.484   34.136
C   52.872   18.610   34.711
C   53.998   19.369   35.333
C   53.419   17.749   33.572
C   52.324   16.727   33.054
C   52.784   15.256   33.232
C   52.297   14.387   32.005
C   51.624   13.100   32.559
C   53.350   14.174   30.885
C   52.647   14.212   29.486
C   53.708   13.995   28.352
C   53.279   12.857   27.412
C   53.608   11.509   27.998
C   53.859   12.910   26.010
H   50.768   20.063   43.562
H   57.138   21.346   43.789
H   56.086   27.449   44.509
H   48.424   24.235   44.050
H   49.157   21.695   42.362
H   48.037   20.696   43.802
H   47.644   22.251   44.683
H   49.009   21.156   45.201
H   47.553   24.920   41.987
H   47.449   23.326   41.720
H   49.941   23.478   40.889
H   49.329   25.073   40.531
H   53.318   17.492   42.686
H   51.754   18.266   43.200
H   52.894   17.628   44.407
H   57.897   19.233   42.954
H   57.816   17.777   43.851
H   57.819   19.195   44.813
H   58.561   23.529   43.116
H   58.080   26.038   44.525
H   58.619   24.525   45.840
H   59.483   23.008   45.143
H   57.927   22.934   45.826
H   56.858   26.157   41.783
H   58.383   25.304   41.782
H   59.409   27.319   41.596
H   58.839   27.734   43.222
H   57.920   28.144   41.814
H   53.345   30.007   44.555
H   54.792   29.184   45.433
H   53.170   29.449   46.185
H   48.827   26.600   44.527
H   47.724   27.048   40.440
H   47.122   28.605   41.087
H   46.118   27.225   41.377
H   46.880   22.061   37.958
H   47.372   20.483   38.712
H   49.600   21.558   37.189
H   46.894   19.512   34.724
H   46.375   21.025   35.482
H   46.484   19.541   36.434
H   49.950   21.336   34.720
H   48.816   20.303   33.802
H   49.779   18.409   34.323
H   50.480   18.903   35.897
H   51.893   20.542   34.199
H   51.528   19.253   33.077
H   52.530   17.969   35.524
H   54.034   19.248   36.415
H   53.968   20.402   34.986
H   54.954   18.897   35.106
H   54.303   17.255   33.976
H   53.630   18.312   32.663
H   52.026   16.965   32.033
H   51.397   16.867   33.611
H   52.365   14.853   34.154
H   53.873   15.252   33.255
H   51.479   14.785   31.405
H   51.429   13.181   33.629
H   52.103   12.200   32.173
H   50.590   13.017   32.225
H   53.818   13.192   30.811
H   54.035   15.014   30.775
H   52.231   15.196   29.270
H   51.827   13.497   29.412
H   54.673   13.737   28.788
H   53.873   14.900   27.769
H   52.212   13.004   27.241
H   53.249   11.451   29.026
H   54.674   11.291   27.931
H   53.056   10.777   27.408
H   54.278   11.957   25.686
H   54.656   13.652   25.949
H   53.048   13.068   25.299

