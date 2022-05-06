%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_51_chromophore_24 TDDFT with blyp functional

0 1
Mg   -0.178   43.604   24.882
C   1.745   43.253   27.803
C   -3.017   42.650   26.648
C   -1.825   43.637   22.046
C   2.819   44.115   23.047
N   -0.635   43.236   27.021
C   0.323   43.156   28.031
C   -0.354   42.862   29.418
C   -1.852   42.529   29.052
C   -1.903   42.804   27.470
C   -2.381   41.115   29.417
C   -0.216   44.157   30.307
C   -0.590   43.941   31.825
C   -0.856   42.503   32.335
O   -0.053   41.589   32.263
O   -2.101   42.357   32.872
N   -2.215   43.324   24.422
C   -3.201   42.946   25.292
C   -4.479   42.931   24.523
C   -4.104   43.167   23.185
C   -2.664   43.398   23.190
C   -5.790   42.514   25.222
C   -4.925   43.096   21.860
O   -4.495   43.401   20.755
C   -6.364   42.791   22.045
N   0.442   43.756   22.820
C   -0.444   43.790   21.876
C   0.148   44.120   20.496
C   1.630   44.503   20.815
C   1.612   44.100   22.346
C   0.074   43.053   19.476
C   1.837   45.998   20.594
C   1.012   46.948   21.495
N   1.858   43.541   25.259
C   2.889   43.776   24.442
C   4.112   43.669   25.249
C   3.748   43.450   26.611
C   2.349   43.392   26.524
C   5.469   43.962   24.813
C   4.216   43.259   27.923
O   5.381   43.297   28.348
C   2.908   43.186   28.758
C   2.997   42.069   29.676
O   2.721   40.895   29.372
O   3.509   42.375   30.865
C   3.760   41.406   31.937
C   -2.449   40.967   33.424
C   -3.577   41.219   34.474
C   -3.632   41.123   35.788
C   -2.420   40.832   36.721
C   -4.974   41.489   36.457
C   -5.487   42.924   36.376
C   -6.934   42.997   35.788
C   -8.034   42.960   36.944
C   -8.659   44.355   37.146
C   -9.162   41.906   36.673
C   -9.366   41.070   37.982
C   -10.666   40.235   37.734
C   -11.491   39.952   39.005
C   -11.967   38.434   39.129
C   -12.691   40.894   39.071
C   -12.280   42.383   39.466
C   -12.270   43.385   38.262
C   -13.304   44.501   38.283
C   -13.326   45.158   36.933
C   -13.231   45.407   39.462
H   -3.957   42.404   27.147
H   -2.238   43.788   21.047
H   3.834   44.334   22.710
H   0.203   42.064   29.908
H   -2.427   43.408   29.345
H   -1.505   40.696   29.912
H   -2.550   40.591   28.476
H   -3.288   41.179   30.018
H   -0.941   44.914   30.008
H   0.698   44.706   30.077
H   -1.415   44.562   32.176
H   0.272   44.318   32.374
H   -6.038   41.490   24.944
H   -6.599   43.219   25.030
H   -5.769   42.487   26.311
H   -6.831   43.605   22.600
H   -6.527   41.846   22.565
H   -6.892   42.679   21.098
H   -0.383   44.984   20.098
H   2.359   43.933   20.238
H   -0.397   42.138   19.836
H   0.971   42.772   18.925
H   -0.586   43.313   18.649
H   1.545   46.208   19.565
H   2.917   46.080   20.709
H   0.393   47.613   20.893
H   1.628   47.559   22.154
H   0.360   46.443   22.208
H   5.487   45.047   24.917
H   5.580   43.719   23.756
H   6.327   43.624   25.395
H   2.980   44.161   29.240
H   3.102   41.565   32.792
H   4.790   41.292   32.274
H   3.551   40.374   31.655
H   -1.628   40.408   33.871
H   -2.811   40.431   32.546
H   -4.512   41.497   33.987
H   -2.074   41.736   37.220
H   -1.578   40.483   36.122
H   -2.644   40.159   37.549
H   -4.866   41.148   37.486
H   -5.702   40.777   36.068
H   -4.794   43.605   35.881
H   -5.497   43.282   37.406
H   -7.017   42.179   35.073
H   -7.007   43.876   35.147
H   -7.631   42.785   37.941
H   -7.923   45.154   37.050
H   -9.125   44.498   38.121
H   -9.459   44.518   36.425
H   -8.808   41.201   35.921
H   -10.119   42.326   36.365
H   -9.315   41.684   38.881
H   -8.513   40.397   38.071
H   -10.291   39.295   37.329
H   -11.319   40.564   36.926
H   -10.837   40.078   39.868
H   -11.071   37.818   39.054
H   -12.614   38.053   38.339
H   -12.467   38.294   40.087
H   -13.409   40.482   39.781
H   -13.229   40.909   38.123
H   -11.395   42.440   40.101
H   -13.064   42.555   40.203
H   -12.488   42.704   37.440
H   -11.307   43.891   38.197
H   -14.250   43.964   38.364
H   -13.916   46.073   36.988
H   -13.784   44.431   36.262
H   -12.281   45.311   36.667
H   -12.297   45.316   40.017
H   -14.031   45.182   40.168
H   -13.406   46.426   39.120

