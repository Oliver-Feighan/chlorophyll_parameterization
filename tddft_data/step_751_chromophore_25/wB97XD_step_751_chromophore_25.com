%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_751_chromophore_25 TDDFT with wB97XD functional

0 1
Mg   -2.590   34.454   27.228
C   -3.852   32.346   29.639
C   -0.998   36.227   29.550
C   -1.718   36.621   24.817
C   -4.387   32.577   24.755
N   -2.317   34.174   29.318
C   -2.873   33.206   30.148
C   -2.448   33.383   31.635
C   -1.589   34.724   31.611
C   -1.584   35.010   30.089
C   -2.226   35.824   32.494
C   -1.718   32.187   32.220
C   -2.364   31.552   33.496
C   -1.611   31.738   34.830
O   -0.973   30.899   35.472
O   -1.755   33.091   35.210
N   -1.490   36.103   27.163
C   -0.840   36.676   28.241
C   -0.095   37.834   27.843
C   -0.111   37.831   26.393
C   -1.142   36.866   26.064
C   0.600   38.741   28.819
C   0.628   38.714   25.407
O   0.359   38.709   24.226
C   1.725   39.681   25.813
N   -3.291   34.786   25.112
C   -2.615   35.636   24.323
C   -2.804   35.385   22.822
C   -3.482   33.963   22.837
C   -3.776   33.768   24.329
C   -3.716   36.485   22.076
C   -2.693   32.738   22.117
C   -3.134   32.563   20.665
N   -3.949   32.833   27.192
C   -4.472   32.162   26.135
C   -5.248   31.045   26.684
C   -5.106   31.130   28.089
C   -4.225   32.190   28.298
C   -6.094   30.058   25.892
C   -5.376   30.493   29.404
O   -5.955   29.469   29.667
C   -4.659   31.378   30.494
C   -5.699   32.058   31.391
O   -6.612   32.782   31.029
O   -5.488   31.606   32.623
C   -6.520   31.957   33.666
C   -1.035   33.484   36.422
C   -1.848   34.679   37.065
C   -1.574   35.668   37.945
C   -0.240   35.800   38.658
C   -2.477   36.833   38.347
C   -3.164   36.667   39.734
C   -2.933   38.010   40.520
C   -3.017   38.049   42.091
C   -3.264   39.509   42.620
C   -1.718   37.565   42.694
C   -1.837   37.218   44.228
C   -0.447   37.427   45.008
C   0.358   36.156   45.362
C   0.523   36.048   46.879
C   1.809   36.279   44.660
C   1.897   35.740   43.239
C   2.674   34.427   43.144
C   3.959   34.588   42.210
C   3.711   34.184   40.723
C   5.088   33.749   42.791
H   -0.448   36.807   30.294
H   -1.420   37.228   23.959
H   -4.921   31.851   24.137
H   -3.422   33.508   32.108
H   -0.545   34.652   31.917
H   -2.549   36.699   31.930
H   -1.528   36.113   33.280
H   -3.159   35.547   32.986
H   -0.656   32.378   32.378
H   -1.569   31.463   31.420
H   -2.402   30.496   33.228
H   -3.359   31.957   33.674
H   0.666   39.766   28.455
H   1.556   38.217   28.821
H   0.252   38.739   29.852
H   2.327   39.773   24.908
H   2.330   39.309   26.640
H   1.303   40.666   26.016
H   -1.812   35.430   22.373
H   -4.460   34.019   22.360
H   -3.043   37.138   21.520
H   -4.307   37.167   22.687
H   -4.438   35.932   21.473
H   -2.774   31.803   22.672
H   -1.621   32.930   22.161
H   -2.243   32.473   20.043
H   -3.809   33.383   20.418
H   -3.738   31.670   20.509
H   -5.864   29.992   24.828
H   -7.166   30.251   25.913
H   -5.921   29.065   26.305
H   -4.012   30.730   31.086
H   -6.324   32.991   33.947
H   -6.487   31.239   34.486
H   -7.498   31.926   33.187
H   -0.093   33.939   36.113
H   -0.791   32.739   37.179
H   -2.806   34.637   36.546
H   0.175   36.798   38.519
H   0.501   35.112   38.252
H   -0.192   35.370   39.659
H   -3.256   36.942   37.592
H   -1.829   37.709   38.359
H   -2.834   35.760   40.242
H   -4.232   36.569   39.537
H   -3.835   38.486   40.134
H   -2.064   38.639   40.327
H   -3.914   37.540   42.442
H   -3.335   40.146   41.739
H   -2.449   39.972   43.176
H   -4.184   39.622   43.194
H   -0.912   38.277   42.519
H   -1.421   36.629   42.221
H   -2.205   36.203   44.373
H   -2.603   37.879   44.633
H   -0.724   37.995   45.895
H   0.224   38.071   44.440
H   -0.149   35.226   45.103
H   1.567   35.979   47.184
H   0.037   35.132   47.217
H   0.104   36.900   47.414
H   2.589   35.967   45.355
H   2.186   37.299   44.574
H   2.334   36.604   42.738
H   0.887   35.495   42.912
H   2.006   33.637   42.802
H   2.982   34.184   44.161
H   4.290   35.626   42.201
H   3.903   35.091   40.150
H   2.675   33.861   40.617
H   4.332   33.355   40.383
H   4.758   32.851   43.313
H   5.479   34.253   43.674
H   5.827   33.533   42.019

