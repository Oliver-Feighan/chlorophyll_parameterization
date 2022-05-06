%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_151_chromophore_21 TDDFT with PBE1PBE functional

0 1
Mg   16.195   52.573   25.260
C   17.667   51.237   28.054
C   13.283   53.113   27.201
C   14.407   53.812   22.611
C   18.409   50.997   23.217
N   15.588   52.334   27.395
C   16.441   51.903   28.388
C   15.791   52.142   29.760
C   14.294   52.334   29.424
C   14.348   52.651   27.918
C   13.295   51.212   29.665
C   16.496   53.207   30.601
C   16.813   52.771   32.062
C   15.988   52.063   33.014
O   16.219   50.902   33.405
O   14.974   52.956   33.347
N   14.287   53.609   25.070
C   13.289   53.627   25.954
C   12.143   54.390   25.370
C   12.490   54.616   24.012
C   13.795   53.952   23.830
C   10.899   54.710   26.080
C   11.647   55.241   22.945
O   12.072   55.383   21.780
C   10.251   55.742   23.246
N   16.306   52.344   23.276
C   15.568   53.063   22.338
C   16.162   53.031   20.943
C   17.402   52.108   21.109
C   17.434   51.817   22.652
C   15.269   52.572   19.779
C   18.730   52.652   20.579
C   19.422   53.800   21.345
N   17.712   51.252   25.509
C   18.601   50.715   24.614
C   19.519   49.851   25.323
C   19.250   50.119   26.716
C   18.185   50.970   26.782
C   20.555   49.008   24.756
C   19.602   49.771   28.103
O   20.437   48.974   28.501
C   18.728   50.741   28.970
C   18.316   49.938   30.206
O   17.407   49.164   30.355
O   19.328   50.171   31.138
C   19.320   49.541   32.444
C   14.176   52.508   34.519
C   12.893   53.437   34.547
C   12.359   54.061   35.645
C   13.001   54.145   37.085
C   11.092   54.807   35.403
C   10.212   55.476   36.493
C   8.748   55.311   36.348
C   7.987   56.637   36.459
C   7.842   56.971   37.993
C   6.597   56.467   35.764
C   6.512   57.131   34.381
C   5.635   58.430   34.398
C   6.232   59.679   35.285
C   5.399   59.784   36.600
C   6.511   61.020   34.529
C   7.886   61.083   33.762
C   9.055   61.467   34.669
C   10.319   60.583   34.657
C   10.391   59.913   35.998
C   11.669   61.293   34.289
H   12.412   53.209   27.852
H   14.041   54.534   21.878
H   19.080   50.546   22.483
H   15.887   51.184   30.271
H   13.902   53.198   29.961
H   13.300   50.915   30.714
H   13.560   50.273   29.178
H   12.313   51.552   29.338
H   15.900   54.095   30.812
H   17.389   53.493   30.045
H   17.191   53.625   32.624
H   17.767   52.273   31.888
H   10.788   55.776   25.884
H   11.038   54.722   27.161
H   10.016   54.129   25.812
H   10.230   56.512   24.018
H   9.623   54.937   23.627
H   9.787   56.082   22.321
H   16.541   54.039   20.776
H   17.266   51.139   20.629
H   15.863   51.904   19.155
H   14.975   53.421   19.162
H   14.417   52.089   20.258
H   18.552   52.873   19.527
H   19.537   51.921   20.511
H   19.791   54.542   20.637
H   20.273   53.464   21.937
H   18.803   54.249   22.122
H   20.655   48.161   25.434
H   21.418   49.673   24.795
H   20.445   48.716   23.711
H   19.321   51.577   29.341
H   19.922   50.115   33.149
H   19.720   48.529   32.389
H   18.315   49.406   32.844
H   14.899   52.562   35.333
H   13.790   51.489   34.498
H   12.343   53.417   33.606
H   12.359   53.775   37.885
H   13.271   55.175   37.317
H   13.889   53.515   37.138
H   10.441   54.029   35.005
H   11.294   55.506   34.591
H   10.547   56.513   36.447
H   10.469   55.153   37.501
H   8.426   54.554   37.063
H   8.487   54.831   35.405
H   8.546   57.440   35.979
H   6.807   56.971   38.337
H   8.367   57.911   38.160
H   8.249   56.131   38.555
H   5.829   56.801   36.462
H   6.399   55.403   35.635
H   6.018   56.473   33.666
H   7.525   57.329   34.031
H   4.598   58.207   34.647
H   5.478   58.708   33.356
H   7.264   59.439   35.544
H   5.813   60.594   37.201
H   5.651   58.898   37.183
H   4.321   59.860   36.460
H   6.627   61.845   35.231
H   5.670   61.276   33.884
H   7.728   61.845   32.999
H   8.150   60.084   33.416
H   8.708   61.655   35.685
H   9.289   62.508   34.444
H   10.312   59.705   34.011
H   10.908   58.957   35.913
H   9.429   59.748   36.482
H   10.941   60.495   36.738
H   12.360   61.028   35.089
H   11.441   62.344   34.111
H   12.033   60.802   33.387

