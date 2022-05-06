%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1101_chromophore_4 ZINDO

0 1
Mg   9.071   4.001   27.934
C   10.042   2.268   30.889
C   7.481   6.378   29.995
C   7.489   5.207   25.179
C   10.283   1.207   25.996
N   8.810   4.290   30.195
C   9.344   3.449   31.164
C   9.072   4.040   32.595
C   8.053   5.158   32.298
C   8.170   5.307   30.693
C   6.659   4.982   32.886
C   10.359   4.448   33.420
C   10.081   4.669   34.871
C   10.611   3.652   35.938
O   9.919   2.739   36.447
O   11.810   4.056   36.511
N   7.870   5.728   27.589
C   7.361   6.553   28.558
C   6.549   7.578   27.869
C   6.489   7.245   26.503
C   7.309   5.986   26.362
C   5.867   8.726   28.571
C   5.854   8.064   25.355
O   5.914   7.651   24.201
C   5.162   9.352   25.656
N   8.914   3.274   25.848
C   8.257   4.011   24.976
C   8.415   3.420   23.565
C   9.495   2.333   23.765
C   9.512   2.198   25.303
C   7.049   2.851   23.011
C   10.950   2.704   23.222
C   11.622   1.580   22.494
N   10.103   2.240   28.274
C   10.525   1.261   27.400
C   11.310   0.308   28.240
C   11.114   0.633   29.586
C   10.369   1.833   29.558
C   12.158   -0.843   27.763
C   11.503   0.311   30.986
O   12.163   -0.597   31.463
C   10.777   1.320   31.851
C   10.071   0.495   32.779
O   8.951   0.070   32.583
O   10.762   0.498   34.012
C   9.942   -0.114   35.126
C   12.232   3.361   37.718
C   12.014   4.316   38.867
C   12.193   4.168   40.202
C   12.645   2.918   40.977
C   11.932   5.344   41.142
C   10.622   5.350   41.954
C   9.572   6.351   41.391
C   9.389   7.598   42.386
C   8.010   7.550   43.167
C   9.567   8.979   41.710
C   10.910   9.542   42.196
C   10.964   11.056   42.201
C   11.499   11.698   43.501
C   11.769   13.203   43.361
C   10.502   11.536   44.673
C   11.066   10.605   45.728
C   10.252   9.337   45.995
C   9.728   9.246   47.447
C   8.859   7.960   47.589
C   9.111   10.528   48.063
H   6.924   7.040   30.662
H   6.887   5.581   24.348
H   10.744   0.322   25.551
H   8.468   3.230   33.005
H   8.366   6.080   32.786
H   6.503   5.895   33.461
H   6.462   4.135   33.542
H   5.849   4.894   32.162
H   10.645   5.409   32.991
H   11.204   3.760   33.437
H   9.011   4.743   35.066
H   10.526   5.616   35.176
H   6.337   9.705   28.665
H   5.917   8.584   29.651
H   4.849   8.709   28.181
H   5.778   10.006   26.273
H   4.346   8.943   26.251
H   4.807   9.936   24.807
H   8.849   4.176   22.911
H   9.067   1.384   23.442
H   7.145   1.799   22.741
H   6.750   3.419   22.130
H   6.247   2.978   23.738
H   11.505   3.098   24.074
H   10.806   3.503   22.494
H   11.588   1.742   21.416
H   11.215   0.596   22.728
H   12.678   1.413   22.709
H   12.713   -0.615   26.853
H   11.468   -1.680   27.655
H   12.829   -1.140   28.569
H   11.569   1.844   32.386
H   9.853   -1.174   34.888
H   8.902   0.204   35.191
H   10.484   -0.046   36.069
H   13.281   3.095   37.591
H   11.830   2.348   37.749
H   11.536   5.218   38.487
H   11.850   2.193   41.150
H   13.051   3.187   41.952
H   13.332   2.421   40.292
H   12.048   6.257   40.558
H   12.735   5.270   41.876
H   10.898   5.460   43.002
H   10.155   4.368   41.875
H   8.622   5.923   41.071
H   9.919   6.686   40.414
H   10.037   7.380   43.235
H   7.857   8.594   43.439
H   8.112   6.952   44.072
H   7.175   7.269   42.524
H   8.731   9.658   41.875
H   9.577   8.919   40.621
H   11.609   9.231   41.420
H   11.312   9.191   43.146
H   10.055   11.535   41.839
H   11.767   11.339   41.520
H   12.463   11.259   43.759
H   12.161   13.416   42.366
H   12.476   13.612   44.082
H   10.839   13.762   43.463
H   9.601   11.086   44.255
H   10.319   12.492   45.162
H   11.038   11.325   46.546
H   12.124   10.350   45.658
H   10.788   8.445   45.672
H   9.300   9.456   45.479
H   10.663   9.014   47.958
H   9.082   7.376   48.482
H   9.102   7.241   46.807
H   7.779   8.095   47.649
H   9.878   11.081   48.605
H   8.264   10.203   48.668
H   8.689   11.197   47.314
