%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_651_chromophore_24 ZINDO

0 1
Mg   0.067   43.882   25.395
C   2.223   43.211   27.962
C   -2.585   43.283   27.623
C   -1.981   44.213   22.704
C   2.886   44.038   23.135
N   -0.167   43.167   27.560
C   0.917   43.070   28.443
C   0.391   42.983   29.847
C   -1.110   42.622   29.737
C   -1.346   43.040   28.261
C   -1.722   41.225   30.126
C   0.824   44.157   30.748
C   1.120   43.840   32.211
C   0.798   42.400   32.703
O   1.640   41.573   32.818
O   -0.467   42.198   33.008
N   -2.022   43.832   25.181
C   -2.936   43.574   26.231
C   -4.279   43.659   25.733
C   -4.148   44.042   24.300
C   -2.680   44.042   23.977
C   -5.438   43.518   26.666
C   -5.177   44.397   23.336
O   -4.836   44.854   22.248
C   -6.614   44.269   23.752
N   0.426   44.022   23.152
C   -0.616   44.226   22.328
C   -0.106   44.344   20.882
C   1.435   44.520   20.967
C   1.622   44.196   22.479
C   -0.441   43.103   20.042
C   2.124   45.909   20.650
C   1.771   47.190   21.432
N   2.110   43.763   25.484
C   3.096   43.855   24.530
C   4.422   43.770   25.148
C   4.143   43.519   26.497
C   2.723   43.489   26.650
C   5.660   43.664   24.335
C   4.749   43.311   27.802
O   5.940   43.229   28.103
C   3.488   43.246   28.796
C   3.789   42.086   29.639
O   3.701   40.854   29.316
O   4.175   42.502   30.912
C   4.596   41.464   31.907
C   -0.778   40.986   33.752
C   -2.125   41.167   34.418
C   -2.755   40.640   35.487
C   -2.086   39.604   36.548
C   -4.229   41.049   35.780
C   -4.309   42.297   36.765
C   -5.468   43.268   36.438
C   -6.176   43.785   37.702
C   -5.404   44.917   38.393
C   -7.639   44.254   37.459
C   -8.654   43.119   37.917
C   -9.382   42.473   36.688
C   -9.422   40.920   36.705
C   -9.045   40.271   35.344
C   -10.884   40.410   37.110
C   -11.168   40.243   38.651
C   -12.100   39.072   38.777
C   -12.883   39.047   40.113
C   -13.469   37.665   40.309
C   -13.829   40.165   40.554
H   -3.327   43.336   28.423
H   -2.636   44.497   21.877
H   3.824   44.084   22.578
H   0.854   42.067   30.215
H   -1.621   43.315   30.405
H   -2.641   41.385   30.690
H   -0.962   40.638   30.641
H   -1.996   40.711   29.206
H   -0.017   44.849   30.783
H   1.582   44.779   30.273
H   0.762   44.568   32.940
H   2.190   43.849   32.418
H   -6.399   43.978   26.436
H   -5.183   44.042   27.587
H   -5.564   42.444   26.798
H   -7.331   44.158   22.938
H   -6.896   45.163   24.308
H   -6.797   43.393   24.373
H   -0.535   45.237   20.428
H   1.993   43.717   20.486
H   0.346   42.427   20.376
H   -0.380   43.346   18.982
H   -1.398   42.646   20.296
H   1.818   46.076   19.617
H   3.213   45.868   20.614
H   2.700   47.588   21.839
H   1.177   46.890   22.296
H   1.345   47.976   20.808
H   5.522   43.079   23.426
H   6.456   43.143   24.869
H   6.145   44.620   24.140
H   3.651   44.150   29.384
H   5.631   41.188   31.704
H   4.002   40.557   31.791
H   4.604   41.741   32.962
H   -0.025   40.640   34.461
H   -1.130   40.120   33.191
H   -2.577   42.017   33.907
H   -1.010   39.763   36.489
H   -2.308   38.558   36.334
H   -2.514   39.782   37.534
H   -4.766   40.268   36.318
H   -4.788   41.176   34.853
H   -3.337   42.789   36.723
H   -4.371   41.975   37.804
H   -6.220   42.791   35.810
H   -4.982   44.133   35.987
H   -6.154   42.939   38.389
H   -5.741   45.928   38.164
H   -4.383   44.907   38.012
H   -5.399   44.741   39.469
H   -7.711   44.467   36.392
H   -7.876   45.174   37.992
H   -9.458   43.513   38.539
H   -8.166   42.237   38.333
H   -8.987   42.816   35.732
H   -10.420   42.807   36.691
H   -8.804   40.605   37.546
H   -8.490   39.339   35.452
H   -8.429   41.005   34.824
H   -9.956   40.114   34.767
H   -11.195   39.495   36.606
H   -11.617   41.163   36.820
H   -11.524   41.162   39.115
H   -10.307   39.890   39.219
H   -11.486   38.180   38.652
H   -12.847   39.113   37.984
H   -12.039   39.089   40.802
H   -13.224   36.961   39.513
H   -14.557   37.672   40.375
H   -13.200   37.357   41.319
H   -13.895   40.940   39.790
H   -13.404   40.707   41.399
H   -14.865   39.937   40.803

