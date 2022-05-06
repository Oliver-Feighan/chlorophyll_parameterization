%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1051_chromophore_23 ZINDO

0 1
Mg   -10.983   41.282   41.744
C   -9.463   38.309   40.614
C   -8.430   43.011   39.974
C   -12.640   44.095   42.220
C   -13.829   39.339   42.538
N   -9.215   40.673   40.386
C   -8.736   39.430   40.229
C   -7.574   39.399   39.182
C   -7.151   40.897   39.234
C   -8.306   41.573   39.960
C   -5.772   41.032   39.847
C   -7.984   38.904   37.771
C   -7.282   37.594   37.224
C   -6.270   37.622   36.032
O   -5.143   38.028   36.131
O   -6.796   36.898   34.939
N   -10.461   43.299   41.383
C   -9.382   43.816   40.647
C   -9.481   45.236   40.762
C   -10.645   45.558   41.488
C   -11.325   44.323   41.732
C   -8.463   46.229   40.205
C   -11.087   46.905   41.867
O   -10.439   47.911   41.493
C   -12.256   47.145   42.614
N   -12.932   41.648   42.137
C   -13.441   42.937   42.373
C   -14.806   42.798   43.094
C   -15.229   41.365   42.672
C   -13.875   40.730   42.464
C   -14.783   43.109   44.597
C   -16.037   41.387   41.367
C   -17.560   40.998   41.581
N   -11.530   39.204   41.790
C   -12.692   38.603   42.131
C   -12.521   37.196   42.120
C   -11.283   36.983   41.495
C   -10.734   38.282   41.279
C   -13.568   36.180   42.569
C   -10.260   36.014   41.035
O   -10.280   34.775   41.025
C   -9.136   36.845   40.273
C   -7.867   36.372   40.838
O   -7.387   36.849   41.868
O   -7.331   35.314   40.109
C   -6.041   34.780   40.634
C   -6.002   36.775   33.703
C   -7.013   36.662   32.624
C   -7.378   37.583   31.625
C   -6.878   39.026   31.350
C   -8.319   37.153   30.522
C   -7.640   36.227   29.464
C   -8.077   36.530   28.021
C   -7.360   37.765   27.433
C   -6.360   37.343   26.342
C   -8.331   38.779   26.812
C   -8.478   40.048   27.734
C   -9.788   40.809   27.458
C   -10.903   40.317   28.361
C   -12.264   40.281   27.519
C   -11.022   41.107   29.620
C   -10.952   40.213   30.888
C   -12.183   40.323   31.817
C   -12.974   38.967   31.879
C   -14.338   38.884   31.117
C   -13.382   38.563   33.355
H   -7.628   43.454   39.381
H   -13.236   44.980   42.453
H   -14.703   38.776   42.873
H   -6.793   38.765   39.603
H   -7.208   41.344   38.241
H   -5.970   41.442   40.837
H   -5.096   41.652   39.259
H   -5.229   40.093   39.960
H   -7.667   39.679   37.073
H   -9.072   38.846   37.743
H   -8.034   36.848   36.968
H   -6.692   37.216   38.059
H   -8.811   46.773   39.327
H   -7.774   45.623   39.618
H   -8.081   46.861   41.006
H   -12.321   46.335   43.341
H   -13.141   47.088   41.981
H   -12.080   47.968   43.307
H   -15.523   43.514   42.693
H   -15.764   40.792   43.430
H   -14.638   42.178   45.144
H   -15.689   43.647   44.876
H   -13.858   43.674   44.709
H   -15.667   40.610   40.698
H   -15.944   42.374   40.914
H   -17.934   40.574   40.649
H   -18.063   41.946   41.769
H   -17.713   40.292   42.398
H   -13.647   36.308   43.649
H   -13.290   35.139   42.408
H   -14.460   36.401   41.983
H   -9.134   36.717   39.191
H   -5.956   33.714   40.422
H   -5.878   34.887   41.707
H   -5.238   35.270   40.083
H   -5.290   35.950   33.736
H   -5.451   37.709   33.587
H   -7.511   35.693   32.579
H   -7.814   39.563   31.193
H   -6.131   39.028   30.556
H   -6.426   39.409   32.264
H   -8.873   37.953   30.031
H   -9.018   36.497   31.039
H   -7.915   35.192   29.668
H   -6.556   36.262   29.573
H   -9.155   36.673   28.086
H   -7.864   35.633   27.439
H   -6.685   38.227   28.153
H   -6.156   36.274   26.401
H   -5.382   37.803   26.479
H   -6.783   37.615   25.374
H   -9.324   38.329   26.838
H   -8.117   39.219   25.838
H   -7.599   40.674   27.583
H   -8.282   39.745   28.763
H   -9.996   40.629   26.403
H   -9.617   41.878   27.586
H   -10.709   39.246   28.421
H   -12.489   39.239   27.290
H   -12.334   40.816   26.572
H   -13.082   40.648   28.139
H   -11.861   41.803   29.618
H   -10.192   41.806   29.716
H   -10.110   40.528   31.504
H   -10.710   39.184   30.623
H   -12.822   41.102   31.401
H   -11.834   40.488   32.837
H   -12.315   38.176   31.521
H   -14.288   37.984   30.504
H   -14.597   39.802   30.590
H   -15.207   38.711   31.752
H   -14.028   39.316   33.806
H   -12.584   38.355   34.067
H   -13.867   37.588   33.406

