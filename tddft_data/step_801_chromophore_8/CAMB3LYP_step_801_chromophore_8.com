%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_801_chromophore_8 TDDFT with cam-b3lyp functional

0 1
Mg   46.131   2.988   47.469
C   43.615   5.360   46.742
C   43.718   0.559   47.278
C   48.596   0.751   47.185
C   48.507   5.549   46.982
N   43.976   2.973   46.885
C   43.083   4.078   46.790
C   41.590   3.632   46.726
C   41.739   2.132   47.182
C   43.270   1.817   47.038
C   40.921   1.791   48.464
C   41.186   3.745   45.198
C   39.889   4.403   44.794
C   39.978   5.596   43.872
O   40.182   6.779   44.215
O   39.993   5.122   42.542
N   46.145   0.898   47.319
C   45.092   0.112   47.266
C   45.560   -1.313   47.184
C   46.979   -1.311   47.018
C   47.351   0.130   47.167
C   44.647   -2.479   47.361
C   47.882   -2.535   46.772
O   47.458   -3.685   46.730
C   49.390   -2.248   46.618
N   48.349   3.122   47.380
C   49.055   2.039   47.181
C   50.562   2.432   47.168
C   50.480   3.920   46.721
C   49.020   4.247   47.040
C   51.444   2.225   48.411
C   50.913   4.258   45.246
C   52.381   3.923   44.799
N   46.076   5.027   47.034
C   47.090   5.940   46.883
C   46.530   7.268   46.602
C   45.147   7.056   46.614
C   44.936   5.671   46.820
C   47.287   8.500   46.372
C   43.882   7.732   46.534
O   43.598   8.887   46.411
C   42.803   6.669   46.685
C   41.948   7.098   47.841
O   42.302   7.315   49.004
O   40.699   7.427   47.301
C   39.747   7.916   48.278
C   40.357   6.186   41.648
C   40.286   5.813   40.204
C   40.910   6.401   39.130
C   41.681   7.744   39.239
C   40.827   5.692   37.723
C   41.968   4.668   37.587
C   43.154   4.977   36.581
C   43.149   4.001   35.346
C   43.637   2.475   35.843
C   43.936   4.610   34.272
C   44.037   3.648   33.031
C   42.825   3.721   32.042
C   43.349   4.379   30.699
C   43.703   5.809   30.823
C   42.172   4.274   29.603
C   42.512   3.536   28.258
C   42.819   4.486   27.126
C   44.193   4.179   26.422
C   44.063   3.642   24.967
C   45.170   5.354   26.502
H   42.875   -0.134   47.316
H   49.518   0.168   47.225
H   49.233   6.271   46.603
H   40.819   4.162   47.286
H   41.257   1.401   46.533
H   40.437   2.683   48.862
H   41.661   1.440   49.183
H   40.143   1.045   48.303
H   41.030   2.741   44.803
H   42.078   4.114   44.691
H   39.420   4.889   45.650
H   39.148   3.703   44.409
H   43.646   -2.160   47.651
H   45.001   -3.266   48.026
H   44.495   -2.896   46.365
H   49.816   -1.929   47.569
H   49.512   -1.464   45.871
H   49.883   -3.171   46.313
H   51.068   2.005   46.302
H   51.054   4.428   47.496
H   50.867   2.018   49.313
H   52.179   3.013   48.573
H   51.933   1.283   48.164
H   50.637   5.305   45.119
H   50.303   3.747   44.501
H   52.891   4.854   44.553
H   52.311   3.205   43.982
H   52.930   3.415   45.592
H   48.070   8.601   47.124
H   46.652   9.385   46.336
H   47.742   8.448   45.382
H   42.203   6.717   45.776
H   39.617   7.299   49.167
H   38.811   8.166   47.778
H   40.151   8.890   48.555
H   41.381   6.494   41.859
H   39.592   6.961   41.687
H   39.842   4.836   40.010
H   41.925   7.968   40.277
H   41.008   8.460   38.768
H   42.542   7.586   38.589
H   40.883   6.497   36.990
H   39.859   5.204   37.606
H   41.520   3.741   37.231
H   42.393   4.485   38.575
H   44.114   4.878   37.086
H   43.112   6.009   36.232
H   42.165   3.777   34.935
H   44.610   2.185   35.448
H   42.866   1.777   35.518
H   43.644   2.328   36.924
H   44.945   4.862   34.598
H   43.431   5.522   33.954
H   44.028   2.598   33.320
H   45.013   3.865   32.598
H   41.981   4.383   32.234
H   42.349   2.752   31.896
H   44.266   3.849   30.443
H   43.243   6.214   31.725
H   43.385   6.382   29.952
H   44.792   5.852   30.803
H   41.771   5.266   29.395
H   41.275   3.806   30.008
H   41.633   2.981   27.930
H   43.309   2.829   28.487
H   42.657   5.513   27.453
H   42.039   4.181   26.429
H   44.696   3.410   27.008
H   44.356   4.376   24.217
H   43.007   3.493   24.741
H   44.607   2.736   24.702
H   45.588   5.383   27.508
H   44.617   6.221   26.140
H   45.949   5.198   25.756

