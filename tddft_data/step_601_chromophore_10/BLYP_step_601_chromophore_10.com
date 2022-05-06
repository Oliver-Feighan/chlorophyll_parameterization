%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_601_chromophore_10 TDDFT with blyp functional

0 1
Mg   41.115   7.502   29.435
C   42.635   9.055   32.094
C   38.925   5.993   31.579
C   39.757   6.068   26.784
C   43.726   8.843   27.290
N   41.012   7.372   31.593
C   41.610   8.200   32.510
C   40.949   7.998   33.880
C   39.992   6.772   33.672
C   39.943   6.614   32.148
C   40.571   5.445   34.289
C   40.199   9.282   34.203
C   40.196   9.684   35.707
C   40.856   11.054   36.020
O   41.659   11.665   35.277
O   40.332   11.607   37.228
N   39.600   6.265   29.212
C   38.700   5.876   30.224
C   37.539   5.184   29.606
C   37.819   5.217   28.186
C   39.080   5.853   28.006
C   36.393   4.427   30.384
C   36.885   4.771   27.008
O   37.225   4.832   25.815
C   35.516   4.462   27.422
N   41.761   7.450   27.327
C   40.930   6.821   26.497
C   41.433   6.929   25.005
C   42.605   7.910   25.153
C   42.710   8.116   26.643
C   41.779   5.573   24.325
C   42.495   9.173   24.358
C   41.329   10.143   24.550
N   42.830   8.720   29.662
C   43.786   9.171   28.745
C   44.722   10.019   29.415
C   44.327   10.016   30.718
C   43.148   9.194   30.814
C   45.914   10.711   28.728
C   44.610   10.506   32.059
O   45.475   11.276   32.454
C   43.476   9.857   32.989
C   44.041   8.879   33.987
O   44.905   8.015   33.726
O   43.723   9.202   35.285
C   44.184   8.446   36.538
C   40.790   12.970   37.603
C   40.339   13.188   39.082
C   41.066   12.864   40.184
C   42.463   12.232   40.129
C   40.494   13.065   41.597
C   39.275   12.263   42.025
C   39.468   10.999   42.880
C   38.595   11.094   44.164
C   39.215   10.319   45.369
C   37.114   10.733   43.906
C   36.081   11.880   44.311
C   35.489   12.657   43.158
C   33.921   12.695   43.302
C   33.230   12.944   41.974
C   33.522   13.820   44.332
C   32.765   13.252   45.530
C   33.357   13.787   46.901
C   33.921   12.670   47.773
C   33.379   12.711   49.278
C   35.526   12.535   47.790
H   38.163   5.693   32.302
H   39.459   5.551   25.870
H   44.403   9.407   26.645
H   41.681   7.764   34.652
H   39.052   6.918   34.204
H   40.598   4.638   33.556
H   39.863   5.171   35.070
H   41.513   5.447   34.837
H   39.148   9.242   33.916
H   40.633   10.067   33.583
H   40.786   8.949   36.253
H   39.171   9.615   36.073
H   36.438   4.459   31.472
H   36.287   3.432   29.951
H   35.456   4.938   30.161
H   34.973   4.216   26.510
H   35.039   5.320   27.896
H   35.616   3.522   27.966
H   40.572   7.364   24.499
H   43.548   7.515   24.773
H   41.315   4.763   24.887
H   42.863   5.485   24.252
H   41.277   5.505   23.360
H   42.401   8.800   23.338
H   43.435   9.723   24.333
H   40.665   10.109   23.687
H   41.630   11.178   24.713
H   40.657   9.965   25.389
H   46.444   11.472   29.299
H   45.567   11.237   27.838
H   46.691   9.987   28.481
H   43.127   10.719   33.558
H   44.799   9.048   37.207
H   44.744   7.572   36.203
H   43.412   8.019   37.178
H   40.333   13.780   37.033
H   41.849   13.207   37.501
H   39.305   13.532   39.116
H   42.365   11.371   40.790
H   43.220   12.886   40.562
H   42.740   11.920   39.122
H   40.317   14.128   41.763
H   41.309   12.850   42.287
H   38.810   11.840   41.135
H   38.502   12.918   42.426
H   40.521   10.777   43.055
H   39.080   10.129   42.352
H   38.600   12.132   44.497
H   38.654   10.269   46.303
H   39.995   10.999   45.713
H   39.725   9.378   45.162
H   36.932   9.769   44.381
H   36.910   10.465   42.870
H   36.648   12.546   44.961
H   35.329   11.399   44.936
H   35.806   12.123   42.262
H   35.834   13.684   43.044
H   33.655   11.668   43.553
H   32.630   12.050   41.802
H   34.000   13.208   41.250
H   32.529   13.778   41.955
H   32.828   14.553   43.920
H   34.365   14.398   44.711
H   32.790   12.170   45.655
H   31.726   13.572   45.459
H   32.566   14.310   47.439
H   34.149   14.522   46.758
H   33.550   11.713   47.404
H   34.079   12.350   50.031
H   32.498   12.083   49.413
H   33.176   13.746   49.553
H   35.856   12.496   48.828
H   35.992   13.399   47.317
H   35.887   11.666   47.239

