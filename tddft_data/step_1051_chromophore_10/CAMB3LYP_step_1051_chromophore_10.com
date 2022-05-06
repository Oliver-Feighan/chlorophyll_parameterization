%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1051_chromophore_10 TDDFT with cam-b3lyp functional

0 1
Mg   41.106   8.239   29.650
C   43.028   9.571   32.122
C   39.175   6.590   32.107
C   39.245   6.610   27.234
C   43.039   9.608   27.195
N   40.977   8.264   31.879
C   41.906   8.874   32.613
C   41.392   8.942   34.052
C   40.439   7.741   34.125
C   40.175   7.465   32.629
C   40.978   6.579   34.847
C   40.728   10.364   34.464
C   41.512   11.136   35.531
C   41.556   10.398   36.912
O   42.343   9.466   37.175
O   40.711   10.923   37.758
N   39.343   6.922   29.676
C   38.733   6.456   30.796
C   37.486   5.829   30.484
C   37.404   5.827   29.033
C   38.719   6.385   28.538
C   36.485   5.226   31.526
C   36.229   5.464   28.133
O   36.288   5.599   26.875
C   34.875   5.068   28.753
N   41.099   8.080   27.609
C   40.262   7.361   26.835
C   40.631   7.380   25.346
C   41.615   8.627   25.281
C   41.922   8.814   26.793
C   41.140   6.055   24.760
C   41.126   9.891   24.688
C   39.652   10.344   24.815
N   42.742   9.337   29.598
C   43.508   9.802   28.546
C   44.674   10.538   29.068
C   44.448   10.513   30.508
C   43.320   9.711   30.736
C   45.796   11.040   28.258
C   44.989   10.948   31.761
O   46.016   11.555   31.967
C   44.069   10.365   32.868
C   44.913   9.652   33.893
O   45.213   8.453   34.037
O   45.229   10.598   34.809
C   45.878   10.217   36.057
C   40.833   10.471   39.135
C   39.864   11.214   40.007
C   39.465   10.986   41.286
C   40.128   9.923   42.132
C   38.372   11.790   41.888
C   37.112   10.972   42.330
C   36.794   10.950   43.889
C   35.529   11.800   44.249
C   35.444   11.900   45.761
C   34.249   11.043   43.730
C   33.206   12.146   43.255
C   33.091   12.261   41.706
C   31.620   12.768   41.410
C   31.238   12.358   40.010
C   31.350   14.302   41.696
C   30.274   14.627   42.761
C   30.724   15.814   43.740
C   31.185   15.375   45.166
C   30.001   15.708   46.232
C   32.517   15.971   45.405
H   38.456   6.256   32.857
H   38.697   5.998   26.515
H   43.727   9.910   26.403
H   42.251   8.837   34.715
H   39.537   8.097   34.623
H   41.259   5.831   34.105
H   40.212   6.085   35.445
H   41.782   6.834   35.536
H   39.685   10.236   34.753
H   40.707   10.967   33.556
H   41.013   12.105   35.568
H   42.585   11.213   35.358
H   35.668   5.817   31.940
H   36.981   4.805   32.401
H   36.163   4.252   31.157
H   34.071   5.075   28.017
H   34.789   5.855   29.502
H   34.810   4.054   29.147
H   39.672   7.483   24.838
H   42.539   8.436   24.735
H   40.850   5.296   25.487
H   42.198   6.063   24.497
H   40.639   5.745   23.843
H   41.468   10.001   23.659
H   41.775   10.582   25.225
H   39.542   11.411   25.012
H   39.090   9.796   25.571
H   39.229   10.254   23.814
H   45.586   10.852   27.205
H   46.653   10.418   28.516
H   46.077   12.061   28.517
H   43.697   11.257   33.372
H   46.735   9.626   35.733
H   45.232   9.577   36.659
H   46.279   11.062   36.616
H   41.837   10.649   39.520
H   40.570   9.414   39.075
H   39.214   11.922   39.491
H   40.228   10.540   43.025
H   41.112   9.665   41.739
H   39.563   8.992   42.180
H   37.995   12.517   41.169
H   38.669   12.433   42.717
H   37.276   9.987   41.892
H   36.327   11.503   41.792
H   37.695   11.366   44.339
H   36.771   9.887   44.132
H   35.598   12.773   43.762
H   34.520   11.555   46.225
H   35.415   12.984   45.871
H   36.324   11.425   46.195
H   33.797   10.441   44.518
H   34.603   10.398   42.926
H   33.519   13.113   43.651
H   32.255   11.946   43.748
H   33.159   11.256   41.290
H   33.780   12.921   41.178
H   30.949   12.125   41.979
H   30.923   11.323   40.145
H   32.055   12.496   39.302
H   30.333   12.909   39.753
H   31.075   14.795   40.764
H   32.196   14.830   42.135
H   30.109   13.731   43.359
H   29.319   14.787   42.260
H   29.800   16.386   43.813
H   31.471   16.368   43.170
H   31.177   14.285   45.135
H   30.203   16.606   46.817
H   29.856   14.814   46.838
H   29.075   15.880   45.683
H   32.742   16.893   44.869
H   33.346   15.312   45.148
H   32.599   16.233   46.459

