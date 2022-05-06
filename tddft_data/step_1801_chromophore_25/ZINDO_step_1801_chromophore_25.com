%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_1801_chromophore_25 ZINDO

0 1
Mg   -2.479   34.491   26.742
C   -3.817   32.887   29.506
C   -1.027   36.836   28.812
C   -1.558   36.307   24.046
C   -3.928   32.189   24.678
N   -2.532   34.860   28.953
C   -2.944   33.931   29.828
C   -2.436   34.293   31.257
C   -1.618   35.686   31.037
C   -1.775   35.852   29.484
C   -2.011   36.966   31.843
C   -1.610   33.185   32.090
C   -2.256   32.567   33.310
C   -1.364   32.287   34.470
O   -0.882   31.206   34.699
O   -1.363   33.407   35.289
N   -1.455   36.320   26.510
C   -0.840   37.075   27.443
C   -0.177   38.164   26.806
C   -0.295   37.984   25.368
C   -1.166   36.843   25.311
C   0.692   39.123   27.551
C   0.267   38.846   24.211
O   0.104   38.651   23.011
C   1.136   40.109   24.499
N   -2.822   34.353   24.637
C   -2.323   35.199   23.715
C   -2.358   34.634   22.269
C   -2.998   33.228   22.507
C   -3.232   33.221   24.031
C   -3.088   35.315   21.126
C   -2.192   32.028   21.919
C   -0.774   31.792   22.522
N   -3.673   32.852   26.964
C   -4.192   31.976   26.034
C   -4.847   30.907   26.690
C   -4.783   31.196   28.092
C   -4.086   32.430   28.173
C   -5.503   29.726   26.127
C   -5.100   30.830   29.469
O   -5.778   29.857   29.839
C   -4.503   31.921   30.444
C   -5.668   32.554   31.219
O   -6.537   33.352   30.835
O   -5.535   32.160   32.506
C   -6.528   32.754   33.441
C   -0.632   33.118   36.575
C   -1.322   34.004   37.592
C   -1.896   33.639   38.747
C   -2.255   32.182   39.059
C   -2.323   34.627   39.766
C   -1.474   35.822   40.184
C   -1.199   35.903   41.775
C   0.096   36.641   42.208
C   1.303   35.824   42.730
C   -0.312   37.688   43.303
C   -1.080   37.235   44.594
C   -0.344   37.528   45.932
C   0.290   36.345   46.681
C   -0.291   36.171   48.079
C   1.840   36.625   46.826
C   2.711   35.392   46.943
C   4.237   35.730   46.878
C   4.883   34.950   45.635
C   6.375   34.680   45.895
C   4.664   35.686   44.312
H   -0.367   37.427   29.450
H   -0.992   36.724   23.211
H   -4.224   31.388   23.998
H   -3.351   34.709   31.677
H   -0.617   35.469   31.410
H   -2.843   36.791   32.526
H   -2.264   37.750   31.129
H   -1.105   37.342   32.318
H   -0.640   33.608   32.350
H   -1.377   32.381   31.392
H   -2.747   31.614   33.114
H   -3.079   33.158   33.711
H   0.832   40.097   27.083
H   1.667   38.707   27.804
H   0.182   39.341   28.489
H   0.706   40.763   25.257
H   1.117   40.776   23.637
H   2.084   39.671   24.812
H   -1.335   34.410   21.966
H   -4.020   33.161   22.134
H   -3.641   36.129   21.596
H   -3.796   34.609   20.693
H   -2.408   35.695   20.363
H   -1.985   32.178   20.859
H   -2.765   31.116   22.088
H   -0.335   32.690   22.959
H   -0.077   31.503   21.735
H   -0.837   30.957   23.220
H   -4.935   29.213   25.350
H   -6.392   30.080   25.605
H   -5.696   29.022   26.936
H   -3.905   31.369   31.168
H   -6.355   33.776   33.777
H   -6.689   32.075   34.278
H   -7.506   32.857   32.970
H   0.355   33.512   36.331
H   -0.460   32.087   36.887
H   -1.150   35.078   37.529
H   -1.528   31.700   39.712
H   -2.404   31.577   38.165
H   -3.269   32.124   39.453
H   -2.695   34.142   40.668
H   -3.273   34.990   39.373
H   -1.856   36.804   39.906
H   -0.529   35.687   39.659
H   -1.149   34.933   42.268
H   -2.128   36.342   42.139
H   0.393   37.131   41.281
H   1.631   36.115   43.727
H   2.138   36.083   42.079
H   1.229   34.742   42.837
H   -0.847   38.504   42.818
H   0.536   38.269   43.668
H   -1.010   36.161   44.420
H   -2.069   37.692   44.570
H   -1.132   38.008   46.512
H   0.463   38.229   45.719
H   0.110   35.418   46.135
H   0.436   35.742   48.769
H   -1.069   35.407   48.103
H   -0.705   37.144   48.343
H   2.002   37.391   47.584
H   2.190   37.050   45.885
H   2.519   34.618   46.199
H   2.564   34.967   47.936
H   4.513   35.448   47.894
H   4.427   36.799   46.776
H   4.320   34.021   45.716
H   6.703   35.242   46.769
H   7.049   34.947   45.081
H   6.462   33.598   46.003
H   3.698   35.418   43.885
H   5.472   35.511   43.601
H   4.624   36.746   44.560
