%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1451_chromophore_5 TDDFT with wB97XD functional

0 1
Mg   24.500   -6.911   47.433
C   27.206   -4.779   46.506
C   22.574   -5.278   45.263
C   22.176   -9.312   47.814
C   26.892   -8.917   49.179
N   24.930   -5.264   45.973
C   26.040   -4.490   45.751
C   25.826   -3.450   44.636
C   24.227   -3.436   44.580
C   23.863   -4.769   45.310
C   23.561   -2.206   45.248
C   26.601   -3.978   43.361
C   27.083   -2.847   42.368
C   26.325   -2.771   41.029
O   25.151   -2.359   40.946
O   27.154   -3.163   39.970
N   22.591   -7.309   46.711
C   22.025   -6.457   45.789
C   20.833   -7.053   45.339
C   20.520   -8.205   46.263
C   21.773   -8.330   46.996
C   19.879   -6.347   44.317
C   19.264   -9.096   46.214
O   18.260   -8.756   45.576
C   19.152   -10.337   47.137
N   24.579   -8.855   48.326
C   23.397   -9.576   48.429
C   23.568   -10.801   49.366
C   25.172   -10.767   49.553
C   25.566   -9.382   49.033
C   22.900   -10.561   50.725
C   25.964   -11.842   48.769
C   26.994   -12.563   49.561
N   26.659   -6.905   47.837
C   27.425   -7.739   48.592
C   28.779   -7.217   48.654
C   28.720   -6.063   47.879
C   27.432   -5.893   47.319
C   29.919   -7.772   49.452
C   29.487   -4.887   47.425
O   30.632   -4.597   47.746
C   28.573   -4.130   46.281
C   28.729   -2.653   46.407
O   29.521   -1.974   45.724
O   28.008   -2.202   47.493
C   28.058   -0.730   47.649
C   26.455   -3.214   38.642
C   26.132   -4.618   38.344
C   26.318   -5.320   37.225
C   26.804   -4.688   35.933
C   25.900   -6.806   37.155
C   24.397   -7.090   37.235
C   23.861   -7.532   35.894
C   23.077   -6.405   35.114
C   21.639   -6.780   34.815
C   23.932   -6.058   33.934
C   23.462   -4.724   33.274
C   24.588   -4.084   32.432
C   24.355   -4.208   30.878
C   23.028   -3.430   30.379
C   25.663   -3.745   30.068
C   25.842   -4.609   28.845
C   25.467   -3.978   27.488
C   24.547   -4.913   26.701
C   24.852   -5.063   25.262
C   23.037   -4.607   26.976
H   21.815   -4.707   44.724
H   21.336   -9.965   48.061
H   27.615   -9.621   49.596
H   26.169   -2.473   44.977
H   23.813   -3.541   43.577
H   22.562   -2.057   44.837
H   24.240   -1.403   44.962
H   23.472   -2.300   46.330
H   25.975   -4.592   42.714
H   27.447   -4.614   43.621
H   28.105   -3.117   42.102
H   27.120   -1.847   42.799
H   20.259   -6.335   43.295
H   19.467   -5.440   44.757
H   19.045   -7.032   44.164
H   19.158   -10.019   48.179
H   20.082   -10.864   46.919
H   18.226   -10.892   46.988
H   23.303   -11.624   48.702
H   25.537   -10.800   50.580
H   21.817   -10.631   50.624
H   23.256   -9.615   51.132
H   23.253   -11.370   51.365
H   26.398   -11.460   47.845
H   25.202   -12.578   48.512
H   27.847   -12.778   48.917
H   26.771   -13.573   49.907
H   27.283   -12.013   50.456
H   30.497   -6.890   49.730
H   30.608   -8.343   48.830
H   29.586   -8.240   50.378
H   28.953   -4.449   45.310
H   27.537   -0.136   46.898
H   29.100   -0.411   47.651
H   27.649   -0.433   48.615
H   27.264   -2.883   37.991
H   25.608   -2.535   38.545
H   25.764   -5.030   39.283
H   27.047   -5.477   35.221
H   27.684   -4.051   36.026
H   26.012   -4.088   35.485
H   26.291   -7.297   38.047
H   26.286   -7.369   36.305
H   23.914   -6.176   37.581
H   24.213   -7.911   37.927
H   23.153   -8.353   36.003
H   24.644   -7.956   35.265
H   23.078   -5.456   35.650
H   21.371   -6.722   33.760
H   21.094   -5.998   35.345
H   21.314   -7.709   35.283
H   23.856   -6.749   33.094
H   24.973   -5.899   34.216
H   22.993   -4.078   34.016
H   22.631   -4.857   32.582
H   25.567   -4.468   32.721
H   24.588   -3.013   32.631
H   24.181   -5.277   30.754
H   23.211   -2.755   29.542
H   22.642   -2.820   31.195
H   22.395   -4.280   30.123
H   26.539   -3.609   30.702
H   25.493   -2.775   29.602
H   25.344   -5.548   29.090
H   26.919   -4.665   28.686
H   26.372   -3.761   26.920
H   24.939   -3.038   27.649
H   24.591   -5.927   27.099
H   24.693   -6.107   24.995
H   25.862   -4.713   25.047
H   24.189   -4.445   24.657
H   22.511   -4.454   26.033
H   22.850   -3.667   27.495
H   22.547   -5.448   27.467

