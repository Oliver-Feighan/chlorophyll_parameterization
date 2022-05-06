%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_101_chromophore_8 TDDFT with blyp functional

0 1
Mg   45.000   2.970   46.565
C   42.675   5.665   46.155
C   42.499   0.820   45.982
C   47.305   0.647   46.781
C   47.556   5.455   46.268
N   42.770   3.223   46.100
C   42.070   4.379   45.943
C   40.590   4.129   45.657
C   40.484   2.582   45.833
C   42.018   2.139   45.801
C   39.764   2.198   47.219
C   40.290   4.593   44.121
C   38.985   5.267   43.949
C   38.157   4.949   42.654
O   37.191   4.180   42.566
O   38.675   5.713   41.609
N   44.957   0.927   46.434
C   43.819   0.210   46.207
C   44.109   -1.231   46.134
C   45.551   -1.294   46.466
C   46.015   0.076   46.654
C   43.155   -2.287   45.706
C   46.318   -2.594   46.702
O   45.856   -3.636   46.487
C   47.716   -2.572   47.212
N   47.045   3.005   46.211
C   47.743   1.952   46.728
C   49.230   2.273   46.838
C   49.413   3.630   46.095
C   47.917   4.105   46.095
C   49.718   2.278   48.350
C   49.969   3.468   44.655
C   51.472   3.059   44.441
N   45.155   5.104   46.440
C   46.240   5.962   46.440
C   45.862   7.320   46.403
C   44.441   7.267   46.338
C   44.086   5.887   46.380
C   46.746   8.513   46.351
C   43.191   8.024   46.326
O   43.004   9.215   46.434
C   41.937   7.012   46.223
C   41.149   7.061   47.504
O   41.280   6.289   48.414
O   40.078   7.935   47.459
C   39.161   7.760   48.617
C   38.018   5.438   40.321
C   39.146   5.236   39.315
C   39.863   6.107   38.608
C   39.824   7.605   38.709
C   41.029   5.622   37.709
C   40.851   4.402   36.744
C   41.400   2.959   37.195
C   42.066   2.050   36.107
C   42.663   0.800   36.701
C   43.081   2.826   35.118
C   42.545   2.663   33.610
C   42.301   4.004   32.802
C   42.906   3.763   31.370
C   44.292   4.405   31.228
C   42.040   4.477   30.330
C   41.732   3.699   29.131
C   41.954   4.465   27.778
C   43.335   4.171   27.148
C   43.277   3.073   26.074
C   43.986   5.444   26.609
H   41.703   0.074   45.934
H   48.130   -0.039   46.986
H   48.386   6.163   46.316
H   39.963   4.664   46.370
H   39.938   2.150   44.995
H   38.914   1.556   46.989
H   39.405   3.123   47.670
H   40.494   1.764   47.903
H   40.373   3.770   43.412
H   41.069   5.324   43.903
H   39.219   6.331   43.922
H   38.300   5.105   44.781
H   43.675   -3.062   45.142
H   42.517   -1.803   44.966
H   42.581   -2.751   46.507
H   48.252   -2.316   46.298
H   48.176   -3.518   47.496
H   47.708   -1.843   48.023
H   49.705   1.462   46.285
H   49.996   4.340   46.681
H   50.465   3.029   48.606
H   50.047   1.296   48.693
H   48.766   2.499   48.831
H   49.615   4.397   44.209
H   49.405   2.654   44.199
H   52.172   3.876   44.264
H   51.522   2.294   43.667
H   51.763   2.719   45.436
H   46.552   9.208   47.168
H   46.451   9.082   45.470
H   47.819   8.319   46.324
H   41.213   7.289   45.457
H   39.308   8.418   49.474
H   39.104   6.790   49.111
H   38.129   7.992   48.353
H   37.397   6.277   40.005
H   37.381   4.555   40.258
H   39.374   4.173   39.229
H   40.771   8.113   38.892
H   39.258   7.765   39.627
H   39.224   8.013   37.896
H   41.772   5.433   38.483
H   41.357   6.532   37.205
H   41.343   4.688   35.815
H   39.786   4.218   36.607
H   40.550   2.519   37.717
H   42.217   3.089   37.904
H   41.298   1.611   35.471
H   43.574   0.469   36.202
H   41.941   0.006   36.513
H   42.656   0.819   37.791
H   44.095   2.435   35.209
H   43.316   3.862   35.361
H   41.684   2.005   33.491
H   43.395   2.287   33.042
H   42.866   4.817   33.258
H   41.260   4.320   32.722
H   43.018   2.725   31.058
H   44.977   3.709   31.711
H   44.172   5.409   31.636
H   44.726   4.569   30.241
H   42.502   5.364   29.896
H   41.156   4.909   30.798
H   40.707   3.333   29.196
H   42.370   2.821   29.231
H   41.796   5.526   27.971
H   41.142   4.064   27.172
H   43.961   3.746   27.933
H   44.072   2.356   26.275
H   43.403   3.430   25.052
H   42.388   2.448   25.995
H   45.032   5.538   26.900
H   43.495   6.388   26.843
H   43.987   5.380   25.521

