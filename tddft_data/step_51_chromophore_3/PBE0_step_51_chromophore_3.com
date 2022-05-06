%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_51_chromophore_3 TDDFT with PBE1PBE functional

0 1
Mg   1.056   7.951   26.056
C   1.734   10.066   28.731
C   1.605   5.278   28.159
C   0.625   5.728   23.456
C   1.205   10.609   23.894
N   1.766   7.641   28.079
C   1.820   8.675   29.043
C   1.928   8.089   30.417
C   2.143   6.577   30.176
C   1.872   6.469   28.731
C   3.529   5.998   30.517
C   0.725   8.472   31.392
C   1.165   9.089   32.704
C   2.139   8.292   33.580
O   3.320   8.507   33.762
O   1.495   7.091   34.024
N   0.918   5.768   25.862
C   1.147   4.888   26.913
C   0.879   3.521   26.428
C   0.602   3.657   24.998
C   0.777   5.086   24.712
C   1.024   2.270   27.331
C   0.331   2.532   23.959
O   0.195   2.776   22.784
C   0.185   1.035   24.289
N   0.803   8.121   24.006
C   0.610   7.109   23.126
C   0.487   7.619   21.731
C   0.291   9.155   21.949
C   0.856   9.363   23.343
C   1.733   7.229   20.848
C   -1.138   9.860   21.711
C   -1.078   10.949   20.601
N   1.431   9.984   26.218
C   1.478   10.943   25.229
C   1.689   12.205   25.883
C   1.796   11.877   27.258
C   1.595   10.548   27.401
C   1.791   13.548   25.174
C   2.037   12.506   28.566
O   2.312   13.579   28.920
C   1.988   11.313   29.657
C   3.254   11.319   30.367
O   4.351   11.045   29.881
O   3.132   11.841   31.632
C   4.250   11.971   32.576
C   2.344   6.210   34.875
C   1.473   5.280   35.649
C   1.668   4.876   36.946
C   2.761   5.404   37.892
C   0.712   3.827   37.440
C   1.275   2.398   37.206
C   0.201   1.300   37.310
C   -0.041   0.903   38.857
C   -1.523   0.673   38.935
C   0.798   -0.433   39.164
C   2.091   -0.112   39.955
C   3.358   -0.237   39.072
C   4.198   -1.497   39.317
C   3.577   -2.851   39.063
C   4.842   -1.432   40.678
C   6.361   -1.755   40.728
C   7.176   -0.475   40.879
C   8.631   -0.555   40.530
C   9.521   -0.472   41.828
C   9.008   0.485   39.408
H   1.660   4.477   28.899
H   0.456   5.134   22.556
H   1.148   11.498   23.262
H   2.887   8.512   30.720
H   1.386   5.978   30.683
H   4.136   5.770   29.641
H   3.269   5.130   31.124
H   3.980   6.735   31.181
H   0.078   7.595   31.402
H   0.058   9.267   31.057
H   0.239   9.257   33.254
H   1.592   10.041   32.390
H   1.566   1.469   26.828
H   0.024   1.896   27.553
H   1.475   2.548   28.283
H   -0.346   0.553   23.468
H   -0.410   0.922   25.195
H   1.112   0.466   24.360
H   -0.465   7.268   21.333
H   0.954   9.723   21.297
H   1.303   6.503   20.158
H   2.477   6.792   21.515
H   2.176   8.093   20.354
H   -1.555   10.258   22.636
H   -1.853   9.154   21.290
H   -0.648   10.514   19.698
H   -0.575   11.843   20.969
H   -2.103   11.252   20.387
H   1.412   13.479   24.154
H   2.823   13.891   25.249
H   1.086   14.204   25.684
H   1.216   11.459   30.413
H   4.963   12.726   32.243
H   4.808   11.035   32.604
H   3.865   12.428   33.488
H   2.675   6.960   35.594
H   3.209   5.769   34.379
H   0.660   4.934   35.010
H   3.669   4.809   37.790
H   2.442   5.335   38.932
H   2.940   6.458   37.681
H   -0.209   4.097   36.925
H   0.457   3.885   38.499
H   1.962   2.325   38.049
H   1.909   2.350   36.320
H   0.412   0.479   36.624
H   -0.692   1.754   36.881
H   0.293   1.749   39.458
H   -2.123   0.901   38.054
H   -1.983   1.105   39.823
H   -1.732   -0.392   39.039
H   0.992   -0.871   38.185
H   0.201   -1.162   39.712
H   2.209   -0.872   40.728
H   2.113   0.870   40.429
H   4.028   0.619   39.150
H   3.224   -0.225   37.990
H   5.011   -1.454   38.593
H   4.361   -3.589   39.233
H   3.139   -2.963   38.071
H   2.796   -3.124   39.773
H   4.419   -2.177   41.353
H   4.604   -0.439   41.059
H   6.760   -2.456   39.995
H   6.406   -2.214   41.715
H   7.074   -0.167   41.920
H   6.706   0.280   40.249
H   8.816   -1.539   40.098
H   9.247   0.509   42.216
H   10.579   -0.426   41.569
H   9.470   -1.281   42.557
H   9.824   1.163   39.661
H   8.216   1.172   39.111
H   9.279   -0.109   38.535

