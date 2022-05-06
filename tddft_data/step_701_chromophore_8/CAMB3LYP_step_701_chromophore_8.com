%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_701_chromophore_8 TDDFT with cam-b3lyp functional

0 1
Mg   44.916   3.809   46.695
C   42.638   6.312   46.014
C   42.279   1.554   46.853
C   47.219   1.492   47.115
C   47.495   6.240   46.103
N   42.723   3.947   46.363
C   42.010   5.038   46.009
C   40.517   4.684   45.728
C   40.418   3.191   46.304
C   41.887   2.875   46.491
C   39.675   3.018   47.651
C   40.209   4.650   44.177
C   39.260   5.703   43.630
C   39.501   6.559   42.276
O   40.098   7.636   42.137
O   38.845   5.890   41.210
N   44.822   1.786   46.905
C   43.637   1.069   46.968
C   44.035   -0.318   47.002
C   45.462   -0.426   47.170
C   45.933   0.952   47.033
C   43.040   -1.483   46.986
C   46.353   -1.724   47.241
O   45.831   -2.800   47.091
C   47.830   -1.714   47.583
N   47.059   3.820   46.523
C   47.817   2.761   46.916
C   49.294   3.135   47.120
C   49.374   4.490   46.354
C   47.883   4.921   46.276
C   49.779   3.095   48.638
C   50.061   4.444   45.010
C   51.598   4.679   44.969
N   45.103   5.876   46.264
C   46.183   6.713   46.062
C   45.707   8.089   45.964
C   44.300   8.016   45.916
C   44.002   6.582   46.096
C   46.549   9.297   45.842
C   43.079   8.700   45.826
O   42.899   9.874   45.642
C   41.954   7.716   45.937
C   41.120   8.124   47.093
O   41.632   8.391   48.202
O   39.801   8.316   46.693
C   38.905   8.613   47.787
C   38.905   6.439   39.814
C   40.254   6.063   39.234
C   40.533   5.988   37.909
C   39.463   6.333   36.817
C   41.965   5.582   37.434
C   42.336   4.090   37.508
C   43.778   3.926   37.801
C   44.434   2.779   37.008
C   45.006   1.607   37.820
C   45.430   3.333   35.963
C   44.667   3.989   34.765
C   45.236   3.450   33.417
C   44.046   2.888   32.626
C   44.105   1.323   32.828
C   44.202   3.246   31.156
C   42.911   2.995   30.307
C   42.625   4.129   29.258
C   43.742   4.155   28.161
C   43.147   3.622   26.854
C   44.212   5.587   27.965
H   41.474   0.817   46.806
H   47.884   0.722   47.511
H   48.350   6.918   46.155
H   39.977   5.421   46.323
H   39.875   2.464   45.701
H   38.804   2.364   47.640
H   39.436   4.009   48.039
H   40.372   2.548   48.346
H   39.611   3.749   44.036
H   41.120   4.523   43.592
H   39.068   6.433   44.416
H   38.361   5.142   43.378
H   42.889   -1.858   47.998
H   43.356   -2.320   46.362
H   42.103   -1.114   46.569
H   48.119   -1.009   48.363
H   48.331   -1.505   46.638
H   48.120   -2.738   47.818
H   49.873   2.357   46.622
H   49.939   5.252   46.891
H   50.337   2.186   48.862
H   48.953   3.191   49.342
H   50.463   3.901   48.906
H   49.573   5.092   44.281
H   49.791   3.434   44.701
H   51.875   5.016   43.970
H   52.059   3.744   45.286
H   51.864   5.457   45.685
H   47.504   9.046   46.303
H   45.954   10.154   46.157
H   46.721   9.569   44.800
H   41.290   7.810   45.078
H   38.989   7.690   48.360
H   37.874   8.778   47.475
H   39.113   9.543   48.316
H   38.736   7.514   39.743
H   38.273   5.869   39.133
H   40.924   5.811   40.056
H   39.020   5.403   36.461
H   39.966   6.807   35.975
H   38.644   6.965   37.161
H   42.753   6.162   37.914
H   42.015   5.873   36.385
H   41.992   3.755   36.529
H   41.828   3.546   38.304
H   43.931   3.903   38.880
H   44.268   4.830   37.439
H   43.634   2.236   36.505
H   44.917   1.902   38.866
H   46.057   1.453   37.575
H   44.496   0.675   37.574
H   46.103   2.510   35.723
H   46.014   4.169   36.349
H   44.800   5.070   34.721
H   43.618   3.716   34.878
H   45.992   2.666   33.461
H   45.582   4.343   32.897
H   43.084   3.247   32.992
H   44.883   0.970   33.505
H   44.263   0.751   31.913
H   43.198   1.087   33.384
H   45.019   2.683   30.703
H   44.456   4.300   31.045
H   42.089   2.878   31.013
H   42.967   2.087   29.707
H   42.464   5.058   29.805
H   41.644   3.908   28.838
H   44.597   3.501   28.334
H   43.917   3.588   26.084
H   42.326   4.275   26.557
H   42.673   2.646   26.960
H   44.008   5.979   26.969
H   45.286   5.626   28.145
H   43.774   6.283   28.680

