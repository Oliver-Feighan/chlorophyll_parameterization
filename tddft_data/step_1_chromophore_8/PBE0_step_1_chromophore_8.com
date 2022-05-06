%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1_chromophore_8 TDDFT with PBE1PBE functional

0 1
Mg   44.805   2.872   46.678
C   42.564   5.571   46.818
C   42.138   0.792   46.357
C   47.013   0.524   46.123
C   47.342   5.378   46.178
N   42.641   3.171   46.669
C   41.942   4.344   46.717
C   40.483   4.049   46.338
C   40.340   2.590   46.757
C   41.789   2.143   46.484
C   39.961   2.426   48.247
C   40.236   4.257   44.798
C   39.269   5.454   44.334
C   39.839   6.574   43.447
O   39.909   7.800   43.735
O   40.258   6.011   42.288
N   44.607   0.937   46.236
C   43.419   0.246   46.169
C   43.720   -1.144   46.025
C   45.142   -1.325   46.069
C   45.676   0.066   46.140
C   42.621   -2.253   45.890
C   45.872   -2.717   46.064
O   45.237   -3.725   46.117
C   47.314   -2.741   46.277
N   46.838   2.954   45.900
C   47.545   1.827   46.153
C   49.019   2.090   46.412
C   49.175   3.562   46.062
C   47.676   4.022   46.033
C   49.446   1.783   47.895
C   49.966   3.979   44.734
C   51.333   4.607   44.784
N   45.003   4.992   46.726
C   46.077   5.865   46.510
C   45.657   7.249   46.552
C   44.294   7.169   46.699
C   43.899   5.793   46.779
C   46.537   8.416   46.377
C   43.082   7.966   46.811
O   42.950   9.194   46.884
C   41.945   6.984   47.000
C   41.328   7.264   48.366
O   42.006   7.201   49.389
O   40.001   7.707   48.253
C   39.367   8.023   49.505
C   40.734   6.971   41.278
C   40.929   6.279   39.954
C   39.945   5.903   39.081
C   38.433   6.215   39.304
C   40.231   5.164   37.780
C   40.224   3.607   38.037
C   41.579   2.833   38.222
C   42.051   2.044   36.889
C   42.658   0.718   37.242
C   42.953   2.919   35.904
C   42.390   2.823   34.428
C   42.039   4.130   33.765
C   42.943   4.450   32.506
C   44.091   5.479   32.843
C   42.191   4.949   31.209
C   42.210   3.914   30.017
C   43.423   4.168   29.071
C   43.036   4.286   27.586
C   44.105   5.092   26.712
C   42.825   2.982   26.907
H   41.262   0.143   46.414
H   47.712   -0.302   46.263
H   48.133   6.130   46.137
H   39.873   4.807   46.830
H   39.594   2.126   46.113
H   39.125   1.738   48.126
H   39.668   3.426   48.568
H   40.731   2.033   48.911
H   39.762   3.350   44.423
H   41.210   4.308   44.312
H   38.982   5.962   45.255
H   38.337   5.132   43.871
H   42.716   -2.971   45.076
H   41.680   -1.717   45.762
H   42.518   -2.855   46.793
H   47.887   -2.174   45.543
H   47.646   -3.779   46.258
H   47.544   -2.496   47.313
H   49.670   1.444   45.822
H   49.627   4.102   46.894
H   49.918   0.808   48.015
H   48.604   1.968   48.562
H   50.202   2.521   48.167
H   49.412   4.488   43.946
H   50.095   2.959   44.371
H   52.118   3.973   44.373
H   51.600   4.817   45.820
H   51.226   5.529   44.213
H   47.399   8.096   45.791
H   46.889   8.667   47.378
H   45.995   9.266   45.965
H   41.251   7.276   46.213
H   39.046   7.156   50.083
H   38.484   8.645   49.361
H   39.982   8.613   50.184
H   41.720   7.217   41.674
H   40.076   7.834   41.171
H   41.954   6.089   39.635
H   38.257   6.543   40.328
H   37.775   5.354   39.189
H   38.187   6.844   38.448
H   41.162   5.507   37.330
H   39.514   5.279   36.967
H   39.716   3.107   37.212
H   39.546   3.342   38.849
H   41.625   2.262   39.150
H   42.353   3.598   38.283
H   41.131   1.716   36.406
H   42.012   -0.120   36.979
H   42.765   0.593   38.320
H   43.605   0.622   36.711
H   44.010   2.654   35.941
H   42.985   3.944   36.272
H   41.403   2.362   34.480
H   43.085   2.245   33.819
H   41.986   4.989   34.433
H   41.048   4.170   33.313
H   43.496   3.585   32.140
H   44.209   6.321   32.162
H   45.058   4.977   32.853
H   44.014   5.897   33.847
H   42.608   5.793   30.661
H   41.161   5.217   31.442
H   41.256   3.929   29.490
H   42.306   2.972   30.557
H   44.006   3.266   29.259
H   44.045   5.047   29.244
H   42.148   4.914   27.507
H   43.632   6.071   26.646
H   44.187   4.595   25.746
H   45.085   5.060   27.188
H   42.139   3.354   26.145
H   42.235   2.317   27.538
H   43.741   2.653   26.415

