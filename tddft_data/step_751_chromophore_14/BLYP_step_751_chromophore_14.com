%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_751_chromophore_14 TDDFT with blyp functional

0 1
Mg   46.722   44.376   43.768
C   43.334   43.651   43.121
C   47.385   41.046   42.800
C   49.980   45.102   43.751
C   45.844   47.743   43.730
N   45.443   42.577   43.049
C   44.056   42.517   42.889
C   43.585   41.075   42.425
C   44.916   40.230   42.386
C   46.004   41.367   42.763
C   44.967   38.911   43.256
C   42.725   41.039   41.136
C   43.293   41.969   40.006
C   43.107   41.538   38.490
O   42.247   40.767   38.055
O   44.054   42.136   37.680
N   48.400   43.203   43.405
C   48.517   41.885   42.989
C   49.911   41.584   42.766
C   50.649   42.775   43.109
C   49.664   43.766   43.476
C   50.417   40.154   42.399
C   52.172   42.911   42.975
O   52.765   41.928   42.691
C   53.021   44.207   43.242
N   47.732   46.170   43.560
C   49.072   46.160   43.804
C   49.617   47.624   43.895
C   48.350   48.518   43.915
C   47.231   47.412   43.687
C   50.779   47.938   45.004
C   48.304   49.682   42.802
C   48.207   51.093   43.440
N   44.919   45.471   43.677
C   44.721   46.830   43.661
C   43.326   47.159   43.610
C   42.723   45.941   43.463
C   43.735   44.933   43.522
C   42.676   48.521   43.642
C   41.458   45.236   43.219
O   40.314   45.668   43.026
C   41.840   43.753   43.035
C   41.170   42.966   44.101
O   40.118   42.395   43.867
O   41.671   43.171   45.400
C   40.732   42.541   46.404
C   43.883   41.861   36.204
C   45.162   42.279   35.447
C   45.963   41.681   34.547
C   45.658   40.309   33.953
C   47.076   42.465   33.922
C   46.508   43.335   32.773
C   47.004   42.967   31.411
C   46.125   43.154   30.148
C   45.504   44.524   30.031
C   45.070   42.027   29.963
C   44.953   41.442   28.541
C   44.087   42.394   27.629
C   45.037   43.266   26.659
C   44.606   44.778   26.451
C   45.196   42.602   25.307
C   46.530   43.042   24.555
C   46.568   42.352   23.126
C   47.127   43.136   21.941
C   46.328   43.123   20.610
C   48.600   42.792   21.668
H   47.438   39.976   42.585
H   51.053   45.281   43.849
H   45.535   48.770   43.935
H   42.903   40.718   43.196
H   45.223   40.118   41.346
H   45.647   38.808   44.102
H   45.072   38.087   42.550
H   43.953   38.938   43.657
H   41.738   41.413   41.406
H   42.655   40.067   40.649
H   44.363   42.161   40.092
H   42.727   42.900   39.982
H   49.542   39.512   42.289
H   51.148   39.756   43.102
H   50.931   40.302   41.449
H   53.177   44.419   44.299
H   52.562   44.979   42.623
H   53.907   44.083   42.619
H   50.029   47.744   42.893
H   48.207   48.792   44.960
H   50.714   47.336   45.911
H   50.525   48.966   45.261
H   51.757   47.964   44.524
H   47.459   49.564   42.123
H   49.214   49.668   42.201
H   47.224   51.450   43.132
H   48.991   51.805   43.184
H   48.062   50.989   44.516
H   43.413   49.319   43.547
H   42.209   48.546   44.626
H   41.943   48.580   42.837
H   41.383   43.556   42.066
H   39.831   43.154   46.429
H   41.323   42.689   47.308
H   40.493   41.508   46.151
H   43.005   42.333   35.764
H   43.719   40.788   36.099
H   45.377   43.327   35.660
H   44.918   39.750   34.526
H   46.568   39.715   34.039
H   45.356   40.469   32.917
H   47.890   41.830   33.573
H   47.554   43.151   34.620
H   46.710   44.392   32.942
H   45.433   43.154   32.786
H   47.232   41.905   31.510
H   47.897   43.576   31.269
H   46.714   42.936   29.257
H   46.132   45.120   29.369
H   45.425   45.064   30.974
H   44.487   44.527   29.638
H   44.083   42.411   30.220
H   45.134   41.173   30.637
H   44.629   40.420   28.741
H   45.970   41.388   28.153
H   43.456   43.013   28.267
H   43.414   41.744   27.070
H   46.036   43.386   27.079
H   44.957   44.994   25.442
H   45.039   45.507   27.137
H   43.521   44.841   26.372
H   44.402   42.996   24.673
H   45.052   41.522   25.284
H   47.390   42.666   25.108
H   46.666   44.118   24.453
H   45.593   41.936   22.874
H   47.184   41.454   23.178
H   47.118   44.159   22.318
H   46.935   43.147   19.705
H   45.666   43.989   20.631
H   45.678   42.263   20.449
H   48.896   42.710   20.623
H   48.832   41.842   22.150
H   49.267   43.560   22.060
