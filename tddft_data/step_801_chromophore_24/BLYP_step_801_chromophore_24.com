%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_801_chromophore_24 TDDFT with blyp functional

0 1
Mg   -0.691   44.019   24.875
C   1.750   43.249   27.286
C   -3.148   43.251   27.268
C   -2.947   44.556   22.532
C   1.805   44.433   22.590
N   -0.666   43.223   26.944
C   0.440   43.052   27.749
C   0.019   42.939   29.176
C   -1.517   42.654   29.122
C   -1.825   42.994   27.670
C   -1.924   41.236   29.491
C   0.309   44.325   29.925
C   1.031   44.126   31.324
C   0.696   42.815   32.071
O   1.392   41.809   32.090
O   -0.611   42.846   32.604
N   -2.782   43.961   24.932
C   -3.616   43.624   25.994
C   -5.001   43.794   25.515
C   -4.964   44.208   24.184
C   -3.516   44.267   23.811
C   -6.156   43.516   26.447
C   -6.105   44.507   23.150
O   -5.922   44.861   21.961
C   -7.498   44.448   23.664
N   -0.552   44.407   22.798
C   -1.582   44.527   22.022
C   -1.304   44.760   20.530
C   0.280   44.879   20.631
C   0.567   44.556   22.086
C   -1.846   43.467   19.710
C   0.810   46.260   20.186
C   1.145   47.307   21.333
N   1.290   43.810   24.884
C   2.155   44.006   23.858
C   3.542   43.875   24.380
C   3.443   43.643   25.723
C   2.054   43.575   25.952
C   4.835   44.048   23.576
C   4.156   43.421   26.931
O   5.377   43.296   27.094
C   3.058   43.274   27.990
C   3.350   42.034   28.755
O   2.956   40.936   28.399
O   4.308   42.250   29.704
C   4.672   41.178   30.650
C   -1.009   41.552   33.130
C   -2.345   41.752   33.891
C   -2.792   41.129   35.015
C   -1.983   40.185   35.824
C   -4.020   41.604   35.778
C   -3.739   42.641   36.872
C   -4.444   43.901   36.622
C   -5.613   44.161   37.642
C   -5.097   45.026   38.811
C   -6.728   44.840   36.859
C   -8.046   44.160   37.043
C   -8.309   43.151   35.817
C   -8.738   41.810   36.260
C   -8.206   40.743   35.326
C   -10.335   41.716   36.475
C   -10.733   41.101   37.874
C   -11.832   40.045   37.739
C   -12.412   39.604   39.105
C   -12.086   38.052   39.335
C   -13.893   39.970   39.194
H   -3.859   43.046   28.070
H   -3.660   44.600   21.706
H   2.665   44.544   21.926
H   0.487   42.106   29.701
H   -2.069   43.256   29.843
H   -2.598   40.939   28.688
H   -2.347   41.119   30.488
H   -1.112   40.515   29.389
H   -0.619   44.879   30.067
H   0.915   45.005   29.325
H   0.741   44.912   32.022
H   2.109   44.151   31.165
H   -6.785   42.860   25.846
H   -6.645   44.476   26.618
H   -5.931   43.094   27.427
H   -8.237   44.945   23.035
H   -7.554   44.930   24.640
H   -7.764   43.396   23.765
H   -1.777   45.673   20.167
H   0.787   44.146   20.004
H   -1.919   42.577   20.335
H   -1.127   43.203   18.935
H   -2.756   43.710   19.160
H   0.188   46.778   19.455
H   1.766   46.092   19.690
H   1.266   48.334   20.988
H   2.089   46.865   21.653
H   0.436   47.317   22.161
H   4.696   43.369   22.735
H   5.757   43.716   24.053
H   4.852   45.059   23.168
H   3.132   44.144   28.643
H   5.237   40.394   30.145
H   3.775   40.799   31.139
H   5.339   41.609   31.397
H   -0.261   41.299   33.881
H   -1.223   40.877   32.301
H   -3.001   42.532   33.501
H   -2.434   39.518   36.559
H   -1.417   40.912   36.405
H   -1.306   39.640   35.166
H   -4.359   40.684   36.254
H   -4.751   41.961   35.053
H   -2.667   42.812   36.777
H   -3.867   42.264   37.887
H   -4.719   43.893   35.567
H   -3.636   44.625   36.721
H   -5.843   43.237   38.173
H   -4.051   45.332   38.826
H   -5.075   44.565   39.799
H   -5.708   45.904   39.016
H   -6.512   44.927   35.794
H   -6.762   45.886   37.165
H   -8.905   44.831   37.030
H   -8.136   43.589   37.967
H   -7.400   43.087   35.218
H   -9.022   43.463   35.053
H   -8.320   41.598   37.244
H   -7.746   41.114   34.410
H   -9.024   40.097   35.007
H   -7.533   40.016   35.780
H   -10.773   41.026   35.754
H   -10.740   42.728   36.484
H   -11.048   41.949   38.481
H   -9.848   40.551   38.194
H   -11.450   39.199   37.169
H   -12.579   40.421   37.040
H   -11.941   40.065   39.973
H   -11.053   38.061   39.683
H   -12.341   37.361   38.532
H   -12.628   37.642   40.187
H   -13.977   40.826   39.863
H   -14.505   39.242   39.727
H   -14.412   40.274   38.285

