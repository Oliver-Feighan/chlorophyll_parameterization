%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1301_chromophore_15 TDDFT with PBE1PBE functional

0 1
Mg   44.953   34.848   28.139
C   43.501   32.599   30.457
C   45.170   37.122   30.705
C   46.339   36.939   25.978
C   44.585   32.466   25.688
N   44.531   34.771   30.339
C   43.987   33.746   31.105
C   43.952   34.091   32.626
C   43.780   35.631   32.419
C   44.599   35.868   31.129
C   42.277   36.064   32.369
C   45.317   33.564   33.235
C   45.319   33.443   34.787
C   44.401   34.280   35.595
O   43.346   33.875   36.055
O   45.022   35.430   35.903
N   45.647   36.752   28.323
C   45.697   37.503   29.480
C   46.231   38.802   29.097
C   46.768   38.714   27.783
C   46.303   37.439   27.272
C   46.301   39.898   30.146
C   47.500   39.793   27.095
O   48.069   39.578   26.016
C   47.638   41.275   27.675
N   45.248   34.771   26.086
C   45.901   35.750   25.384
C   46.279   35.276   24.013
C   45.470   33.882   23.892
C   45.193   33.623   25.336
C   45.769   36.317   22.927
C   46.273   32.829   23.044
C   45.560   32.143   21.908
N   44.326   32.902   28.032
C   44.109   32.095   26.935
C   43.507   30.869   27.375
C   43.066   31.093   28.725
C   43.696   32.339   29.089
C   43.367   29.666   26.650
C   42.479   30.440   29.872
O   41.974   29.369   30.022
C   42.612   31.458   31.055
C   43.102   30.745   32.244
O   44.110   30.079   32.332
O   42.249   31.062   33.341
C   42.542   30.285   34.615
C   44.446   36.131   37.020
C   45.333   37.206   37.505
C   45.236   38.039   38.619
C   43.958   38.087   39.417
C   46.189   39.078   38.904
C   45.740   40.470   38.332
C   46.953   41.150   37.696
C   46.690   42.621   37.471
C   47.528   43.208   36.341
C   46.969   43.453   38.745
C   45.909   44.577   38.953
C   46.391   45.718   39.836
C   46.541   47.096   39.109
C   47.896   47.916   39.259
C   45.372   48.047   39.512
C   44.071   47.654   38.724
C   42.927   47.148   39.673
C   41.579   47.026   39.046
C   40.485   47.504   40.082
C   41.321   45.537   38.619
H   45.185   37.822   31.543
H   46.907   37.599   25.318
H   44.644   31.686   24.926
H   43.029   33.791   33.123
H   44.306   36.087   33.258
H   42.148   36.913   31.697
H   42.091   36.554   33.324
H   41.567   35.286   32.089
H   46.170   34.185   32.962
H   45.526   32.573   32.831
H   46.364   33.616   35.045
H   45.069   32.388   34.902
H   45.989   40.890   29.818
H   47.278   39.928   30.627
H   45.672   39.680   31.009
H   48.180   41.274   28.620
H   46.613   41.623   27.797
H   48.131   41.939   26.965
H   47.335   35.093   23.818
H   44.478   33.958   23.446
H   45.035   35.846   22.274
H   46.551   36.765   22.313
H   45.296   37.190   23.378
H   46.723   32.062   23.675
H   47.154   33.374   22.707
H   45.837   31.095   21.797
H   45.796   32.587   20.941
H   44.480   32.206   22.039
H   44.273   29.064   26.583
H   43.140   29.945   25.621
H   42.588   28.931   26.855
H   41.565   31.739   31.164
H   41.730   30.071   35.310
H   43.168   30.815   35.333
H   43.178   29.412   34.462
H   44.211   35.541   37.905
H   43.439   36.400   36.698
H   46.263   37.224   36.937
H   43.877   37.481   40.320
H   43.044   37.833   38.880
H   43.876   39.087   39.842
H   47.178   38.773   38.560
H   46.279   39.253   39.976
H   45.288   41.013   39.162
H   45.003   40.397   37.532
H   47.134   40.797   36.681
H   47.835   40.990   38.317
H   45.688   42.761   37.067
H   46.865   43.199   35.476
H   48.395   42.571   36.165
H   47.908   44.207   36.555
H   47.896   44.022   38.804
H   47.004   42.738   39.567
H   45.175   44.065   39.576
H   45.471   44.929   38.019
H   47.396   45.558   40.225
H   45.659   45.755   40.643
H   46.408   46.873   38.050
H   48.449   48.007   38.324
H   48.610   47.359   39.865
H   47.736   48.916   39.661
H   45.547   49.063   39.158
H   45.123   48.154   40.567
H   44.240   46.792   38.079
H   43.729   48.467   38.084
H   42.987   47.947   40.412
H   43.230   46.184   40.082
H   41.581   47.560   38.097
H   39.733   46.770   40.372
H   40.009   48.311   39.525
H   40.820   47.908   41.038
H   42.177   44.865   38.674
H   40.987   45.578   37.583
H   40.534   45.038   39.184

