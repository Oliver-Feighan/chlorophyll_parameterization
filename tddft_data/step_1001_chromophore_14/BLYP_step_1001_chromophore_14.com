%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1001_chromophore_14 TDDFT with blyp functional

0 1
Mg   46.387   44.799   43.806
C   43.091   43.694   43.386
C   47.514   41.503   43.278
C   49.662   45.851   43.724
C   45.268   48.019   43.939
N   45.430   42.814   43.231
C   44.102   42.614   43.197
C   43.730   41.089   42.964
C   45.120   40.410   43.237
C   46.092   41.644   43.132
C   45.166   39.809   44.617
C   43.088   40.773   41.564
C   43.641   41.525   40.342
C   43.254   40.979   38.941
O   42.372   40.185   38.727
O   43.993   41.568   37.943
N   48.372   43.816   43.530
C   48.578   42.423   43.440
C   49.984   42.166   43.483
C   50.657   43.434   43.639
C   49.531   44.437   43.665
C   50.612   40.822   43.308
C   52.165   43.538   43.786
O   52.881   42.571   43.766
C   52.849   44.883   43.842
N   47.356   46.747   43.585
C   48.645   46.861   43.898
C   49.011   48.310   44.238
C   47.583   49.040   44.030
C   46.639   47.863   43.808
C   49.636   48.375   45.735
C   47.485   50.131   42.854
C   47.584   51.570   43.283
N   44.638   45.707   43.659
C   44.287   47.023   43.751
C   42.813   47.189   43.672
C   42.324   45.888   43.526
C   43.476   45.027   43.536
C   42.026   48.426   43.672
C   41.149   45.058   43.432
O   39.982   45.400   43.429
C   41.584   43.592   43.333
C   40.986   42.878   44.501
O   40.087   42.029   44.421
O   41.544   43.403   45.677
C   40.839   42.861   46.899
C   43.352   41.395   36.639
C   44.330   41.868   35.587
C   45.212   41.139   34.844
C   45.546   39.635   35.127
C   46.023   41.864   33.852
C   45.489   41.956   32.339
C   44.844   43.326   31.940
C   45.578   43.975   30.662
C   46.321   45.241   31.043
C   44.631   44.236   29.435
C   44.594   43.013   28.460
C   43.671   43.294   27.276
C   44.480   43.831   26.041
C   43.648   44.905   25.308
C   44.856   42.743   25.016
C   46.185   43.041   24.332
C   46.387   42.682   22.829
C   47.376   43.608   22.135
C   46.709   44.780   21.306
C   48.311   42.821   21.189
H   47.748   40.452   43.455
H   50.678   46.246   43.800
H   44.905   49.003   44.244
H   42.987   40.682   43.651
H   45.415   39.621   42.545
H   44.259   39.243   44.825
H   45.442   40.527   45.390
H   45.975   39.078   44.597
H   42.051   41.007   41.803
H   43.234   39.724   41.306
H   44.728   41.555   40.418
H   43.187   42.506   40.481
H   51.167   40.742   42.373
H   49.905   39.994   43.363
H   51.117   40.659   44.260
H   52.329   45.266   44.720
H   52.823   45.471   42.925
H   53.916   44.785   44.046
H   49.668   48.649   43.437
H   47.230   49.485   44.961
H   49.143   49.205   46.240
H   50.689   48.654   45.698
H   49.479   47.427   46.250
H   46.490   50.121   42.409
H   48.189   50.003   42.032
H   47.556   51.704   44.364
H   46.709   52.131   42.955
H   48.466   52.138   42.989
H   41.492   48.581   44.610
H   41.261   48.273   42.910
H   42.487   49.401   43.514
H   41.278   43.138   42.391
H   40.904   41.773   46.901
H   39.780   43.115   46.840
H   41.262   43.250   47.826
H   42.441   41.959   36.440
H   43.194   40.350   36.372
H   44.223   42.938   35.409
H   45.110   39.106   34.279
H   45.025   39.169   35.963
H   46.607   39.384   35.118
H   47.015   41.416   33.789
H   46.102   42.886   34.222
H   44.809   41.137   32.103
H   46.305   41.779   31.639
H   44.859   44.030   32.772
H   43.763   43.243   31.824
H   46.421   43.410   30.263
H   45.637   45.984   31.454
H   46.722   45.625   30.105
H   47.125   45.031   31.749
H   44.835   45.189   28.946
H   43.581   44.246   29.727
H   44.334   42.049   28.896
H   45.648   42.950   28.187
H   42.918   44.019   27.584
H   42.952   42.495   27.095
H   45.445   44.197   26.390
H   44.469   45.512   24.927
H   42.999   45.529   25.922
H   43.006   44.527   24.511
H   44.102   42.604   24.241
H   44.972   41.754   25.460
H   46.831   42.474   25.003
H   46.502   44.066   24.527
H   45.424   42.888   22.363
H   46.674   41.644   22.658
H   47.999   44.089   22.889
H   45.692   44.874   21.686
H   46.718   44.526   20.246
H   47.188   45.719   21.586
H   48.974   42.101   21.669
H   48.996   43.533   20.728
H   47.693   42.217   20.525

