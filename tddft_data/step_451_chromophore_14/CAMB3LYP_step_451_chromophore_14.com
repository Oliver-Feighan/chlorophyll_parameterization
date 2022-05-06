%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_451_chromophore_14 TDDFT with cam-b3lyp functional

0 1
Mg   46.765   44.098   43.613
C   43.279   43.573   43.308
C   47.303   40.845   42.668
C   50.106   44.814   42.764
C   46.124   47.573   43.575
N   45.431   42.442   43.016
C   44.030   42.458   42.987
C   43.500   41.023   42.706
C   44.854   40.196   42.618
C   45.930   41.228   42.814
C   44.991   39.056   43.678
C   42.533   40.893   41.487
C   42.830   41.744   40.189
C   42.323   41.267   38.890
O   41.734   40.196   38.688
O   42.517   42.277   37.961
N   48.478   42.964   42.991
C   48.497   41.623   42.697
C   49.820   41.153   42.401
C   50.668   42.279   42.450
C   49.767   43.441   42.788
C   50.074   39.692   42.056
C   52.186   42.364   42.117
O   52.760   41.308   41.789
C   53.064   43.591   42.040
N   47.890   45.869   43.173
C   49.281   45.917   42.947
C   49.802   47.402   42.849
C   48.583   48.165   43.426
C   47.437   47.158   43.347
C   51.230   47.661   43.486
C   48.335   49.535   42.668
C   48.463   50.746   43.608
N   45.087   45.379   43.745
C   44.998   46.746   43.746
C   43.545   47.081   43.898
C   42.884   45.871   43.782
C   43.815   44.870   43.549
C   42.855   48.439   43.974
C   41.556   45.311   43.733
O   40.402   45.757   43.816
C   41.827   43.693   43.455
C   41.186   42.925   44.606
O   40.289   42.128   44.522
O   41.690   43.396   45.809
C   41.053   42.896   47.034
C   42.012   42.074   36.605
C   43.208   42.358   35.783
C   44.108   41.564   35.210
C   44.125   40.033   35.444
C   45.364   42.103   34.514
C   45.417   41.948   32.925
C   44.918   43.281   32.197
C   45.010   43.281   30.648
C   45.729   44.561   30.172
C   43.609   43.140   29.958
C   43.543   42.081   28.888
C   43.077   42.621   27.472
C   44.388   42.678   26.619
C   44.133   43.659   25.431
C   44.721   41.258   26.048
C   46.277   41.097   25.924
C   46.757   40.556   24.560
C   48.070   41.177   24.072
C   47.828   42.597   23.589
C   48.868   40.300   23.073
H   47.345   39.810   42.323
H   51.150   45.126   42.833
H   45.848   48.613   43.757
H   42.891   40.705   43.553
H   44.970   39.879   41.582
H   45.097   38.081   43.201
H   44.087   38.956   44.278
H   45.893   39.347   44.216
H   41.548   41.047   41.926
H   42.695   39.890   41.092
H   43.903   41.899   40.076
H   42.340   42.707   40.330
H   50.004   39.548   40.978
H   49.295   39.042   42.456
H   50.997   39.333   42.513
H   52.910   44.160   42.957
H   52.659   44.183   41.220
H   54.120   43.357   41.906
H   49.866   47.595   41.778
H   48.771   48.458   44.459
H   51.256   48.639   43.967
H   51.996   47.569   42.716
H   51.442   46.814   44.137
H   47.389   49.519   42.127
H   49.127   49.799   41.966
H   49.161   50.572   44.427
H   47.488   51.169   43.852
H   49.033   51.521   43.095
H   42.243   48.572   43.082
H   43.616   49.186   44.200
H   42.186   48.299   44.823
H   41.209   43.407   42.603
H   41.825   42.359   47.585
H   40.266   42.175   46.813
H   40.669   43.754   47.586
H   41.339   42.863   36.270
H   41.634   41.075   36.388
H   43.219   43.367   35.371
H   43.614   39.790   36.375
H   45.026   39.450   35.252
H   43.572   39.675   34.576
H   46.176   41.601   35.040
H   45.436   43.168   34.736
H   44.758   41.118   32.672
H   46.494   41.836   32.801
H   45.501   44.083   32.651
H   43.923   43.435   32.614
H   45.659   42.468   30.323
H   45.708   44.670   29.088
H   46.778   44.578   30.470
H   45.251   45.395   30.685
H   43.284   44.110   29.581
H   42.898   42.748   30.684
H   42.885   41.241   29.112
H   44.509   41.579   28.836
H   42.673   43.633   27.514
H   42.423   41.815   27.139
H   45.186   42.958   27.308
H   44.873   43.596   24.633
H   44.035   44.663   25.845
H   43.137   43.429   25.054
H   44.292   41.067   25.065
H   44.385   40.427   26.668
H   46.500   40.278   26.609
H   46.855   41.962   26.250
H   46.060   40.721   23.739
H   46.902   39.476   24.538
H   48.708   41.250   24.953
H   46.851   42.934   23.935
H   47.808   42.726   22.507
H   48.702   43.094   24.010
H   48.787   40.630   22.037
H   48.671   39.234   23.185
H   49.930   40.326   23.319

