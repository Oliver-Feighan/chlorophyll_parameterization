%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1051_chromophore_3 TDDFT with wB97XD functional

0 1
Mg   1.669   8.235   26.938
C   2.010   10.279   29.699
C   2.505   5.477   28.889
C   1.759   6.402   24.187
C   1.845   11.244   24.902
N   2.183   7.851   29.044
C   2.056   8.888   29.968
C   2.184   8.303   31.327
C   2.545   6.721   31.166
C   2.512   6.694   29.610
C   3.953   6.447   31.685
C   0.915   8.493   32.155
C   1.183   8.957   33.641
C   2.412   8.471   34.375
O   3.401   9.110   34.712
O   2.201   7.177   34.707
N   1.890   6.196   26.600
C   2.227   5.214   27.507
C   2.208   3.893   26.842
C   2.076   4.171   25.460
C   1.869   5.585   25.375
C   2.386   2.583   27.598
C   2.161   3.149   24.301
O   2.104   3.547   23.170
C   2.331   1.729   24.535
N   1.618   8.795   24.841
C   1.691   7.785   23.897
C   1.869   8.292   22.553
C   1.527   9.848   22.768
C   1.641   10.035   24.292
C   3.328   8.079   22.025
C   0.234   10.522   22.262
C   0.403   11.789   21.368
N   1.896   10.287   27.155
C   1.884   11.413   26.285
C   1.894   12.658   27.059
C   2.050   12.254   28.396
C   1.961   10.803   28.384
C   2.124   14.095   26.560
C   2.269   12.702   29.740
O   2.468   13.830   30.160
C   2.099   11.430   30.667
C   3.284   11.487   31.502
O   4.389   11.006   31.246
O   3.097   12.110   32.700
C   4.179   12.458   33.566
C   3.220   6.479   35.527
C   2.450   5.487   36.343
C   2.301   5.387   37.701
C   3.099   6.345   38.631
C   1.163   4.481   38.268
C   1.083   3.044   37.869
C   1.032   2.033   39.100
C   2.181   0.932   39.177
C   3.319   1.597   39.921
C   1.786   -0.475   39.803
C   2.926   -1.552   39.680
C   3.547   -1.993   40.977
C   4.564   -3.248   40.854
C   3.936   -4.572   41.341
C   5.866   -2.999   41.620
C   6.810   -1.975   40.963
C   7.388   -0.906   41.945
C   8.836   -0.562   41.573
C   9.734   -0.469   42.781
C   8.783   0.763   40.755
H   2.692   4.639   29.564
H   1.605   5.801   23.288
H   1.982   12.167   24.334
H   3.032   8.776   31.823
H   1.776   6.119   31.650
H   4.656   6.608   30.867
H   3.984   5.384   31.919
H   4.204   6.988   32.597
H   0.349   7.564   32.221
H   0.239   9.137   31.593
H   0.384   8.576   34.278
H   1.115   10.043   33.706
H   1.770   1.784   27.184
H   2.282   2.628   28.682
H   3.438   2.425   27.360
H   2.548   1.169   23.625
H   1.354   1.324   24.797
H   3.205   1.575   25.167
H   1.182   7.809   21.858
H   2.386   10.364   22.338
H   3.197   7.289   21.285
H   3.947   7.660   22.818
H   3.797   8.983   21.637
H   -0.382   10.849   23.099
H   -0.304   9.834   21.610
H   0.549   12.717   21.920
H   -0.563   11.997   20.907
H   1.265   11.760   20.702
H   1.176   14.521   26.230
H   2.898   13.891   25.820
H   2.509   14.721   27.365
H   1.190   11.457   31.268
H   4.444   11.685   34.286
H   3.821   13.204   34.276
H   5.089   12.620   32.988
H   3.782   7.164   36.162
H   3.903   5.970   34.846
H   1.852   4.773   35.777
H   3.853   5.747   39.144
H   2.341   6.630   39.361
H   3.428   7.228   38.084
H   0.254   5.054   38.085
H   1.346   4.492   39.342
H   1.967   2.823   37.271
H   0.193   3.061   37.240
H   0.142   1.403   39.095
H   0.872   2.531   40.057
H   2.470   0.744   38.143
H   3.551   2.474   39.317
H   4.232   1.004   39.958
H   2.952   1.973   40.876
H   0.966   -0.914   39.236
H   1.398   -0.369   40.816
H   3.675   -1.123   39.014
H   2.594   -2.416   39.104
H   2.733   -2.221   41.665
H   4.084   -1.136   41.383
H   4.871   -3.178   39.811
H   3.281   -4.339   42.181
H   4.654   -5.303   41.711
H   3.378   -4.931   40.476
H   6.381   -3.957   41.692
H   5.629   -2.588   42.601
H   6.209   -1.432   40.234
H   7.502   -2.473   40.283
H   7.315   -1.270   42.970
H   6.699   -0.062   41.961
H   9.284   -1.362   40.985
H   10.517   -1.221   42.690
H   9.194   -0.721   43.694
H   10.221   0.506   42.819
H   7.737   0.985   40.544
H   9.339   0.707   39.819
H   9.084   1.646   41.319

