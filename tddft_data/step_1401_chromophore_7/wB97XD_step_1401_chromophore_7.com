%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1401_chromophore_7 TDDFT with wB97XD functional

0 1
Mg   26.500   -0.442   29.118
C   28.129   -0.897   32.123
C   23.530   -0.600   30.871
C   24.771   -0.219   26.172
C   29.259   -1.062   27.371
N   25.948   -0.720   31.223
C   26.801   -0.765   32.312
C   26.116   -0.792   33.706
C   24.574   -0.679   33.250
C   24.673   -0.622   31.690
C   23.682   -1.790   33.848
C   26.491   0.331   34.685
C   25.672   0.380   35.991
C   26.471   0.228   37.270
O   26.873   -0.811   37.772
O   26.381   1.380   38.020
N   24.505   -0.255   28.634
C   23.413   -0.438   29.489
C   22.145   -0.275   28.752
C   22.454   0.016   27.378
C   24.009   -0.157   27.352
C   20.771   -0.439   29.396
C   21.493   0.306   26.195
O   21.851   0.482   25.027
C   20.060   0.349   26.445
N   26.966   -0.757   27.070
C   26.151   -0.399   26.059
C   26.859   -0.236   24.715
C   28.368   -0.400   25.101
C   28.220   -0.696   26.624
C   26.269   -1.366   23.697
C   29.318   0.849   24.911
C   30.762   0.624   24.656
N   28.329   -0.875   29.591
C   29.361   -1.123   28.768
C   30.537   -1.454   29.515
C   30.103   -1.355   30.876
C   28.759   -0.978   30.873
C   31.910   -1.764   28.944
C   30.542   -1.463   32.197
O   31.678   -1.637   32.705
C   29.302   -1.094   33.132
C   29.088   -2.253   34.049
O   28.689   -3.381   33.768
O   29.408   -1.823   35.297
C   29.198   -2.825   36.439
C   26.567   1.278   39.459
C   25.267   1.723   40.116
C   24.816   1.557   41.366
C   25.570   0.794   42.483
C   23.497   2.152   41.826
C   23.625   3.405   42.756
C   22.958   3.126   44.078
C   23.804   3.322   45.399
C   24.152   1.994   46.022
C   23.022   4.173   46.431
C   23.225   5.657   45.975
C   22.194   6.645   46.661
C   23.035   7.770   47.596
C   22.783   7.466   49.096
C   22.518   9.200   47.268
C   23.707   10.141   46.880
C   23.437   11.061   45.696
C   24.546   12.204   45.640
C   23.892   13.644   45.921
C   25.258   12.292   44.311
H   22.620   -0.583   31.474
H   24.158   -0.162   25.270
H   30.202   -1.203   26.839
H   26.348   -1.753   34.166
H   24.125   0.291   33.463
H   23.557   -2.532   33.059
H   22.729   -1.332   34.112
H   24.135   -2.141   34.776
H   26.595   1.382   34.414
H   27.550   0.143   34.866
H   24.812   -0.290   36.010
H   25.169   1.334   36.150
H   20.146   -1.117   28.814
H   20.224   0.504   29.432
H   21.001   -0.897   30.358
H   19.627   0.790   25.547
H   20.017   1.118   27.216
H   19.474   -0.553   26.623
H   26.611   0.784   24.421
H   28.887   -1.255   24.667
H   25.909   -2.111   24.406
H   27.086   -1.831   23.145
H   25.395   -1.013   23.149
H   29.421   1.477   25.796
H   29.085   1.543   24.104
H   31.348   1.361   25.206
H   31.065   0.801   23.624
H   31.077   -0.419   24.696
H   32.114   -1.516   27.903
H   32.037   -2.837   29.089
H   32.705   -1.333   29.552
H   29.565   -0.202   33.700
H   29.538   -3.802   36.095
H   28.171   -2.943   36.785
H   29.879   -2.793   37.290
H   27.199   2.142   39.665
H   27.033   0.439   39.974
H   24.583   2.247   39.448
H   24.827   0.184   42.999
H   26.011   1.490   43.197
H   26.402   0.163   42.171
H   22.876   1.341   42.205
H   22.922   2.360   40.924
H   23.181   4.240   42.216
H   24.680   3.663   42.666
H   22.502   2.142   44.188
H   22.123   3.825   44.034
H   24.760   3.805   45.195
H   23.358   1.536   46.611
H   24.967   2.199   46.716
H   24.491   1.163   45.403
H   23.547   3.996   47.370
H   21.959   3.933   46.421
H   23.159   5.703   44.888
H   24.226   5.999   46.241
H   21.479   6.087   47.266
H   21.568   7.066   45.875
H   24.107   7.724   47.403
H   23.205   8.207   49.775
H   23.351   6.559   49.304
H   21.727   7.280   49.290
H   22.048   9.721   48.102
H   21.856   9.204   46.402
H   24.577   9.516   46.676
H   23.928   10.666   47.810
H   22.402   11.394   45.773
H   23.493   10.473   44.780
H   25.188   12.070   46.510
H   23.720   14.229   45.017
H   24.654   14.180   46.486
H   22.991   13.599   46.532
H   25.681   11.332   44.016
H   25.996   13.089   44.403
H   24.507   12.491   43.547

