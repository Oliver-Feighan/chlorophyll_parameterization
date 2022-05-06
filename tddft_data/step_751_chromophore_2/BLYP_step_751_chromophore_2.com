%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_751_chromophore_2 TDDFT with blyp functional

0 1
Mg   3.330   0.793   44.727
C   6.171   2.876   44.431
C   1.515   3.180   43.096
C   0.832   -1.461   44.394
C   5.537   -1.844   45.755
N   3.814   2.687   43.666
C   5.026   3.391   43.814
C   4.956   4.751   43.205
C   3.414   4.890   42.933
C   2.863   3.464   43.192
C   2.673   5.852   43.893
C   5.779   4.829   41.838
C   6.181   3.527   41.069
C   6.473   3.755   39.495
O   6.867   4.718   38.905
O   6.232   2.575   38.866
N   1.366   0.923   44.004
C   0.796   2.000   43.399
C   -0.540   1.702   43.088
C   -0.834   0.299   43.392
C   0.444   -0.144   43.987
C   -1.440   2.856   42.649
C   -2.150   -0.456   43.179
O   -3.072   0.116   42.602
C   -2.313   -1.859   43.597
N   3.262   -1.369   44.870
C   2.048   -2.036   44.768
C   2.175   -3.535   45.150
C   3.737   -3.682   45.500
C   4.234   -2.212   45.375
C   1.132   -3.866   46.333
C   4.510   -4.707   44.587
C   5.113   -5.888   45.324
N   5.399   0.588   45.231
C   6.102   -0.530   45.638
C   7.473   -0.148   45.837
C   7.519   1.181   45.506
C   6.277   1.570   44.997
C   8.532   -1.065   46.302
C   8.388   2.321   45.389
O   9.521   2.432   45.804
C   7.568   3.480   44.700
C   7.632   4.703   45.576
O   8.627   5.422   45.660
O   6.436   4.817   46.351
C   6.535   5.892   47.313
C   6.521   2.523   37.383
C   5.162   2.219   36.690
C   4.915   1.464   35.540
C   5.998   1.039   34.572
C   3.462   1.128   35.138
C   2.876   2.094   34.056
C   2.446   1.402   32.773
C   0.853   1.292   32.649
C   0.345   -0.183   32.771
C   0.329   1.912   31.308
C   0.133   3.442   31.418
C   -1.333   3.898   31.093
C   -1.335   4.849   29.780
C   -2.347   6.035   29.889
C   -1.578   4.065   28.452
C   -1.310   4.663   27.030
C   -2.549   4.613   26.087
C   -2.103   3.968   24.673
C   -3.045   4.578   23.596
C   -2.307   2.437   24.887
H   0.875   3.973   42.704
H   0.041   -2.213   44.435
H   6.145   -2.636   46.197
H   5.387   5.439   43.933
H   3.308   5.169   41.884
H   2.137   5.358   44.703
H   2.080   6.565   43.319
H   3.358   6.529   44.403
H   6.773   5.219   42.061
H   5.321   5.554   41.164
H   5.444   2.749   41.268
H   7.061   3.064   41.516
H   -1.284   3.203   41.628
H   -1.459   3.651   43.395
H   -2.485   2.548   42.614
H   -2.236   -1.904   44.684
H   -1.519   -2.361   43.045
H   -3.263   -2.321   43.330
H   1.933   -4.115   44.259
H   3.814   -4.033   46.529
H   0.364   -4.497   45.887
H   0.699   -2.967   46.774
H   1.632   -4.411   47.134
H   5.409   -4.198   44.239
H   3.903   -5.037   43.743
H   4.565   -6.786   45.036
H   5.133   -5.904   46.414
H   6.130   -6.028   44.959
H   8.521   -1.895   45.596
H   8.276   -1.264   47.343
H   9.568   -0.782   46.116
H   8.100   3.704   43.776
H   6.124   6.802   46.877
H   7.559   6.053   47.650
H   5.935   5.634   48.186
H   7.223   1.698   37.262
H   6.999   3.400   36.945
H   4.388   2.452   37.422
H   6.277   0.087   35.024
H   6.858   1.702   34.670
H   5.697   0.957   33.528
H   2.962   1.170   36.105
H   3.328   0.097   34.812
H   3.591   2.875   33.795
H   2.003   2.526   34.545
H   2.911   0.425   32.641
H   2.715   1.988   31.895
H   0.328   1.832   33.437
H   0.450   -0.680   31.807
H   -0.729   -0.210   32.952
H   0.898   -0.702   33.554
H   -0.557   1.332   31.052
H   1.109   1.850   30.549
H   0.725   3.921   30.638
H   0.375   3.895   32.379
H   -1.784   4.323   31.989
H   -1.979   3.035   30.933
H   -0.414   5.430   29.727
H   -3.085   6.007   29.088
H   -1.860   7.005   29.787
H   -2.932   5.950   30.805
H   -2.529   3.534   28.457
H   -0.858   3.247   28.489
H   -0.424   4.147   26.659
H   -0.997   5.698   27.166
H   -2.943   5.628   26.037
H   -3.284   3.866   26.390
H   -1.089   4.191   24.340
H   -2.487   5.494   23.406
H   -3.991   4.920   24.017
H   -3.239   3.927   22.743
H   -2.301   2.025   25.896
H   -1.664   1.901   24.188
H   -3.296   2.203   24.493

