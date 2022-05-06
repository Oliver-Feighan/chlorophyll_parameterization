%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_551_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   35.362   1.151   29.763
C   32.996   2.261   32.146
C   37.738   1.207   32.200
C   37.589   0.317   27.374
C   32.876   1.619   27.258
N   35.369   1.740   31.887
C   34.276   2.058   32.707
C   34.689   2.111   34.130
C   36.226   2.165   34.074
C   36.488   1.598   32.609
C   36.708   3.626   34.242
C   34.180   0.942   35.011
C   34.172   1.105   36.614
C   34.767   2.430   37.212
O   34.139   3.484   37.402
O   36.048   2.180   37.636
N   37.406   0.656   29.807
C   38.222   0.897   30.926
C   39.577   0.589   30.533
C   39.496   0.145   29.120
C   38.113   0.378   28.685
C   40.733   0.527   31.463
C   40.610   -0.350   28.177
O   40.442   -0.425   26.966
C   41.921   -0.782   28.792
N   35.157   0.674   27.609
C   36.268   0.427   26.868
C   35.940   0.782   25.446
C   34.427   0.963   25.417
C   34.109   1.129   26.857
C   36.826   1.881   24.705
C   33.585   -0.158   24.712
C   33.177   -1.382   25.682
N   33.361   1.840   29.704
C   32.453   1.947   28.603
C   31.074   2.220   29.101
C   31.280   2.410   30.501
C   32.662   2.135   30.772
C   29.892   2.490   28.215
C   30.671   2.785   31.738
O   29.516   3.099   31.956
C   31.732   2.538   32.863
C   31.763   3.737   33.776
O   32.435   4.668   33.578
O   30.944   3.580   34.889
C   31.042   4.580   35.945
C   36.726   3.250   38.360
C   37.456   2.697   39.604
C   36.990   2.118   40.784
C   35.571   1.924   41.136
C   38.046   1.694   41.815
C   38.769   0.385   41.542
C   39.182   -0.245   42.825
C   40.536   -0.964   42.782
C   40.411   -2.519   43.194
C   41.691   -0.220   43.645
C   42.298   0.838   42.860
C   42.698   1.920   43.832
C   43.705   2.926   43.160
C   45.134   2.375   43.566
C   43.533   4.453   43.500
C   43.562   5.340   42.217
C   44.579   6.439   42.381
C   44.041   7.757   41.702
C   45.169   8.598   41.063
C   43.234   8.552   42.733
H   38.490   1.102   32.984
H   38.240   0.112   26.522
H   32.134   1.774   26.472
H   34.341   3.008   34.643
H   36.714   1.459   34.747
H   36.092   4.208   33.556
H   37.757   3.801   34.002
H   36.613   3.869   35.300
H   34.614   -0.001   34.678
H   33.130   0.902   34.722
H   34.700   0.284   37.098
H   33.115   0.918   36.806
H   41.555   1.003   30.928
H   40.964   -0.479   31.811
H   40.491   1.110   32.352
H   42.497   0.073   29.146
H   42.581   -1.264   28.071
H   41.770   -1.538   29.563
H   36.202   -0.166   24.977
H   34.322   1.951   24.968
H   36.873   1.792   23.620
H   37.851   1.837   25.076
H   36.359   2.858   24.825
H   34.131   -0.480   23.825
H   32.732   0.410   24.342
H   33.384   -2.307   25.144
H   32.124   -1.452   25.956
H   33.749   -1.470   26.605
H   30.370   2.901   27.326
H   29.226   3.252   28.620
H   29.306   1.621   27.916
H   31.298   1.645   33.314
H   30.173   5.234   35.888
H   32.014   5.040   36.123
H   30.994   3.896   36.793
H   36.145   4.077   38.766
H   37.398   3.736   37.653
H   38.518   2.918   39.498
H   34.923   2.253   40.324
H   35.222   2.449   42.025
H   35.472   0.854   41.314
H   37.617   1.818   42.809
H   38.775   2.504   41.812
H   39.545   0.664   40.829
H   38.063   -0.255   41.014
H   38.323   -0.900   42.974
H   39.177   0.525   43.596
H   40.915   -1.153   41.778
H   40.859   -3.222   42.491
H   39.360   -2.793   43.103
H   40.777   -2.647   44.213
H   42.471   -0.973   43.755
H   41.264   0.147   44.579
H   41.493   1.250   42.252
H   43.128   0.397   42.308
H   43.029   1.470   44.768
H   41.770   2.444   44.064
H   43.709   2.837   42.073
H   45.742   2.049   42.722
H   45.157   1.516   44.237
H   45.719   3.139   44.077
H   44.325   4.756   44.185
H   42.649   4.615   44.117
H   42.561   5.755   42.093
H   43.791   4.750   41.330
H   45.551   6.145   41.986
H   44.796   6.746   43.403
H   43.306   7.370   40.996
H   45.953   7.917   40.731
H   45.679   9.254   41.769
H   44.879   9.176   40.185
H   42.297   8.879   42.283
H   43.664   9.402   43.262
H   42.907   7.918   43.557

