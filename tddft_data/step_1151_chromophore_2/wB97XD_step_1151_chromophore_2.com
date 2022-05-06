%nproc=24
%mem=75GB
#p wB97XD/Def2SVP td=(nstates=5)

step_1151_chromophore_2 TDDFT with wB97XD functional

0 1
Mg   3.600   0.362   44.387
C   6.429   2.401   44.069
C   1.808   2.968   43.052
C   1.098   -1.696   43.772
C   5.638   -2.367   45.182
N   4.117   2.402   43.467
C   5.356   2.952   43.435
C   5.323   4.341   42.786
C   3.735   4.674   42.795
C   3.206   3.247   43.065
C   3.416   5.659   43.989
C   6.075   4.397   41.374
C   6.186   2.986   40.631
C   6.451   2.930   39.157
O   6.769   3.841   38.450
O   6.665   1.636   38.789
N   1.645   0.630   43.642
C   1.102   1.785   43.254
C   -0.344   1.561   43.063
C   -0.576   0.182   43.336
C   0.757   -0.366   43.562
C   -1.277   2.612   42.692
C   -1.930   -0.489   43.331
O   -2.920   0.167   43.049
C   -2.024   -1.935   43.668
N   3.442   -1.821   44.160
C   2.242   -2.418   44.091
C   2.242   -3.893   44.457
C   3.743   -4.124   44.780
C   4.357   -2.740   44.658
C   1.251   -4.416   45.499
C   4.482   -5.077   43.775
C   5.216   -6.247   44.388
N   5.633   0.066   44.664
C   6.273   -1.086   45.096
C   7.596   -0.705   45.421
C   7.756   0.618   45.022
C   6.517   1.048   44.534
C   8.597   -1.721   45.922
C   8.644   1.711   45.048
O   9.822   1.791   45.449
C   7.866   2.904   44.304
C   7.920   4.062   45.220
O   8.825   4.905   45.309
O   6.788   4.068   45.965
C   6.358   5.243   46.655
C   6.808   1.552   37.322
C   5.407   1.589   36.710
C   5.067   1.328   35.466
C   6.098   1.405   34.350
C   3.596   0.946   35.009
C   2.721   1.983   34.272
C   2.135   1.630   32.847
C   0.625   1.806   32.796
C   -0.123   0.784   33.695
C   0.098   1.820   31.327
C   -0.366   3.190   30.995
C   -1.790   3.241   30.246
C   -1.860   4.158   29.009
C   -2.353   5.606   29.302
C   -2.903   3.513   27.980
C   -2.148   3.061   26.643
C   -3.196   2.616   25.577
C   -2.912   3.322   24.190
C   -4.205   3.637   23.466
C   -1.720   2.773   23.325
H   1.273   3.777   42.550
H   0.316   -2.442   43.614
H   6.163   -3.225   45.607
H   5.779   5.062   43.465
H   3.468   5.066   41.814
H   2.571   6.282   43.696
H   4.247   6.356   44.096
H   3.143   5.097   44.882
H   7.076   4.825   41.423
H   5.516   5.099   40.756
H   5.263   2.433   40.801
H   7.090   2.561   41.067
H   -1.580   3.007   43.662
H   -2.119   2.284   42.082
H   -0.728   3.453   42.270
H   -1.411   -2.116   44.550
H   -1.737   -2.488   42.774
H   -3.085   -2.004   43.910
H   2.035   -4.432   43.533
H   3.830   -4.449   45.817
H   0.581   -3.616   45.812
H   1.748   -4.763   46.405
H   0.688   -5.264   45.108
H   5.203   -4.566   43.136
H   3.710   -5.563   43.178
H   5.091   -6.320   45.468
H   6.274   -6.213   44.127
H   4.788   -7.172   44.000
H   9.642   -1.507   45.696
H   8.406   -2.675   45.432
H   8.383   -1.916   46.972
H   8.434   3.224   43.430
H   5.515   5.720   46.155
H   7.157   5.984   46.672
H   6.207   4.946   47.693
H   7.243   0.554   37.272
H   7.452   2.257   36.796
H   4.611   1.482   37.447
H   6.602   0.457   34.159
H   6.791   2.154   34.731
H   5.715   1.792   33.406
H   3.113   0.830   35.980
H   3.713   -0.025   34.530
H   3.467   2.773   34.175
H   1.931   2.310   34.948
H   2.518   0.668   32.508
H   2.606   2.317   32.143
H   0.411   2.762   33.275
H   -0.762   0.088   33.152
H   -0.664   1.432   34.384
H   0.619   0.147   34.176
H   -0.777   1.191   31.163
H   0.809   1.491   30.569
H   0.394   3.615   30.340
H   -0.506   3.899   31.810
H   -2.531   3.741   30.869
H   -2.119   2.217   30.070
H   -0.889   4.225   28.518
H   -2.033   6.146   28.412
H   -1.865   5.981   30.201
H   -3.429   5.573   29.477
H   -3.595   4.275   27.621
H   -3.484   2.727   28.463
H   -1.449   2.250   26.850
H   -1.449   3.819   26.289
H   -4.199   2.885   25.910
H   -3.139   1.532   25.488
H   -2.689   4.320   24.567
H   -4.393   4.708   23.400
H   -5.098   3.175   23.887
H   -4.072   3.467   22.398
H   -1.324   1.863   23.775
H   -0.983   3.576   23.283
H   -1.884   2.641   22.255
