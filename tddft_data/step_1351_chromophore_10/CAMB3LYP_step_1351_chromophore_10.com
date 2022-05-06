%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1351_chromophore_10 TDDFT with cam-b3lyp functional

0 1
Mg   40.751   8.100   29.776
C   42.317   9.473   32.576
C   38.480   6.631   31.840
C   39.431   6.524   27.048
C   43.236   9.496   27.805
N   40.558   7.927   31.957
C   41.168   8.713   32.929
C   40.589   8.513   34.335
C   39.459   7.405   34.020
C   39.520   7.263   32.526
C   39.674   6.006   34.682
C   39.998   9.862   34.912
C   40.627   10.362   36.262
C   39.696   10.831   37.394
O   38.971   11.821   37.338
O   40.064   10.247   38.568
N   39.126   6.804   29.463
C   38.342   6.347   30.410
C   37.259   5.566   29.771
C   37.446   5.629   28.363
C   38.734   6.262   28.243
C   36.137   4.826   30.606
C   36.565   5.125   27.186
O   36.880   5.198   25.983
C   35.321   4.385   27.504
N   41.229   8.124   27.675
C   40.626   7.234   26.790
C   41.128   7.544   25.354
C   42.138   8.716   25.592
C   42.200   8.817   27.104
C   41.720   6.288   24.737
C   41.791   10.036   24.897
C   40.696   10.900   25.487
N   42.525   9.158   30.063
C   43.432   9.598   29.187
C   44.507   10.293   29.920
C   44.013   10.390   31.269
C   42.901   9.590   31.301
C   45.740   10.781   29.337
C   44.306   10.842   32.568
O   45.179   11.660   32.788
C   43.190   10.256   33.513
C   43.781   9.234   34.427
O   44.140   8.145   34.054
O   43.911   9.715   35.720
C   44.209   8.813   36.834
C   39.780   10.886   39.890
C   38.624   10.193   40.609
C   38.380   10.297   41.983
C   39.070   11.260   42.894
C   37.289   9.505   42.593
C   36.037   10.407   42.964
C   35.532   10.199   44.410
C   34.927   11.450   45.170
C   35.598   11.576   46.598
C   33.373   11.516   45.394
C   32.775   12.856   44.888
C   32.106   12.667   43.418
C   30.883   13.582   43.187
C   29.583   12.704   42.857
C   31.156   14.522   41.977
C   30.197   15.698   42.054
C   30.885   17.068   41.911
C   30.721   17.945   43.199
C   29.957   19.317   43.043
C   32.176   18.228   43.916
H   37.656   6.419   32.524
H   39.073   5.868   26.251
H   43.965   9.954   27.133
H   41.304   7.974   34.955
H   38.430   7.706   34.218
H   39.574   5.244   33.909
H   38.819   5.980   35.358
H   40.622   5.864   35.200
H   38.915   9.824   35.036
H   40.166   10.690   34.223
H   41.343   11.172   36.123
H   41.311   9.631   36.692
H   36.831   4.401   31.332
H   35.503   4.025   30.226
H   35.545   5.592   31.108
H   35.537   3.437   27.997
H   34.908   4.014   26.565
H   34.551   4.991   27.981
H   40.231   7.845   24.812
H   43.090   8.251   25.337
H   42.265   5.742   25.507
H   42.550   6.504   24.065
H   40.846   5.746   24.377
H   41.488   9.767   23.885
H   42.701   10.637   24.885
H   40.237   10.442   26.364
H   39.897   10.974   24.749
H   40.972   11.908   25.795
H   46.554   10.711   30.058
H   45.540   11.802   29.010
H   46.086   10.180   28.496
H   42.651   11.058   34.016
H   44.974   9.241   37.482
H   44.607   7.880   36.434
H   43.472   8.553   37.593
H   39.463   11.926   39.821
H   40.660   10.883   40.534
H   38.000   9.472   40.081
H   39.976   11.789   42.597
H   39.429   10.618   43.698
H   38.304   11.934   43.278
H   37.714   8.910   43.401
H   36.870   8.809   41.865
H   35.232   10.124   42.286
H   36.179   11.436   42.633
H   36.260   9.701   45.050
H   34.730   9.468   44.307
H   35.222   12.332   44.601
H   35.210   10.755   47.202
H   35.374   12.512   47.110
H   36.674   11.416   46.533
H   33.127   11.219   46.414
H   32.928   10.676   44.862
H   33.448   13.714   44.903
H   32.002   13.013   45.639
H   31.883   11.620   43.214
H   32.846   12.898   42.652
H   30.809   14.149   44.115
H   28.847   12.812   43.654
H   29.850   11.649   42.796
H   29.208   12.973   41.870
H   31.004   13.995   41.035
H   32.191   14.864   41.974
H   29.614   15.708   42.974
H   29.466   15.695   41.246
H   30.446   17.653   41.103
H   31.891   16.930   41.514
H   30.005   17.509   43.896
H   29.689   19.657   42.043
H   30.493   20.092   43.590
H   29.042   19.204   43.624
H   33.002   17.654   43.497
H   32.031   17.979   44.967
H   32.554   19.249   43.876

