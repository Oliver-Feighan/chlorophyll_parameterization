%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1151_chromophore_3 TDDFT with cam-b3lyp functional

0 1
Mg   1.995   7.595   26.669
C   2.470   9.693   29.417
C   2.452   4.937   28.686
C   1.805   5.612   23.871
C   1.856   10.319   24.583
N   2.319   7.394   28.765
C   2.412   8.329   29.736
C   2.261   7.759   31.079
C   2.554   6.224   30.859
C   2.513   6.181   29.330
C   3.852   5.629   31.563
C   0.873   7.990   31.768
C   0.879   8.308   33.324
C   1.879   7.538   34.268
O   3.067   7.839   34.541
O   1.306   6.456   34.835
N   2.107   5.589   26.344
C   2.273   4.629   27.307
C   2.176   3.323   26.760
C   2.163   3.419   25.344
C   2.053   4.919   25.113
C   2.214   2.026   27.511
C   2.200   2.310   24.274
O   2.397   2.545   23.075
C   1.921   0.947   24.648
N   1.878   7.937   24.457
C   1.675   6.962   23.580
C   1.270   7.504   22.235
C   1.332   9.039   22.429
C   1.637   9.154   23.948
C   2.292   6.968   21.228
C   0.123   9.815   21.962
C   0.195   10.894   20.785
N   2.238   9.603   26.874
C   2.142   10.580   25.943
C   2.295   11.933   26.604
C   2.413   11.622   27.988
C   2.368   10.176   28.093
C   2.137   13.263   25.956
C   2.662   12.126   29.320
O   3.010   13.236   29.768
C   2.547   10.954   30.261
C   3.728   11.005   31.228
O   4.872   10.555   31.132
O   3.232   11.647   32.344
C   4.195   11.615   33.464
C   2.057   5.880   35.913
C   1.262   4.762   36.607
C   1.562   4.292   37.804
C   2.816   4.635   38.520
C   0.580   3.337   38.496
C   0.572   1.810   38.220
C   0.764   0.906   39.460
C   2.050   0.019   39.515
C   3.141   0.791   40.269
C   1.817   -1.455   39.935
C   2.916   -2.410   39.269
C   3.582   -3.364   40.331
C   5.137   -3.374   40.213
C   5.655   -4.852   40.319
C   5.794   -2.445   41.305
C   6.402   -1.132   40.813
C   7.906   -1.172   40.521
C   8.701   -0.246   41.461
C   9.792   0.641   40.780
C   9.372   -1.134   42.630
H   2.537   4.019   29.273
H   1.542   4.823   23.164
H   1.899   11.212   23.956
H   3.041   8.274   31.640
H   1.775   5.524   31.162
H   3.654   4.694   32.086
H   4.112   6.292   32.389
H   4.627   5.338   30.855
H   0.178   7.152   31.706
H   0.376   8.827   31.276
H   -0.069   8.012   33.775
H   1.009   9.377   33.491
H   3.152   1.586   27.172
H   1.352   1.394   27.301
H   2.302   2.250   28.574
H   1.941   0.280   23.786
H   0.970   0.916   25.180
H   2.776   0.688   25.272
H   0.305   7.051   22.006
H   2.265   9.438   22.030
H   3.170   6.570   21.735
H   2.553   7.620   20.395
H   1.919   6.075   20.727
H   -0.381   10.247   22.827
H   -0.670   9.117   21.695
H   0.449   10.468   19.814
H   0.921   11.681   20.992
H   -0.788   11.366   20.793
H   1.716   13.142   24.958
H   3.131   13.711   25.952
H   1.472   13.950   26.479
H   1.599   11.041   30.790
H   5.135   12.017   33.085
H   4.259   10.624   33.914
H   3.874   12.310   34.240
H   2.130   6.741   36.578
H   3.052   5.512   35.661
H   0.488   4.358   35.954
H   3.350   5.493   38.111
H   3.433   3.798   38.845
H   2.466   5.087   39.448
H   -0.423   3.714   38.697
H   0.945   3.405   39.521
H   1.366   1.584   37.508
H   -0.280   1.491   37.621
H   -0.120   0.272   39.525
H   0.744   1.532   40.352
H   2.475   0.058   38.512
H   2.686   1.490   40.970
H   3.761   1.185   39.463
H   3.769   0.080   40.805
H   0.845   -1.786   39.568
H   1.877   -1.546   41.019
H   3.541   -1.781   38.636
H   2.399   -3.117   38.620
H   3.130   -4.346   40.471
H   3.391   -2.943   41.318
H   5.364   -3.075   39.190
H   4.808   -5.483   40.588
H   6.335   -5.030   41.152
H   6.142   -5.216   39.414
H   6.509   -3.122   41.771
H   5.112   -2.089   42.077
H   6.041   -0.308   41.429
H   5.973   -0.857   39.850
H   8.032   -0.905   39.472
H   8.301   -2.188   40.503
H   8.013   0.492   41.873
H   9.533   1.681   40.979
H   9.826   0.495   39.700
H   10.763   0.379   41.201
H   8.629   -1.837   43.008
H   9.724   -0.535   43.470
H   10.188   -1.680   42.158

