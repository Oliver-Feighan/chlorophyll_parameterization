%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_1001_chromophore_1 TDDFT with cam-b3lyp functional

0 1
Mg   -1.969   17.241   26.689
C   -2.270   15.084   29.304
C   -3.130   19.787   28.825
C   -1.840   19.356   24.102
C   -1.286   14.640   24.513
N   -2.812   17.404   28.904
C   -2.669   16.358   29.763
C   -2.934   16.928   31.159
C   -3.630   18.343   30.887
C   -3.236   18.532   29.439
C   -5.198   18.320   31.001
C   -1.622   16.999   32.093
C   -1.769   16.803   33.639
C   -0.411   16.695   34.415
O   0.644   16.403   33.931
O   -0.560   16.826   35.814
N   -2.145   19.304   26.540
C   -2.671   20.137   27.535
C   -2.602   21.524   26.990
C   -2.189   21.491   25.655
C   -2.039   20.013   25.346
C   -2.973   22.693   27.834
C   -1.943   22.496   24.640
O   -1.607   22.260   23.509
C   -1.951   23.863   24.939
N   -1.544   17.089   24.624
C   -1.651   18.084   23.749
C   -1.645   17.613   22.276
C   -1.147   16.122   22.439
C   -1.357   15.872   23.971
C   -2.948   17.840   21.456
C   0.316   15.909   21.985
C   1.343   16.552   22.862
N   -1.851   15.238   26.843
C   -1.561   14.281   25.908
C   -1.558   13.011   26.536
C   -1.761   13.225   27.871
C   -2.005   14.608   28.000
C   -1.349   11.737   25.822
C   -1.786   12.736   29.159
O   -1.525   11.611   29.520
C   -2.280   13.812   30.180
C   -1.278   13.732   31.275
O   -0.089   13.882   31.193
O   -1.938   13.407   32.466
C   -1.141   13.091   33.663
C   0.613   16.647   36.687
C   0.114   17.427   37.912
C   -0.122   16.953   39.161
C   0.128   15.508   39.650
C   -0.659   17.848   40.244
C   0.351   18.220   41.426
C   1.544   19.088   40.867
C   1.552   20.484   41.600
C   0.745   21.445   40.728
C   3.008   20.959   41.848
C   3.090   21.928   42.940
C   4.006   21.364   44.055
C   4.551   22.424   45.088
C   6.070   22.429   45.015
C   4.031   22.160   46.504
C   2.642   22.732   46.636
C   1.555   21.599   46.900
C   0.158   21.751   46.287
C   -0.921   22.119   47.290
C   -0.239   20.579   45.369
H   -3.515   20.644   29.381
H   -1.859   20.073   23.279
H   -1.013   13.787   23.888
H   -3.558   16.162   31.620
H   -3.144   19.194   31.365
H   -5.657   19.216   30.581
H   -5.368   18.259   32.076
H   -5.500   17.441   30.432
H   -1.196   17.984   31.899
H   -0.898   16.314   31.651
H   -2.359   15.907   33.829
H   -2.277   17.619   34.152
H   -2.369   23.561   27.572
H   -2.891   22.465   28.896
H   -4.023   22.905   27.633
H   -2.952   24.293   24.902
H   -1.388   24.400   24.176
H   -1.486   24.115   25.892
H   -0.897   18.178   21.720
H   -1.842   15.457   21.925
H   -2.887   18.074   20.394
H   -3.514   18.670   21.881
H   -3.552   16.953   21.647
H   0.483   16.172   20.940
H   0.495   14.838   22.075
H   1.376   16.206   23.895
H   1.574   17.593   22.635
H   2.296   16.135   22.536
H   -0.566   11.783   25.066
H   -2.302   11.349   25.463
H   -0.992   11.015   26.556
H   -3.208   13.494   30.655
H   -1.471   12.238   34.257
H   -1.224   13.906   34.381
H   -0.079   12.928   33.483
H   1.516   17.057   36.235
H   0.637   15.566   36.821
H   -0.207   18.461   37.786
H   0.706   15.483   40.574
H   0.621   14.887   38.903
H   -0.830   15.070   39.930
H   -1.547   17.330   40.608
H   -1.060   18.775   39.835
H   0.745   17.328   41.914
H   -0.188   18.775   42.194
H   1.347   19.151   39.797
H   2.524   18.623   40.982
H   0.972   20.474   42.523
H   1.362   22.341   40.674
H   -0.268   21.632   41.085
H   0.617   21.066   39.714
H   3.461   21.459   40.992
H   3.676   20.098   41.867
H   2.101   22.115   43.361
H   3.456   22.871   42.533
H   4.919   20.853   43.750
H   3.399   20.591   44.526
H   4.146   23.390   44.787
H   6.571   21.793   44.285
H   6.438   21.995   45.945
H   6.418   23.451   44.865
H   4.683   22.652   47.225
H   4.093   21.097   46.738
H   2.403   23.374   45.789
H   2.657   23.455   47.451
H   1.474   21.455   47.978
H   1.961   20.680   46.478
H   0.285   22.651   45.685
H   -1.376   23.076   47.036
H   -0.463   22.352   48.251
H   -1.711   21.373   47.384
H   -0.519   19.669   45.899
H   0.491   20.391   44.582
H   -1.170   20.999   44.988

