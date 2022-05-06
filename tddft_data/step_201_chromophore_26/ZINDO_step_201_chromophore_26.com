%nproc=24
%mem=50gb
#p ZINDO=(nstates=5)

step_201_chromophore_26 ZINDO

0 1
Mg   -9.145   18.428   42.791
C   -5.593   18.186   42.942
C   -8.773   21.735   42.300
C   -12.468   18.492   42.055
C   -9.261   14.888   42.720
N   -7.288   19.778   42.551
C   -5.952   19.465   42.656
C   -5.056   20.724   42.672
C   -6.088   21.821   42.649
C   -7.455   21.088   42.396
C   -5.955   22.750   43.909
C   -4.082   20.821   41.474
C   -4.676   20.136   40.264
C   -4.033   20.137   38.828
O   -2.911   19.653   38.564
O   -4.796   20.807   37.955
N   -10.462   20.012   42.338
C   -10.147   21.274   42.162
C   -11.335   21.985   41.833
C   -12.439   21.086   41.743
C   -11.824   19.819   42.131
C   -11.269   23.453   41.529
C   -13.826   21.346   41.352
O   -14.166   22.464   41.013
C   -14.925   20.337   41.385
N   -10.592   16.912   42.172
C   -11.911   17.215   42.116
C   -12.830   16.020   42.269
C   -11.719   14.842   42.315
C   -10.409   15.587   42.324
C   -13.867   16.063   43.423
C   -11.604   13.706   41.200
C   -12.848   12.716   41.106
N   -7.736   16.834   42.895
C   -7.959   15.474   42.921
C   -6.669   14.781   43.178
C   -5.730   15.808   43.275
C   -6.466   17.023   43.079
C   -6.411   13.283   43.039
C   -4.310   16.100   43.417
O   -3.344   15.404   43.677
C   -4.161   17.695   43.306
C   -3.852   18.216   44.688
O   -4.521   18.159   45.700
O   -2.511   18.650   44.777
C   -2.085   18.858   46.173
C   -4.201   21.036   36.639
C   -5.099   20.368   35.580
C   -5.170   19.117   35.111
C   -4.413   17.865   35.638
C   -6.110   18.780   34.055
C   -7.619   18.992   34.407
C   -8.582   18.073   33.585
C   -9.940   18.681   33.152
C   -11.017   17.614   32.875
C   -9.863   19.751   32.045
C   -9.815   21.213   32.620
C   -8.717   22.068   32.059
C   -9.273   23.303   31.211
C   -8.824   24.669   31.823
C   -8.830   23.226   29.684
C   -9.912   22.887   28.668
C   -9.756   21.489   28.093
C   -10.722   21.402   26.886
C   -11.325   19.977   26.705
C   -10.113   21.872   25.630
H   -8.683   22.809   42.124
H   -13.536   18.337   41.889
H   -9.375   13.804   42.663
H   -4.461   20.679   43.584
H   -5.929   22.590   41.893
H   -5.422   23.643   43.584
H   -5.581   22.111   44.709
H   -6.973   22.974   44.227
H   -3.137   20.299   41.624
H   -3.889   21.860   41.204
H   -5.734   20.389   40.197
H   -4.618   19.065   40.456
H   -11.909   24.038   42.190
H   -11.588   23.448   40.487
H   -10.274   23.892   41.456
H   -15.097   19.938   42.385
H   -14.747   19.534   40.670
H   -15.829   20.899   41.147
H   -13.373   16.022   41.324
H   -11.842   14.340   43.275
H   -14.347   15.085   43.403
H   -14.612   16.846   43.286
H   -13.283   16.337   44.302
H   -10.774   13.016   41.353
H   -11.499   14.148   40.209
H   -12.754   11.887   41.808
H   -12.928   12.350   40.082
H   -13.749   13.297   41.300
H   -5.728   12.922   43.808
H   -6.075   13.052   42.029
H   -7.402   12.829   43.048
H   -3.315   17.974   42.677
H   -2.595   19.684   46.668
H   -1.021   19.086   46.240
H   -2.135   17.922   46.730
H   -3.166   20.751   36.452
H   -4.210   22.124   36.573
H   -5.831   21.019   35.103
H   -3.763   17.445   34.870
H   -5.104   17.110   36.014
H   -3.659   18.039   36.406
H   -5.871   17.809   33.621
H   -5.915   19.497   33.258
H   -7.837   20.049   34.257
H   -7.760   18.751   35.460
H   -8.701   17.157   34.163
H   -7.947   17.708   32.778
H   -10.280   19.270   34.004
H   -10.963   16.770   33.563
H   -11.099   17.249   31.851
H   -11.964   18.072   33.160
H   -10.747   19.713   31.408
H   -8.914   19.582   31.536
H   -9.607   21.206   33.690
H   -10.798   21.679   32.538
H   -8.059   21.475   31.424
H   -8.116   22.368   32.918
H   -10.363   23.276   31.217
H   -8.475   24.487   32.840
H   -9.682   25.325   31.974
H   -7.918   25.071   31.370
H   -8.016   22.526   29.496
H   -8.397   24.175   29.368
H   -9.872   23.610   27.853
H   -10.920   22.943   29.080
H   -10.108   20.858   28.910
H   -8.705   21.461   27.808
H   -11.595   22.039   27.030
H   -10.972   19.538   25.771
H   -12.409   20.050   26.618
H   -10.977   19.328   27.509
H   -10.722   22.600   25.094
H   -9.729   21.130   24.930
H   -9.196   22.422   25.841

