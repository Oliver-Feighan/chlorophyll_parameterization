%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_301_chromophore_8 TDDFT with PBE1PBE functional

0 1
Mg   44.995   2.380   46.797
C   43.250   5.477   46.146
C   41.976   0.822   46.570
C   46.598   -0.461   46.749
C   47.900   4.191   46.408
N   42.930   3.068   46.339
C   42.404   4.375   46.191
C   40.881   4.442   46.047
C   40.536   2.963   46.609
C   41.879   2.237   46.542
C   39.780   2.921   48.025
C   40.385   4.585   44.536
C   39.423   5.747   44.360
C   38.554   5.699   43.147
O   37.411   5.328   43.066
O   39.237   6.225   42.035
N   44.357   0.417   46.723
C   43.091   -0.018   46.752
C   43.132   -1.483   46.834
C   44.472   -1.904   46.703
C   45.245   -0.642   46.653
C   41.929   -2.353   46.937
C   44.931   -3.374   46.699
O   44.163   -4.320   46.582
C   46.469   -3.689   46.720
N   46.971   1.993   46.636
C   47.376   0.740   46.717
C   48.947   0.618   46.707
C   49.395   2.105   46.470
C   47.993   2.848   46.472
C   49.516   -0.050   48.045
C   50.173   2.394   45.119
C   51.441   3.246   45.222
N   45.574   4.460   46.433
C   46.773   4.994   46.289
C   46.664   6.411   46.308
C   45.265   6.723   46.182
C   44.653   5.463   46.272
C   47.890   7.340   46.318
C   44.206   7.702   45.965
O   44.270   8.942   45.838
C   42.823   6.892   46.075
C   42.192   7.339   47.365
O   42.739   7.254   48.450
O   40.960   7.840   47.110
C   40.136   8.201   48.293
C   38.547   6.148   40.789
C   39.669   5.543   39.991
C   40.354   6.194   38.969
C   39.861   7.494   38.293
C   41.740   5.696   38.561
C   41.765   4.861   37.310
C   42.043   3.357   37.518
C   41.519   2.442   36.380
C   40.915   1.153   36.933
C   42.634   2.176   35.369
C   42.756   3.192   34.253
C   44.100   3.851   34.167
C   44.662   3.939   32.725
C   46.252   4.047   32.722
C   43.977   5.143   31.904
C   42.598   4.807   31.333
C   42.402   5.202   29.847
C   42.821   4.132   28.859
C   41.746   4.007   27.832
C   44.235   4.558   28.206
H   41.103   0.206   46.797
H   47.230   -1.350   46.811
H   48.832   4.757   46.478
H   40.345   5.132   46.699
H   39.853   2.467   45.919
H   40.088   3.694   48.728
H   40.067   1.976   48.486
H   38.730   2.981   47.739
H   39.819   3.692   44.274
H   41.192   4.746   43.821
H   40.121   6.565   44.179
H   38.753   5.853   45.213
H   41.251   -1.773   47.563
H   42.027   -3.372   47.312
H   41.418   -2.381   45.975
H   46.994   -3.300   45.848
H   46.701   -4.753   46.684
H   46.899   -3.310   47.648
H   49.283   0.125   45.794
H   49.932   2.510   47.328
H   50.120   0.599   48.679
H   50.109   -0.947   47.866
H   48.620   -0.229   48.639
H   49.522   2.890   44.400
H   50.461   1.482   44.597
H   52.262   2.637   44.843
H   51.586   3.511   46.270
H   51.359   4.175   44.659
H   48.510   7.215   45.430
H   48.532   6.998   47.130
H   47.580   8.382   46.396
H   42.195   7.120   45.214
H   40.710   8.450   49.185
H   39.362   7.505   48.615
H   39.808   9.184   47.956
H   38.204   7.158   40.565
H   37.642   5.540   40.752
H   40.160   4.662   40.404
H   39.780   7.354   37.215
H   40.484   8.341   38.581
H   38.868   7.616   38.726
H   42.095   5.072   39.381
H   42.360   6.591   38.618
H   42.616   5.383   36.871
H   40.990   4.942   36.547
H   41.594   3.029   38.456
H   43.088   3.139   37.741
H   40.756   3.017   35.856
H   41.146   0.888   37.965
H   41.125   0.290   36.301
H   39.845   1.349   36.987
H   42.458   1.205   34.905
H   43.566   2.033   35.916
H   42.073   4.036   34.336
H   42.474   2.738   33.303
H   44.745   3.416   34.931
H   44.004   4.813   34.672
H   44.424   3.011   32.206
H   46.711   4.777   33.388
H   46.742   4.083   31.748
H   46.627   3.128   33.172
H   44.744   5.464   31.199
H   43.794   5.984   32.573
H   41.802   5.307   31.885
H   42.431   3.740   31.480
H   42.960   6.115   29.638
H   41.363   5.499   29.702
H   43.072   3.186   29.338
H   40.926   3.486   28.325
H   42.119   3.467   26.962
H   41.430   5.010   27.544
H   44.939   4.179   28.947
H   44.419   5.632   28.171
H   44.379   4.010   27.275

