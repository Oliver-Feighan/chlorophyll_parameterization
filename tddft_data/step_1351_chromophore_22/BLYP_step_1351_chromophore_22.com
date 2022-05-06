%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1351_chromophore_22 TDDFT with blyp functional

0 1
Mg   8.805   47.945   24.884
C   6.641   47.916   27.515
C   11.293   49.297   26.836
C   10.810   47.591   22.353
C   5.959   47.534   22.724
N   8.891   48.616   26.942
C   7.829   48.490   27.868
C   8.320   49.017   29.201
C   9.785   49.643   28.849
C   10.000   49.193   27.427
C   9.846   51.159   29.042
C   8.228   47.940   30.354
C   9.185   48.108   31.529
C   8.925   47.022   32.677
O   9.137   45.800   32.483
O   8.449   47.617   33.807
N   10.774   48.429   24.620
C   11.681   48.897   25.587
C   13.051   48.844   25.012
C   12.913   48.486   23.613
C   11.487   48.181   23.436
C   14.267   49.304   25.926
C   14.037   48.268   22.559
O   13.698   47.989   21.408
C   15.461   48.373   22.947
N   8.449   47.968   22.674
C   9.484   47.534   21.979
C   9.038   46.900   20.612
C   7.473   47.046   20.642
C   7.283   47.489   22.109
C   9.672   47.585   19.350
C   6.691   45.786   20.094
C   5.786   45.994   18.881
N   6.713   47.735   25.032
C   5.668   47.568   24.113
C   4.385   47.383   24.832
C   4.710   47.541   26.183
C   6.160   47.689   26.265
C   3.027   47.220   24.085
C   4.187   47.592   27.569
O   3.020   47.401   27.963
C   5.380   47.748   28.509
C   5.401   46.525   29.330
O   6.214   45.604   29.277
O   4.443   46.688   30.287
C   4.405   45.786   31.472
C   8.227   46.865   34.964
C   8.852   47.441   36.191
C   8.662   47.204   37.539
C   7.637   46.278   38.070
C   9.488   47.926   38.606
C   10.738   47.116   39.041
C   11.353   47.766   40.267
C   11.725   46.720   41.392
C   11.693   47.404   42.762
C   12.992   45.879   41.049
C   12.882   44.326   41.067
C   12.940   43.872   42.541
C   14.232   43.102   42.848
C   14.732   43.378   44.296
C   14.149   41.632   42.630
C   15.488   41.056   42.068
C   15.353   40.627   40.601
C   16.279   41.307   39.574
C   15.431   41.666   38.309
C   17.472   40.383   39.147
H   12.028   49.616   27.578
H   11.468   47.259   21.547
H   5.188   47.127   22.066
H   7.590   49.815   29.335
H   10.588   49.177   29.421
H   10.477   51.479   29.871
H   8.823   51.471   29.257
H   10.246   51.694   28.181
H   8.322   46.994   29.819
H   7.271   48.036   30.866
H   8.989   49.086   31.968
H   10.210   48.032   31.167
H   13.989   49.320   26.980
H   14.588   50.272   25.541
H   15.011   48.517   25.806
H   16.047   48.573   22.050
H   15.700   47.428   23.435
H   15.588   49.228   23.612
H   9.325   45.848   20.627
H   7.283   47.929   20.031
H   9.013   48.072   18.630
H   10.239   46.824   18.814
H   10.353   48.371   19.674
H   6.054   45.449   20.911
H   7.373   44.940   20.000
H   4.803   45.631   19.180
H   6.178   45.419   18.042
H   5.728   47.041   18.582
H   2.224   46.937   24.765
H   3.124   46.376   23.402
H   2.775   48.061   23.438
H   5.133   48.612   29.126
H   4.519   46.275   32.440
H   5.075   44.926   31.471
H   3.434   45.292   31.470
H   8.360   45.786   35.046
H   7.148   46.954   35.093
H   9.572   48.216   35.927
H   6.735   46.457   37.486
H   7.376   46.572   39.087
H   8.016   45.259   37.995
H   8.857   48.159   39.464
H   9.943   48.805   38.149
H   11.475   47.094   38.239
H   10.429   46.079   39.174
H   10.651   48.516   40.632
H   12.292   48.205   39.930
H   10.899   46.010   41.434
H   11.862   48.472   42.625
H   12.442   47.064   43.477
H   10.742   47.287   43.283
H   13.847   46.126   41.677
H   13.265   46.190   40.041
H   13.697   43.967   40.438
H   11.954   43.932   40.652
H   12.096   43.232   42.797
H   12.854   44.714   43.229
H   15.090   43.466   42.283
H   15.472   44.177   44.323
H   15.143   42.479   44.754
H   14.016   43.829   44.983
H   13.310   41.525   41.942
H   13.930   41.073   43.539
H   15.750   40.154   42.622
H   16.387   41.667   42.148
H   14.355   40.842   40.221
H   15.558   39.559   40.517
H   16.661   42.202   40.064
H   14.617   42.320   38.619
H   14.924   40.856   37.785
H   16.138   42.212   37.683
H   17.466   39.334   39.443
H   18.362   40.856   39.562
H   17.711   40.349   38.084
