%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_651_chromophore_12 TDDFT with PBE1PBE functional

0 1
Mg   46.560   16.322   27.930
C   44.738   15.442   30.928
C   48.080   18.756   30.084
C   48.187   17.397   25.230
C   44.639   14.215   26.257
N   46.536   16.912   30.340
C   45.741   16.344   31.312
C   46.121   16.882   32.685
C   47.157   18.029   32.373
C   47.228   17.910   30.858
C   46.758   19.478   32.789
C   46.668   15.782   33.656
C   46.345   16.068   35.160
C   47.499   16.045   36.251
O   48.371   15.121   36.248
O   47.543   17.143   37.125
N   47.884   17.867   27.734
C   48.397   18.772   28.711
C   49.282   19.709   28.017
C   49.351   19.356   26.662
C   48.512   18.157   26.483
C   50.116   20.679   28.716
C   50.085   19.978   25.462
O   49.966   19.622   24.310
C   51.110   21.101   25.659
N   46.360   15.993   26.025
C   47.347   16.301   25.097
C   47.100   15.579   23.717
C   46.129   14.400   24.193
C   45.639   14.894   25.514
C   46.437   16.418   22.619
C   46.848   13.007   24.249
C   45.994   11.791   23.790
N   45.012   15.117   28.411
C   44.269   14.336   27.585
C   43.251   13.606   28.278
C   43.408   13.993   29.576
C   44.487   14.924   29.624
C   42.269   12.736   27.688
C   42.759   13.973   30.894
O   41.723   13.515   31.311
C   43.715   14.694   31.769
C   42.913   15.640   32.551
O   41.972   16.295   32.071
O   43.247   15.546   33.893
C   42.476   16.560   34.739
C   48.593   17.112   38.223
C   48.832   18.521   38.909
C   50.003   19.033   39.387
C   51.235   18.110   39.633
C   50.094   20.472   40.052
C   51.211   21.482   39.509
C   52.244   21.886   40.621
C   53.726   21.716   40.045
C   54.719   21.489   41.211
C   54.107   22.935   39.092
C   54.392   22.442   37.645
C   55.132   23.547   36.868
C   54.744   23.484   35.285
C   56.078   23.322   34.613
C   54.153   24.783   34.663
C   52.644   24.802   34.402
C   52.197   24.851   32.868
C   51.665   23.534   32.309
C   51.771   23.517   30.751
C   50.207   23.283   32.743
H   48.693   19.453   30.659
H   48.654   17.671   24.282
H   43.933   13.589   25.708
H   45.275   17.322   33.213
H   48.162   17.886   32.770
H   47.444   20.312   32.644
H   46.227   19.624   33.729
H   45.944   19.691   32.096
H   47.758   15.812   33.636
H   46.224   14.833   33.356
H   45.800   15.135   35.299
H   45.760   16.924   35.496
H   51.187   20.605   28.528
H   50.195   20.548   29.795
H   49.803   21.718   28.615
H   51.543   21.740   24.890
H   51.970   20.727   26.215
H   50.600   21.801   26.320
H   48.028   15.229   23.264
H   45.228   14.340   23.581
H   45.469   16.000   22.344
H   47.167   16.416   21.809
H   46.267   17.417   23.020
H   47.227   12.667   25.213
H   47.727   13.035   23.604
H   46.424   11.321   22.906
H   44.974   12.104   23.565
H   45.834   11.038   24.562
H   41.404   13.384   27.547
H   41.922   11.908   28.306
H   42.567   12.430   26.685
H   44.101   13.902   32.411
H   42.660   16.526   35.812
H   41.413   16.411   34.547
H   42.761   17.584   34.500
H   49.610   17.086   37.829
H   48.542   16.293   38.941
H   47.907   19.097   38.889
H   51.254   17.046   39.396
H   51.390   18.257   40.702
H   52.112   18.555   39.162
H   50.329   20.467   41.116
H   49.123   20.964   39.996
H   50.836   22.435   39.134
H   51.762   21.050   38.673
H   52.337   21.151   41.421
H   51.982   22.843   41.071
H   53.751   20.759   39.522
H   54.498   20.650   41.871
H   54.725   22.421   41.775
H   55.744   21.315   40.883
H   55.048   23.382   39.412
H   53.312   23.678   39.031
H   53.422   22.273   37.179
H   55.025   21.556   37.695
H   56.201   23.500   37.075
H   54.744   24.457   37.327
H   54.189   22.600   34.973
H   56.671   24.187   34.910
H   56.073   23.293   33.524
H   56.431   22.333   34.909
H   54.653   25.017   33.723
H   54.448   25.578   35.348
H   52.106   25.543   34.994
H   52.286   23.853   34.799
H   52.994   25.254   32.242
H   51.390   25.579   32.781
H   52.316   22.791   32.770
H   52.288   24.429   30.452
H   50.791   23.305   30.323
H   52.445   22.710   30.464
H   50.142   23.363   33.828
H   49.936   22.238   32.588
H   49.536   23.963   32.218

