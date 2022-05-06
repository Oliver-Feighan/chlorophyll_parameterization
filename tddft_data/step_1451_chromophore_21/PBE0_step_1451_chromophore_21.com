%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_1451_chromophore_21 TDDFT with PBE1PBE functional

0 1
Mg   15.932   52.349   25.464
C   17.624   50.969   28.268
C   13.347   53.161   27.558
C   14.102   53.436   22.765
C   18.283   50.994   23.435
N   15.566   51.918   27.736
C   16.435   51.576   28.674
C   15.907   51.774   30.052
C   14.529   52.491   29.844
C   14.434   52.521   28.308
C   13.269   51.929   30.613
C   16.964   52.511   30.928
C   17.228   51.887   32.354
C   16.078   51.043   32.968
O   16.260   49.917   33.419
O   14.921   51.769   33.278
N   14.046   53.276   25.200
C   13.189   53.496   26.187
C   11.960   54.226   25.625
C   12.204   54.416   24.237
C   13.493   53.696   23.996
C   10.772   54.512   26.510
C   11.338   55.075   23.153
O   11.620   54.980   21.978
C   10.184   55.917   23.558
N   16.104   52.209   23.325
C   15.246   52.681   22.418
C   15.838   52.608   21.007
C   17.188   51.857   21.228
C   17.202   51.624   22.724
C   14.904   52.108   19.854
C   18.478   52.528   20.653
C   19.664   51.718   20.062
N   17.663   51.269   25.724
C   18.453   50.706   24.806
C   19.596   50.090   25.432
C   19.276   50.097   26.818
C   18.102   50.809   26.911
C   20.630   49.277   24.847
C   19.664   49.482   28.083
O   20.524   48.596   28.311
C   18.625   50.127   29.130
C   18.031   48.971   29.952
O   17.131   48.221   29.571
O   18.595   48.907   31.206
C   18.096   47.802   32.096
C   13.972   51.190   34.197
C   12.700   52.077   34.182
C   11.372   51.869   34.157
C   10.850   50.420   33.904
C   10.339   52.971   34.190
C   9.909   53.529   35.565
C   10.074   54.984   35.700
C   8.805   55.818   35.964
C   9.058   57.320   35.653
C   8.266   55.729   37.404
C   6.712   55.812   37.665
C   6.299   57.201   38.350
C   4.943   57.741   37.744
C   5.059   59.243   37.626
C   3.699   57.422   38.704
C   2.322   57.283   37.817
C   1.207   58.219   38.179
C   0.532   57.918   39.609
C   -0.808   57.296   39.247
C   0.368   59.287   40.336
H   12.534   53.292   28.276
H   13.451   53.767   21.953
H   19.054   50.750   22.701
H   15.767   50.750   30.398
H   14.599   53.544   30.117
H   12.938   52.718   31.288
H   13.416   51.002   31.167
H   12.443   51.688   29.944
H   16.589   53.526   31.061
H   17.948   52.599   30.467
H   17.450   52.751   32.980
H   18.213   51.423   32.401
H   10.930   55.397   27.127
H   10.645   53.694   27.219
H   9.833   54.677   25.982
H   10.411   56.228   24.578
H   9.293   55.290   23.551
H   9.946   56.732   22.874
H   16.099   53.631   20.740
H   17.053   50.861   20.805
H   15.302   51.258   19.300
H   14.573   52.884   19.165
H   14.077   51.621   20.370
H   18.886   53.037   21.526
H   18.164   53.236   19.886
H   20.527   52.158   20.562
H   19.714   51.989   19.007
H   19.601   50.631   20.108
H   20.799   49.750   23.880
H   20.373   48.228   24.698
H   21.504   49.286   25.498
H   19.241   50.807   29.718
H   17.022   47.637   32.008
H   18.245   48.134   33.123
H   18.591   46.875   31.808
H   14.310   51.320   35.225
H   13.693   50.158   33.988
H   12.934   53.140   34.222
H   10.580   49.925   34.837
H   11.706   49.864   33.520
H   10.067   50.273   33.161
H   9.495   52.567   33.632
H   10.763   53.787   33.604
H   10.558   53.074   36.313
H   8.891   53.185   35.749
H   10.550   55.432   34.829
H   10.819   55.216   36.460
H   7.972   55.487   35.344
H   8.350   57.969   36.169
H   8.914   57.460   34.582
H   10.025   57.641   36.039
H   8.724   56.544   37.964
H   8.618   54.832   37.913
H   6.254   54.936   38.125
H   6.188   55.945   36.718
H   7.102   57.937   38.322
H   6.103   57.054   39.412
H   4.755   57.472   36.705
H   4.846   59.793   38.542
H   4.336   59.618   36.901
H   6.073   59.363   37.246
H   3.459   58.085   39.536
H   3.788   56.424   39.132
H   1.926   56.274   37.927
H   2.575   57.436   36.767
H   0.486   58.232   37.361
H   1.604   59.232   38.107
H   1.133   57.230   40.203
H   -1.046   57.026   38.218
H   -1.555   58.033   39.544
H   -0.866   56.323   39.735
H   -0.635   59.640   40.576
H   0.730   60.176   39.818
H   0.936   59.146   41.255

