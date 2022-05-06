%nproc=24
%mem=75GB
#p blyp/Def2SVP td=(nstates=5)

step_1851_chromophore_18 TDDFT with blyp functional

0 1
Mg   35.794   49.274   25.041
C   35.443   47.642   28.247
C   34.174   52.039   26.352
C   35.605   50.598   22.058
C   36.481   46.081   23.689
N   34.815   49.652   27.126
C   34.928   48.965   28.298
C   34.421   49.775   29.468
C   34.050   51.227   28.808
C   34.346   50.961   27.297
C   32.608   51.724   29.173
C   35.367   49.802   30.722
C   34.925   49.107   32.020
C   33.418   49.246   32.332
O   32.582   48.345   32.109
O   33.137   50.316   33.125
N   35.103   51.109   24.358
C   34.454   52.157   25.008
C   34.235   53.325   24.204
C   34.783   52.953   22.971
C   35.204   51.541   23.080
C   33.444   54.510   24.623
C   34.897   53.745   21.769
O   35.380   53.273   20.729
C   34.339   55.180   21.650
N   35.953   48.467   23.115
C   35.976   49.223   22.027
C   36.467   48.499   20.783
C   36.754   47.106   21.396
C   36.348   47.171   22.826
C   35.581   48.496   19.463
C   38.269   46.668   21.283
C   39.322   47.554   21.899
N   35.934   47.293   25.703
C   36.332   46.142   25.056
C   36.471   45.013   25.983
C   36.098   45.548   27.269
C   35.825   46.938   27.049
C   36.697   43.562   25.624
C   35.979   45.209   28.665
O   36.078   44.158   29.279
C   35.450   46.530   29.322
C   34.154   46.213   29.866
O   33.170   46.036   29.178
O   34.256   45.940   31.237
C   33.143   45.266   31.887
C   31.708   50.535   33.559
C   31.534   52.006   33.860
C   30.893   52.610   34.855
C   30.068   51.903   35.784
C   31.042   54.110   35.259
C   30.282   55.091   34.379
C   30.900   56.460   34.026
C   30.970   57.497   35.170
C   32.355   57.801   35.617
C   30.178   58.875   34.916
C   28.777   58.865   35.608
C   28.832   59.763   36.863
C   28.764   61.287   36.488
C   29.817   62.082   37.313
C   27.318   61.792   36.816
C   27.041   63.048   35.878
C   25.499   63.231   35.779
C   25.063   64.767   35.715
C   24.519   65.223   37.072
C   24.005   65.044   34.538
H   33.976   52.975   26.878
H   35.798   51.060   21.088
H   36.749   45.115   23.255
H   33.518   49.252   29.782
H   34.657   52.015   29.255
H   32.018   51.022   29.763
H   32.151   52.069   28.246
H   32.621   52.691   29.675
H   35.578   50.848   30.944
H   36.349   49.451   30.404
H   35.483   49.677   32.764
H   35.229   48.061   32.052
H   34.087   55.384   24.725
H   32.959   54.311   25.578
H   32.698   54.682   23.847
H   33.250   55.157   21.610
H   34.832   55.643   20.795
H   34.708   55.716   22.524
H   37.367   48.912   20.327
H   36.100   46.322   21.014
H   34.664   49.084   19.456
H   35.363   47.511   19.050
H   36.227   48.816   18.645
H   38.515   46.560   20.227
H   38.308   45.617   21.572
H   39.938   48.117   21.198
H   39.958   46.973   22.568
H   38.813   48.147   22.659
H   37.164   43.151   26.519
H   37.438   43.422   24.838
H   35.777   43.026   25.389
H   36.118   46.725   30.161
H   32.648   45.981   32.545
H   33.571   44.467   32.492
H   32.480   44.830   31.140
H   31.734   49.925   34.462
H   30.834   50.220   32.988
H   32.217   52.651   33.309
H   29.361   52.529   36.327
H   30.803   51.584   36.524
H   29.452   51.081   35.419
H   32.099   54.373   35.315
H   30.617   54.311   36.242
H   29.263   55.204   34.748
H   30.084   54.656   33.399
H   30.329   56.974   33.253
H   31.863   56.304   33.540
H   30.526   57.036   36.053
H   32.745   58.596   34.981
H   33.056   56.967   35.562
H   32.154   58.129   36.637
H   30.147   59.188   33.872
H   30.755   59.721   35.288
H   28.561   57.850   35.941
H   28.016   59.191   34.899
H   29.743   59.618   37.443
H   28.050   59.512   37.580
H   28.795   61.544   35.429
H   29.466   63.019   37.745
H   30.585   62.483   36.652
H   30.162   61.573   38.213
H   27.121   62.105   37.841
H   26.669   60.965   36.526
H   27.355   62.821   34.860
H   27.546   63.966   36.181
H   24.909   62.608   36.451
H   25.266   62.976   34.745
H   25.973   65.354   35.595
H   23.488   64.870   37.072
H   24.310   66.293   37.050
H   25.167   64.897   37.885
H   23.150   64.370   34.491
H   24.500   64.936   33.573
H   23.612   66.051   34.675

