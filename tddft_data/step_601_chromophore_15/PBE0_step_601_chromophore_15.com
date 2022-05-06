%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_601_chromophore_15 TDDFT with PBE1PBE functional

0 1
Mg   46.399   34.531   27.933
C   44.267   32.666   30.191
C   46.375   36.989   30.406
C   47.915   36.525   25.750
C   45.946   32.092   25.531
N   45.591   34.746   30.077
C   44.756   33.829   30.723
C   44.633   34.336   32.178
C   45.016   35.855   32.227
C   45.724   35.897   30.844
C   43.808   36.770   32.312
C   45.424   33.372   33.126
C   46.301   33.870   34.292
C   45.706   34.813   35.362
O   44.629   35.365   35.387
O   46.697   34.956   36.299
N   47.174   36.474   28.153
C   47.115   37.304   29.216
C   47.881   38.493   28.882
C   48.327   38.389   27.500
C   47.792   37.086   27.066
C   47.856   39.686   29.818
C   49.155   39.308   26.628
O   49.501   39.107   25.432
C   49.683   40.650   27.242
N   46.629   34.404   25.882
C   47.415   35.378   25.201
C   47.741   34.860   23.782
C   47.225   33.379   23.805
C   46.486   33.287   25.067
C   46.946   35.696   22.737
C   48.418   32.419   23.756
C   48.369   31.247   22.771
N   45.334   32.757   27.893
C   45.263   31.889   26.786
C   44.562   30.694   27.271
C   44.055   31.031   28.555
C   44.525   32.269   28.891
C   44.450   29.319   26.582
C   43.250   30.522   29.598
O   42.616   29.454   29.649
C   43.170   31.625   30.677
C   43.321   31.111   32.045
O   44.112   30.276   32.457
O   42.317   31.678   32.870
C   42.176   31.145   34.237
C   46.398   35.614   37.560
C   46.744   37.123   37.358
C   45.874   38.118   37.295
C   44.362   38.126   37.340
C   46.413   39.471   36.949
C   46.920   40.184   38.207
C   48.364   40.754   38.103
C   48.385   42.277   37.887
C   49.799   42.758   37.362
C   48.057   43.076   39.144
C   47.093   44.270   38.758
C   47.917   45.578   38.773
C   47.220   46.817   38.175
C   48.208   48.019   37.831
C   45.950   47.248   39.102
C   44.640   46.792   38.498
C   43.457   46.647   39.528
C   42.101   47.144   38.994
C   41.252   47.790   40.065
C   41.317   46.019   38.339
H   46.371   37.855   31.071
H   48.509   37.036   24.989
H   45.827   31.257   24.837
H   43.591   34.202   32.466
H   45.723   36.168   32.996
H   43.842   37.173   33.324
H   42.875   36.223   32.174
H   43.784   37.642   31.659
H   46.122   32.804   32.511
H   44.674   32.723   33.578
H   47.084   34.437   33.789
H   46.770   32.960   34.668
H   47.552   40.626   29.359
H   48.854   39.949   30.169
H   47.199   39.689   30.687
H   50.383   41.130   26.558
H   50.159   40.352   28.176
H   48.879   41.361   27.434
H   48.805   35.027   23.616
H   46.530   33.156   22.996
H   47.500   36.622   22.581
H   46.068   36.009   23.302
H   46.776   35.065   21.865
H   48.496   31.933   24.728
H   49.349   32.986   23.769
H   47.475   31.293   22.149
H   48.482   30.264   23.229
H   49.249   31.450   22.161
H   43.813   28.604   27.103
H   45.400   28.809   26.417
H   43.950   29.495   25.629
H   42.173   32.033   30.512
H   42.718   30.218   34.420
H   41.198   31.048   34.709
H   42.661   31.942   34.800
H   47.148   35.176   38.218
H   45.365   35.489   37.884
H   47.789   37.431   37.380
H   43.994   38.705   38.187
H   43.992   37.116   37.513
H   44.042   38.422   36.342
H   45.712   40.117   36.420
H   47.216   39.295   36.234
H   46.785   39.565   39.094
H   46.204   40.933   38.548
H   48.872   40.354   37.225
H   48.819   40.419   39.035
H   47.642   42.407   37.100
H   50.462   41.987   36.970
H   50.381   43.093   38.221
H   49.637   43.508   36.588
H   48.900   43.551   39.644
H   47.670   42.371   39.880
H   46.135   44.287   39.278
H   46.772   44.075   37.735
H   48.918   45.298   38.447
H   47.850   45.825   39.833
H   46.859   46.574   37.176
H   49.072   47.875   38.479
H   47.775   48.971   38.137
H   48.494   48.095   36.782
H   45.922   48.313   39.335
H   45.997   46.767   40.079
H   44.775   45.792   38.084
H   44.321   47.531   37.763
H   43.758   47.309   40.339
H   43.476   45.621   39.895
H   42.327   47.927   38.270
H   41.357   48.843   39.805
H   41.740   47.634   41.027
H   40.232   47.408   40.004
H   41.996   45.541   37.634
H   40.596   46.380   37.605
H   40.705   45.381   38.976

