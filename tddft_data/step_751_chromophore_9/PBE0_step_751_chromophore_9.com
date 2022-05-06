%nproc=24
%mem=75GB
#p PBE1PBE/Def2SVP td=(nstates=5)

step_751_chromophore_9 TDDFT with PBE1PBE functional

0 1
Mg   35.367   1.414   29.888
C   32.885   2.352   32.147
C   37.725   1.797   32.425
C   37.855   0.820   27.539
C   33.048   1.656   27.359
N   35.317   2.075   32.090
C   34.207   2.429   32.741
C   34.460   2.570   34.263
C   36.043   2.743   34.296
C   36.409   2.159   32.868
C   36.530   4.208   34.401
C   33.860   1.437   35.074
C   33.594   1.609   36.530
C   34.315   2.740   37.296
O   33.682   3.714   37.693
O   35.605   2.572   37.518
N   37.489   1.265   29.966
C   38.238   1.345   31.173
C   39.518   0.794   30.842
C   39.661   0.569   29.531
C   38.318   0.880   28.929
C   40.574   0.583   31.982
C   40.778   0.068   28.648
O   40.658   -0.002   27.425
C   42.132   -0.235   29.231
N   35.525   1.263   27.758
C   36.622   1.138   26.984
C   36.251   0.991   25.471
C   34.751   1.287   25.497
C   34.443   1.389   26.927
C   37.050   2.002   24.536
C   33.925   0.133   24.850
C   33.875   -1.209   25.640
N   33.419   1.970   29.724
C   32.535   1.931   28.622
C   31.194   2.331   29.029
C   31.282   2.403   30.427
C   32.648   2.234   30.787
C   29.938   2.590   28.322
C   30.527   2.621   31.639
O   29.342   2.867   31.850
C   31.486   2.522   32.831
C   31.544   3.790   33.668
O   32.161   4.799   33.327
O   30.925   3.680   34.884
C   30.990   4.817   35.798
C   36.359   3.537   38.313
C   37.423   2.939   39.155
C   37.302   2.521   40.472
C   35.938   2.306   41.178
C   38.569   2.318   41.305
C   39.093   0.832   41.012
C   39.691   0.129   42.295
C   41.149   -0.506   42.134
C   40.981   -2.068   41.956
C   41.991   -0.135   43.370
C   42.419   1.365   43.110
C   43.905   1.563   43.504
C   44.773   2.288   42.290
C   46.038   1.575   42.006
C   45.073   3.727   42.608
C   44.144   4.749   42.041
C   44.118   6.056   42.893
C   44.008   7.441   42.027
C   45.258   8.292   42.321
C   42.800   8.323   42.353
H   38.370   1.983   33.286
H   38.457   0.462   26.701
H   32.269   1.741   26.597
H   34.035   3.504   34.631
H   36.642   2.219   35.041
H   35.846   4.952   34.808
H   36.741   4.468   33.364
H   37.483   4.277   34.925
H   34.571   0.642   34.848
H   32.896   1.107   34.688
H   33.871   0.653   36.974
H   32.508   1.667   36.593
H   40.873   -0.462   31.904
H   40.330   0.895   32.998
H   41.474   1.162   31.776
H   42.621   0.678   29.571
H   42.716   -0.561   28.370
H   42.188   -1.087   29.909
H   36.422   -0.071   25.293
H   34.541   2.201   24.942
H   37.860   1.459   24.049
H   37.626   2.736   25.100
H   36.457   2.632   23.873
H   34.332   -0.142   23.877
H   32.945   0.571   24.661
H   34.510   -1.312   26.519
H   34.144   -1.992   24.932
H   32.867   -1.354   26.030
H   29.910   2.029   27.388
H   29.804   3.612   27.967
H   29.153   2.187   28.962
H   31.367   1.675   33.507
H   31.964   4.759   36.286
H   30.248   4.692   36.586
H   30.962   5.784   35.296
H   35.664   4.078   38.956
H   36.767   4.322   37.676
H   38.428   2.961   38.734
H   35.734   1.329   41.616
H   35.058   2.324   40.535
H   35.772   3.072   41.935
H   38.372   2.373   42.376
H   39.283   3.099   41.041
H   39.828   0.779   40.208
H   38.161   0.320   40.776
H   38.923   -0.618   42.496
H   39.775   0.783   43.163
H   41.562   -0.107   41.207
H   39.918   -2.303   41.897
H   41.498   -2.702   42.677
H   41.365   -2.306   40.965
H   42.881   -0.738   43.549
H   41.377   -0.161   44.271
H   41.815   2.223   43.407
H   42.290   1.660   42.069
H   44.356   0.714   44.018
H   43.890   2.207   44.383
H   44.145   2.276   41.399
H   45.855   0.539   41.718
H   46.624   1.699   42.917
H   46.581   2.078   41.206
H   45.935   4.015   42.006
H   45.218   3.980   43.659
H   43.200   4.211   42.131
H   44.362   4.943   40.991
H   44.882   6.023   43.669
H   43.172   6.028   43.435
H   43.973   7.177   40.971
H   44.989   9.101   43.000
H   45.653   8.684   41.383
H   45.991   7.683   42.849
H   41.952   7.869   41.841
H   42.895   9.331   41.949
H   42.821   8.445   43.436

