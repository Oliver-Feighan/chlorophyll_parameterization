%nproc=24
%mem=75GB
#p cam-b3lyp/Def2SVP td=(nstates=5)

step_101_chromophore_15 TDDFT with cam-b3lyp functional

0 1
Mg   47.224   35.103   28.298
C   45.383   33.507   30.729
C   47.534   37.729   30.500
C   48.808   36.910   25.758
C   46.590   32.601   26.093
N   46.665   35.437   30.403
C   45.995   34.583   31.205
C   45.979   35.025   32.655
C   46.417   36.581   32.477
C   46.882   36.601   31.040
C   45.426   37.681   32.774
C   47.044   34.226   33.453
C   47.203   34.505   34.957
C   46.150   35.366   35.789
O   44.947   35.454   35.538
O   46.871   35.968   36.805
N   48.048   36.945   28.172
C   48.081   37.897   29.216
C   48.633   39.090   28.723
C   48.974   38.889   27.354
C   48.630   37.476   27.016
C   49.060   40.293   29.555
C   49.580   39.958   26.413
O   50.150   39.624   25.353
C   49.418   41.374   26.732
N   47.703   34.841   26.290
C   48.363   35.641   25.427
C   48.298   35.108   23.961
C   47.821   33.614   24.151
C   47.315   33.682   25.560
C   47.402   36.077   23.147
C   48.920   32.525   24.010
C   48.510   31.384   23.021
N   46.149   33.363   28.390
C   46.031   32.423   27.354
C   45.258   31.322   28.006
C   44.862   31.753   29.248
C   45.535   32.966   29.478
C   44.978   29.963   27.334
C   44.149   31.362   30.430
O   43.535   30.344   30.720
C   44.360   32.531   31.383
C   44.591   32.020   32.750
O   45.413   31.211   33.111
O   43.700   32.609   33.583
C   43.773   32.105   34.950
C   45.992   36.782   37.654
C   46.889   37.780   38.358
C   46.626   38.599   39.418
C   45.261   38.938   39.982
C   47.826   39.196   40.206
C   48.181   40.609   39.774
C   48.714   41.505   40.893
C   49.501   42.662   40.218
C   51.026   42.341   40.309
C   49.125   44.038   40.947
C   48.107   44.935   40.167
C   46.902   45.224   40.945
C   46.348   46.636   40.587
C   47.115   47.764   41.419
C   44.826   46.680   40.822
C   43.981   46.652   39.585
C   42.663   45.930   39.766
C   41.454   46.524   39.029
C   40.137   46.045   39.667
C   41.503   46.226   37.464
H   47.631   38.575   31.182
H   49.148   37.582   24.967
H   46.453   31.689   25.507
H   44.981   34.853   33.057
H   47.117   36.833   33.274
H   44.432   37.234   32.784
H   45.484   38.510   32.069
H   45.559   38.101   33.771
H   47.931   34.485   32.875
H   47.051   33.140   33.358
H   48.223   34.889   34.969
H   47.168   33.527   35.437
H   50.141   40.381   29.445
H   48.770   40.205   30.602
H   48.433   41.122   29.227
H   49.527   41.946   25.810
H   50.082   41.710   27.529
H   48.381   41.457   27.056
H   49.292   34.979   23.532
H   46.884   33.435   23.623
H   48.151   36.330   22.396
H   46.985   36.908   23.716
H   46.575   35.577   22.642
H   49.238   32.098   24.962
H   49.802   32.964   23.543
H   49.275   31.352   22.245
H   47.542   31.596   22.567
H   48.486   30.444   23.573
H   43.945   29.692   27.551
H   45.667   29.232   27.759
H   45.074   29.996   26.249
H   43.442   33.118   31.349
H   43.748   31.015   34.966
H   42.973   32.551   35.540
H   44.770   32.394   35.284
H   45.556   36.023   38.303
H   45.238   37.402   37.169
H   47.956   37.637   38.189
H   45.359   38.433   40.943
H   44.375   38.546   39.483
H   45.187   40.006   40.184
H   48.731   38.620   40.015
H   47.608   39.209   41.274
H   47.199   41.036   39.569
H   48.824   40.574   38.895
H   49.391   40.909   41.504
H   47.823   41.889   41.391
H   49.311   42.800   39.153
H   51.501   42.718   39.403
H   51.126   41.266   40.158
H   51.556   42.723   41.181
H   50.051   44.610   41.009
H   48.772   43.867   41.964
H   47.730   44.513   39.235
H   48.667   45.843   39.939
H   46.971   45.219   42.033
H   46.116   44.501   40.726
H   46.536   46.807   39.528
H   46.578   48.189   42.267
H   47.197   48.581   40.703
H   48.072   47.538   41.890
H   44.559   47.596   41.349
H   44.464   45.890   41.479
H   44.545   46.209   38.765
H   43.805   47.721   39.464
H   42.408   45.920   40.826
H   42.807   44.950   39.312
H   41.395   47.610   39.104
H   39.731   46.947   40.126
H   40.351   45.269   40.402
H   39.471   45.577   38.943
H   40.462   46.071   37.183
H   42.045   45.310   37.228
H   41.816   47.063   36.840

